use std::path::{Path, PathBuf};

use hf_hub::HFClient;
use thiserror::Error;

use crate::{models::ModelMetadata, progress::spinner};

#[derive(Debug, Error)]
pub enum ImportError {
    #[error("Hugging Face Hub request failed: {0}")]
    Hub(#[from] hf_hub::HFError),
    #[error("Failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Invalid config.json for model {model}: {source}")]
    InvalidConfig {
        model: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("Invalid model metadata: {0}")]
    Metadata(#[from] crate::models::MetadataError),
    #[error("Model {model} has no ONNX file. Choose a repository with an exported ONNX model.")]
    NoOnnx { model: String },
    #[error("Invalid file path {file:?} in model {model}")]
    InvalidFilename { model: String, file: String },
    #[error("Model file name collision at {path}; repository files would overwrite each other")]
    FileCollision { path: PathBuf },
    #[error("Failed to create model directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Failed to configure download progress: {0}")]
    Progress(#[from] indicatif::style::TemplateError),
    #[error("Failed to download {file} for model {model}: {source}")]
    DownloadFile {
        model: String,
        file: String,
        #[source]
        source: hf_hub::HFError,
    },
    #[error("Failed to copy {file} for model {model} to {destination}: {source}")]
    CopyFile {
        model: String,
        file: String,
        destination: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub fn get_client() -> hf_hub::HFResult<HFClient> {
    Ok(HFClient::new()?)
}

pub async fn get_hub_metadata(
    client: &HFClient,
    author: &str,
    model: &str,
) -> Result<ModelMetadata, ImportError> {
    let info = client.model(author, model).info().send().await?;
    let mut metadata = ModelMetadata::from_huggingface(&info)?;

    if metadata.files.iter().any(|file| file.path == "config.json") {
        let repo = client.model(author, model);
        let config = if let Some(revision) = &metadata.revision {
            repo.download_file()
                .filename("config.json")
                .revision(revision)
                .send()
                .await?
        } else {
            repo.download_file().filename("config.json").send().await?
        };

        let bytes = tokio::fs::read(&config)
            .await
            .map_err(|source| ImportError::Read {
                path: config,
                source,
            })?;
        let config: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|source| ImportError::InvalidConfig {
                model: metadata.id.clone(),
                source,
            })?;
        metadata.model_type = config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        metadata.architectures = config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_owned)
            .collect();
    }

    Ok(metadata)
}

pub async fn download_model(
    client: &HFClient,
    metadata: &ModelMetadata,
) -> Result<Vec<PathBuf>, ImportError> {
    if !metadata.has_onnx() {
        return Err(ImportError::NoOnnx {
            model: metadata.id.clone(),
        });
    }

    let dir = PathBuf::from("onnx_models").join(&metadata.model);
    let mut files: Vec<_> = metadata
        .inference_files()
        .map(|file| file.path.as_str())
        .collect();

    for (required, predicate) in [("model.onnx", 0u8), ("tokenizer.json", 1u8)] {
        if !files.iter().any(|file| *file == required) {
            let candidate = metadata.files.iter().find(|file| match predicate {
                0 => file.path.ends_with(".onnx"),
                _ => file.path.ends_with("tokenizer.json"),
            });
            if let Some(candidate) = candidate {
                files.retain(|file| *file != candidate.path);
                files.push(candidate.path.as_str());
            }
        }
    }

    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|source| ImportError::CreateDir {
            path: dir.clone(),
            source,
        })?;

    let pb = spinner(format!("Checking {}...", metadata.model))?;

    let mut paths = Vec::new();
    let mut missing = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for file in &files {
        let destination = destination(&dir, file)?;
        if !seen.insert(destination.clone()) {
            pb.finish_with_message(format!("✗ {} has colliding file names", metadata.model));
            return Err(ImportError::FileCollision { path: destination });
        }

        if is_valid_file(&destination).await {
            paths.push(destination);
        } else {
            missing.push((*file, destination));
        }
    }

    if missing.is_empty() {
        pb.finish_with_message(format!("✓ {} is ready", metadata.model));
        return Ok(paths);
    }

    pb.set_message(format!(
        "Downloading {} ({} files)...",
        metadata.model,
        missing.len()
    ));

    let repo = client.model(&metadata.author, &metadata.model);

    for (file, destination) in missing {
        pb.set_message(format!("Downloading {file}..."));

        let cached = if let Some(revision) = &metadata.revision {
            repo.download_file()
                .filename(file)
                .revision(revision)
                .send()
                .await
        } else {
            repo.download_file().filename(file).send().await
        }
        .map_err(|source| ImportError::DownloadFile {
            model: metadata.id.clone(),
            file: file.to_string(),
            source,
        })?;

        let final_destination = if file.ends_with(".onnx") {
            dir.join("model.onnx")
        } else {
            destination.clone()
        };
        if let Some(parent) = final_destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| ImportError::CreateDir {
                    path: parent.to_path_buf(),
                    source,
                })?;
        }
        let filename = final_destination
            .file_name()
            .ok_or_else(|| ImportError::InvalidFilename {
                model: metadata.id.clone(),
                file: file.to_string(),
            })?
            .to_string_lossy();
        let temp = final_destination.with_file_name(format!("{filename}.download-part"));
        if let Err(source) = tokio::fs::copy(&cached, &temp).await {
            let _ = tokio::fs::remove_file(&temp).await;
            pb.finish_with_message(format!("✗ Failed to download {}", metadata.model));
            return Err(ImportError::CopyFile {
                model: metadata.id.clone(),
                file: file.to_string(),
                destination: final_destination,
                source,
            });
        }
        if !is_valid_file(&temp).await {
            let _ = tokio::fs::remove_file(&temp).await;
            pb.finish_with_message(format!("✗ Invalid file for {}", metadata.model));
            return Err(ImportError::CopyFile {
                model: metadata.id.clone(),
                file: file.to_string(),
                destination: final_destination,
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "downloaded file is empty or not a regular file",
                ),
            });
        }
        if let Err(source) = tokio::fs::rename(&temp, &final_destination).await {
            let _ = tokio::fs::remove_file(&temp).await;
            pb.finish_with_message(format!("✗ Failed to save {}", metadata.model));
            return Err(ImportError::CopyFile {
                model: metadata.id.clone(),
                file: file.to_string(),
                destination: final_destination,
                source,
            });
        }
        paths.push(final_destination);
    }

    pb.finish_with_message(format!("✓ {} is ready", metadata.model));

    Ok(paths)
}

fn destination(dir: &Path, file: &str) -> Result<PathBuf, ImportError> {
    let relative = Path::new(file);
    let is_safe = !relative.as_os_str().is_empty()
        && !relative.is_absolute()
        && relative
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)));
    is_safe
        .then(|| dir.join(relative))
        .ok_or_else(|| ImportError::InvalidFilename {
            model: dir.display().to_string(),
            file: file.to_owned(),
        })
}

async fn is_valid_file(path: &Path) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}
