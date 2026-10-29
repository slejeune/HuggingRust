use std::path::PathBuf;

use hf_hub::HFClient;

use crate::{file_store, hub::ImportError, models::ModelMetadata, progress::spinner};

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
    if !files.iter().any(|file| file.ends_with("tokenizer.json"))
        && let Some(file) = metadata
            .files
            .iter()
            .find(|file| file.path.ends_with("tokenizer.json"))
    {
        files.push(file.path.as_str());
    }

    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|source| ImportError::CreateDir {
            path: dir.clone(),
            source,
        })?;
    let pb = spinner(format!("Checking {}...", metadata.model))?;
    let (mut paths, missing) = file_store::existing_files(&dir, &files, metadata, &pb).await?;
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
    for (file, path) in missing {
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
            file: file.to_owned(),
            source,
        })?;

        if let Err(error) = file_store::install_file(&cached, &path, &metadata.id, file).await {
            pb.finish_with_message(format!("✗ Failed to save {}", metadata.model));
            return Err(error);
        }
        paths.push(path);
    }
    pb.finish_with_message(format!("✓ {} is ready", metadata.model));
    Ok(paths)
}
