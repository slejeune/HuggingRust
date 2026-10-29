use std::path::PathBuf;

use hf_hub::HFClient;
use thiserror::Error;

use crate::models::ModelMetadata;

pub use crate::download::download_model;

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
    HFClient::new()
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
