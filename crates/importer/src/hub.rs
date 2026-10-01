use std::path::{Path, PathBuf};

use hf_hub::HFClient;

use crate::{models::ModelMetadata, progress::spinner};

pub fn get_client() -> hf_hub::HFResult<HFClient> {
    Ok(HFClient::new()?)
}

pub async fn get_hub_metadata(
    client: &HFClient,
    author: &str,
    model: &str,
) -> hf_hub::HFResult<ModelMetadata> {
    let info = client.model(author, model).info().send().await?;

    Ok(ModelMetadata::from_huggingface(&info))
}

pub async fn download_model(
    client: &HFClient,
    metadata: &ModelMetadata,
) -> hf_hub::HFResult<PathBuf> {
    if !metadata.has_onnx() {
        return Err(hf_hub::HFError::malformed_response(
            "No ONNX model available",
        ));
    }

    let dir = PathBuf::from("onnx_models").join(&metadata.model);
    let files: Vec<_> = metadata.inference_files().collect();

    tokio::fs::create_dir_all(&dir).await?;

    let pb = spinner(format!("Checking {}...", metadata.model));

    let mut missing = Vec::new();

    for file in &files {
        let destination = destination(&dir, file)?;

        if !is_valid_file(&destination).await {
            missing.push((*file, destination));
        }
    }

    if missing.is_empty() {
        pb.finish_with_message(format!("✓ {} is ready", metadata.model));
        return Ok(dir);
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
                .await?
        } else {
            repo.download_file().filename(file).send().await?
        };

        tokio::fs::copy(&cached, &destination).await?;
    }

    pb.finish_with_message(format!("✓ {} is ready", metadata.model));

    Ok(dir)
}

fn destination(dir: &Path, file: &str) -> hf_hub::HFResult<PathBuf> {
    let filename = Path::new(file)
        .file_name()
        .ok_or_else(|| hf_hub::HFError::malformed_response("Invalid model filename"))?;

    Ok(dir.join(filename))
}

async fn is_valid_file(path: &Path) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}
