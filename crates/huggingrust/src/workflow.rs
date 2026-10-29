use inference::ModelArtifact;

use crate::{input::ModelInput, models};

pub async fn load_model(
    input: ModelInput,
) -> Result<(ModelArtifact, models::InferenceFamily), Box<dyn std::error::Error>> {
    match input {
        ModelInput::Local(path) => load_local_model(path),
        ModelInput::Hub { author, model } => load_hub_model(&author, &model).await,
    }
}

fn load_local_model(
    path: std::path::PathBuf,
) -> Result<(ModelArtifact, models::InferenceFamily), Box<dyn std::error::Error>> {
    let config_path = path.join("config.json");
    if !config_path.is_file() {
        return Err(format!(
            "Cannot determine model family for local model '{}': config.json is required",
            path.display()
        )
        .into());
    }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let model_type = local_model_type(&config, &config_path)?;
    let family = models::inference_family_from_model_type(Some(model_type));
    if family == models::InferenceFamily::Unknown {
        return Err(format!(
            "Unsupported local model_type '{}' in '{}'",
            model_type,
            config_path.display()
        )
        .into());
    }
    models::ensure_supported_task(
        path.to_string_lossy().as_ref(),
        family,
        config
            .get("pipeline_tag")
            .and_then(serde_json::Value::as_str),
    )?;
    Ok((ModelArtifact::from_directory(path), family))
}

fn local_model_type<'a>(
    config: &'a serde_json::Value,
    config_path: &std::path::Path,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    let value = config.get("model_type").ok_or_else(|| {
        format!(
            "Local model config '{}' is missing the required 'model_type' field",
            config_path.display()
        )
    })?;
    value.as_str().ok_or_else(|| {
        format!(
            "Invalid 'model_type' in '{}': expected a string, found {}",
            config_path.display(),
            value
        )
        .into()
    })
}

async fn load_hub_model(
    author: &str,
    model: &str,
) -> Result<(ModelArtifact, models::InferenceFamily), Box<dyn std::error::Error>> {
    let client = importer::get_client()?;
    let metadata = importer::get_hub_metadata(&client, author, model).await?;
    let family = models::ensure_supported(&metadata)?;
    models::ensure_supported_task(&metadata.id, family, metadata.pipeline.as_deref())?;
    let files = importer::download_model(&client, &metadata).await?;
    Ok((ModelArtifact::new(files)?, family))
}
