use hf_hub::HFClient;

use crate::models::ModelMetadata;

pub async fn get_hub_metadata(author: &str, model: &str) -> hf_hub::HFResult<ModelMetadata> {
    let client = HFClient::new()?;
    let info = client.model(author, model).info().send().await?;

    Ok(ModelMetadata::from_huggingface(&info))
}
