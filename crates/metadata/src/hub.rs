use hf_hub::HFClient;

use crate::models::ModelMetadata;

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
