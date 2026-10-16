use inference::ModelArtifact;

use crate::{input::ModelInput, models};

pub async fn load_model(
    input: ModelInput,
) -> Result<(ModelArtifact, models::InferenceFamily), Box<dyn std::error::Error>> {
    match input {
        ModelInput::Local(path) => Ok((
            ModelArtifact::from_directory(path),
            models::InferenceFamily::EncoderOnly,
        )),
        ModelInput::Hub { author, model } => {
            let client = importer::get_client()?;
            let metadata = importer::get_hub_metadata(&client, &author, &model).await?;
            let family = models::ensure_supported(&metadata)?;
            let files = importer::download_model(&client, &metadata).await?;
            Ok((ModelArtifact::new(files)?, family))
        }
    }
}
