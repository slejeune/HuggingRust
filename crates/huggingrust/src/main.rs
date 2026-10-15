mod input;
mod models;
mod user_error;

use inference::{Inference, ModelArtifact};
use input::{ModelInput, prompt_model_input};
use models::Model;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let artifact = match prompt_model_input()? {
        ModelInput::Local(path) => ModelArtifact::from_directory(path),
        ModelInput::Hub {
            author,
            model: model_name,
        } => {
            let client = importer::get_client()?;
            let metadata = importer::get_hub_metadata(&client, &author, &model_name).await?;
            let model = Model::from(metadata.clone());
            model.inference_family.ensure_supported(&model.metadata)?;
            println!("{:?}", model);
            let paths = importer::download_model(&client, &metadata).await?;
            ModelArtifact::new(paths)?
        }
    };

    let mut inference = Inference::load(artifact)?;
    let embedding = inference.run("Hello from huggingrust!")?;

    println!("embedding size: {}", embedding.len());

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Error: {}", user_error::UserError(error));
        std::process::exit(1);
    }
}
