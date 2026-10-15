mod models;
mod user_error;

use inference::{Inference, ModelArtifact};
use models::Model;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let client = importer::get_client()?;
    let metadata = importer::get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;

    let model = Model::from(metadata.clone());
    model.inference_family.ensure_supported(&model.metadata)?;
    println!("{:?}", model);

    let paths = importer::download_model(&client, &metadata).await?;
    let artifact = ModelArtifact::new(paths)?;

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
