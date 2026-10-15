mod models;

use inference::{Inference, ModelArtifact};
use models::Model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = importer::get_client()?;
    let metadata = importer::get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;

    let paths = importer::download_model(&client, &metadata).await?;
    let model = Model::from(metadata.clone());
    println!("{:?}", model);
    let artifact = ModelArtifact::new(metadata, paths);

    let mut inference = Inference::load(artifact)?;
    let embedding = inference.run("Hello from huggingrust")?;

    println!("embedding size: {}", embedding.len());

    Ok(())
}
