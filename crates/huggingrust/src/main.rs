use inference::{Inference, ModelArtifact};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = importer::get_client()?;
    let metadata = importer::get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;

    let artifact_path = importer::download_model(&client, &metadata).await?;

    let artifact = ModelArtifact { artifact_path };
    let mut inference = Inference::load(artifact)?;
    let embedding = inference.run("Hello from huggingrust")?;

    println!("embedding size: {}", embedding.len());

    Ok(())
}
