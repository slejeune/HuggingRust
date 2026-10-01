use inference::{Inference, ModelArtifact};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = importer::get_client()?;
    let metadata = importer::get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;

    let paths = importer::download_model(&client, &metadata).await?;

    let artifact_path = paths
        .first()
        .and_then(|path| path.parent())
        .ok_or("model download returned no files")?
        .to_path_buf();

    let artifact = ModelArtifact {
        metadata,
        artifact_path,
    };

    let mut inference = Inference::load(artifact)?;
    let embedding = inference.run("Hello from huggingrust")?;

    println!("embedding size: {}", embedding.len());

    Ok(())
}
