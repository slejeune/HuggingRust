use importer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = importer::get_client()?;
    let metadata = importer::get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;
    println!("{:#?}", &metadata);

    importer::get_hub_model(&client, &metadata).await?;

    Ok(())
}
