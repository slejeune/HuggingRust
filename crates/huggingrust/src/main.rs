use metadata;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = metadata::get_client()?;
    let metadata = metadata::get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;
    println!("{:#?}", &metadata);

    metadata::get_hub_model(&client, &metadata).await?;

    Ok(())
}
