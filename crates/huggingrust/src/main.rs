use metadata::get_hub_metadata;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let metadata = get_hub_metadata("BAAI", "bge-small-en-v1.5").await?;
    println!("{:#?}", metadata);

    Ok(())
}
