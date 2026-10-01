use metadata::{get_client, get_hub_metadata};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = get_client()?;
    let metadata = get_hub_metadata(&client, "BAAI", "bge-small-en-v1.5").await?;
    println!("{:#?}", metadata);

    Ok(())
}
