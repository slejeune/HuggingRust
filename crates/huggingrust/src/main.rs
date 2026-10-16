mod input;
mod models;
mod user_error;
mod workflow;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (artifact, family) = workflow::load_model(input::prompt_model_input()?).await?;
    let mut inference = models::load_inference(family, artifact)?;

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
