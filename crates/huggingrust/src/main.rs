mod input;
mod models;
mod user_error;
mod workflow;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (artifact, family) = workflow::load_model(input::prompt_model_input()?).await?;
    let mut inference = models::load_inference(family, artifact)?;

    match family {
        models::InferenceFamily::DecoderOnly => {
            let generated = inference.generate("Once upon a time", 12)?;
            println!("generated text: {generated}");
        }
        models::InferenceFamily::EncoderOnly => {
            let hidden_states = inference.run("Hello from huggingrust!")?;
            println!("hidden state values: {}", hidden_states.len());
        }
        models::InferenceFamily::Unknown => {
            unreachable!("unsupported models are rejected when loading")
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Error: {}", user_error::UserError(error));
        std::process::exit(1);
    }
}
