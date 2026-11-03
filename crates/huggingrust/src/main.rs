mod input;
mod models;
mod user_error;
mod workflow;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (artifact, family, task) = workflow::load_model(input::prompt_model_input()?).await?;
    let mut inference = models::load_inference(family, artifact)?;

    if let Some(task) = task.as_deref() {
        let output = if matches!(task, "question-answering" | "zero-shot-classification") {
            let first = input::prompt_text("Question or text")?;
            let second = input::prompt_text("Context or candidate")?;
            inference.run_pair_task(task, &first, &second)?
        } else {
            let text = input::prompt_text("Text")?;
            inference.run_task(Some(task), &text, 32)?
        };
        print_task_output(output);
    } else {
        match family {
            models::InferenceFamily::DecoderOnly => {
                let generated = inference.generate("Once upon a time", 12)?;
                println!("generated text: {generated}");
            }
            models::InferenceFamily::EncoderDecoder => {
                let generated =
                    inference.generate("Translate to French: Hello from huggingrust!", 32)?;
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
    }

    Ok(())
}

fn print_task_output(output: inference::TaskOutput) {
    match output {
        inference::TaskOutput::GeneratedText(text) => println!("generated text: {text}"),
        inference::TaskOutput::Embedding(values) => {
            println!("embedding values: {}", values.len())
        }
        inference::TaskOutput::Tensors(outputs) => {
            for output in outputs {
                println!(
                    "{} {:?} ({} values)",
                    output.name,
                    output.shape,
                    output.values.len()
                );
            }
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Error: {}", user_error::UserError(error));
        std::process::exit(1);
    }
}
