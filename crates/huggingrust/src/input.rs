use std::{
    io::{self, Write},
    path::PathBuf,
};

use crate::models::{ModelCapabilities, ModelFamily};

pub enum ModelInput {
    Local(PathBuf),
    Hub { author: String, model: String },
}

pub fn prompt_model_input() -> Result<ModelInput, Box<dyn std::error::Error>> {
    print!("Model path or Hugging Face model: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(parse_model_input(&input)?)
}

pub fn prompt_text(label: &str) -> Result<String, std::io::Error> {
    print!("{label}: ");
    io::stdout().flush()?;
    let mut text = String::new();
    io::stdin().read_line(&mut text)?;
    let text = text.trim().to_owned();
    if text.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Input text cannot be empty",
        ));
    }
    Ok(text)
}

pub fn prompt_task_text(family: ModelFamily, task: &str) -> Result<String, std::io::Error> {
    match (family, task) {
        (ModelFamily::EncoderDecoder, "text2text-generation") => {
            println!("This model accepts free-form text-to-text instructions.");
            println!(
                "Examples: `summarize: ...`, `translate English to German: ...`, or `question: ... context: ...`."
            );
            prompt_text("Instruction, task prefix, and input")
        }
        (ModelFamily::DecoderOnly, "text-generation") => {
            println!("Enter any prompt; the model will continue from it.");
            prompt_text("Prompt")
        }
        _ => prompt_text("Text"),
    }
}

pub fn choose_task(capabilities: &ModelCapabilities) -> Result<String, Box<dyn std::error::Error>> {
    println!("\nModel: {}", capabilities.model_id);
    if let Some(architecture) = &capabilities.architecture {
        println!("Architecture: {architecture}");
    }
    if let Some(pipeline) = &capabilities.pipeline {
        println!("Hugging Face primary task: {pipeline}");
    }
    if !capabilities.task_tags.is_empty() {
        println!(
            "Additional Hugging Face task tags: {}",
            capabilities.task_tags.join(", ")
        );
    }
    if capabilities.tasks.is_empty() {
        return Err(format!(
            "No supported text task could be inferred from the Hugging Face metadata for '{}'",
            capabilities.model_id
        )
        .into());
    }
    if capabilities.tasks.len() == 1 {
        let task = &capabilities.tasks[0];
        println!("Detected task: {} — {}", task.name, task.description);
        return Ok(task.name.to_owned());
    }

    println!("Supported tasks indicated by model metadata:");
    for (index, task) in capabilities.tasks.iter().enumerate() {
        println!("  {}. {} — {}", index + 1, task.name, task.description);
    }
    loop {
        print!("Choose a task [1-{}]: ", capabilities.tasks.len());
        io::stdout().flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if let Ok(choice) = answer.trim().parse::<usize>()
            && let Some(task) = choice
                .checked_sub(1)
                .and_then(|index| capabilities.tasks.get(index))
        {
            return Ok(task.name.to_owned());
        }
        println!("Enter a number from 1 to {}.", capabilities.tasks.len());
    }
}

fn parse_model_input(input: &str) -> Result<ModelInput, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Please enter a model name, Hugging Face URL, or local directory.".into());
    }
    let path = PathBuf::from(input);
    if path.is_dir() {
        return Ok(ModelInput::Local(path));
    }

    let hub_id = if let Some(url) = input
        .strip_prefix("https://huggingface.co/")
        .or_else(|| input.strip_prefix("http://huggingface.co/"))
    {
        let id = url
            .split(['?', '#'])
            .next()
            .ok_or("Invalid Hugging Face URL")?;
        let mut parts = id.split('/');
        if parts.next() == Some("models") {
            parts.collect::<Vec<_>>().join("/")
        } else {
            id.to_owned()
        }
    } else {
        input.to_owned()
    };

    if let Some((author, model)) = hub_id.split_once('/')
        && !author.is_empty()
        && !model.is_empty()
        && !model.contains('/')
    {
        return Ok(ModelInput::Hub {
            author: author.to_owned(),
            model: model.to_owned(),
        });
    }

    Err("Enter a Hugging Face model as `author/model` (or its URL), or a local directory containing config.json, an ONNX model, and tokenizer.json.".into())
}
