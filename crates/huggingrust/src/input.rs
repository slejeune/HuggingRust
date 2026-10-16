use std::{
    io::{self, Write},
    path::PathBuf,
};

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

fn parse_model_input(input: &str) -> Result<ModelInput, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Please enter a model name, Hugging Face URL, or local directory.".into());
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

    if let Some((author, model)) = hub_id.split_once('/') {
        if !author.is_empty() && !model.is_empty() && !model.contains('/') {
            return Ok(ModelInput::Hub {
                author: author.to_owned(),
                model: model.to_owned(),
            });
        }
    }

    let path = PathBuf::from(input);
    if path.is_dir() {
        return Ok(ModelInput::Local(path));
    }
    Err("Enter a Hugging Face model as `author/model` (or its URL), or a local directory containing model.onnx and tokenizer.json.".into())
}
