use std::path::PathBuf;

use inference::{Inference, Model, ModelType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model {
        name: "bge-small-en-v1.5".to_owned(),
        artifact_path: PathBuf::from("onnx_models/bge-small-en-v1.5"),
        model_type: ModelType::SentenceEmbedding,
    };

    let mut inference = Inference::load(model)?;

    let embedding = inference.run("Awesome test sentence to embed!")?;

    println!("Embedding contains {} values", embedding.len());

    Ok(())
}
