use std::path::Path;

use ort::{session::Session, value::Tensor};

use crate::models::{Model, ModelType};

pub fn load_model(model: &Model) -> ort::Result<Session> {
    Session::builder()?.commit_from_file(model.onnx_path())
}

pub fn run_sentence_embedding(
    session: &mut Session,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
    sequence_length: usize,
) -> ort::Result<Vec<f32>> {
    let input_ids = Tensor::from_array(([1, sequence_length], input_ids))?;

    let attention_mask = Tensor::from_array(([1, sequence_length], attention_mask))?;

    let token_type_ids = Tensor::from_array(([1, sequence_length], token_type_ids))?;

    let outputs = session.run(ort::inputs![
        "input_ids" => input_ids,
        "attention_mask" => attention_mask,
        "token_type_ids" => token_type_ids,
    ])?;

    let output = outputs
        .get("last_hidden_state")
        .expect("model did not produce last_hidden_state");

    let (_, values) = output.try_extract_tensor::<f32>()?;

    Ok(values.to_vec())
}

pub fn model_type_supported(model: &Model) -> bool {
    matches!(model.model_type, ModelType::SentenceEmbedding)
}

pub fn model_exists(model: &Model) -> bool {
    Path::new(&model.onnx_path()).is_file()
}
