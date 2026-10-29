use crate::ModelArtifact;
use ort::{session::Session, value::Tensor};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OnnxRunError {
    #[error("ONNX Runtime error: {0}")]
    Runtime(#[from] ort::Error),
    #[error("ONNX model did not return the required output tensor")]
    MissingOutput,
    #[error("ONNX model did not return the required '{0}' output")]
    MissingNamedOutput(String),
    #[error("Unsupported ONNX input signature: {0}")]
    UnsupportedSignature(String),
}

pub fn load_model(model: &ModelArtifact) -> ort::Result<Session> {
    Session::builder()?.commit_from_file(model.onnx_path())
}

pub fn run_encoder_hidden_states(
    session: &mut Session,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
) -> Result<Vec<f32>, OnnxRunError> {
    let (_, values) = run_text_output(
        session,
        &input_ids,
        Some(&attention_mask),
        Some(&token_type_ids),
        "last_hidden_state",
    )?;
    Ok(values)
}

pub fn run_decoder_logits(
    session: &mut Session,
    input_ids: &[i64],
) -> Result<Vec<f32>, OnnxRunError> {
    let (shape, values) = run_text_output(session, input_ids, None, None, "logits")?;
    let vocabulary_size = *shape.last().ok_or(OnnxRunError::MissingOutput)? as usize;
    if vocabulary_size == 0 || values.len() < vocabulary_size {
        return Err(OnnxRunError::MissingOutput);
    }
    Ok(values[values.len() - vocabulary_size..].to_vec())
}

pub fn run_text_output(
    session: &mut Session,
    input_ids: &[i64],
    attention_mask: Option<&[i64]>,
    token_type_ids: Option<&[i64]>,
    output_name: &str,
) -> Result<(Vec<i64>, Vec<f32>), OnnxRunError> {
    let sequence_length = input_ids.len();
    let input_names: Vec<_> = session.inputs().iter().map(|input| input.name()).collect();
    let known = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
    ];
    if let Some(name) = input_names.iter().find(|name| !known.contains(name)) {
        return Err(OnnxRunError::UnsupportedSignature((*name).to_owned()));
    }
    if !input_names.contains(&"input_ids") {
        return Err(OnnxRunError::UnsupportedSignature(
            "missing input_ids".into(),
        ));
    }

    let ids = Tensor::from_array(([1, sequence_length], input_ids.to_vec()))?;
    let mut inputs = ort::inputs!["input_ids" => ids];
    if input_names.contains(&"attention_mask") {
        let values = attention_mask
            .map(<[i64]>::to_vec)
            .unwrap_or_else(|| vec![1; sequence_length]);
        let mask = Tensor::from_array(([1, sequence_length], values))?;
        inputs.push(("attention_mask".into(), mask.into()));
    }
    if input_names.contains(&"token_type_ids") {
        let values = token_type_ids
            .map(<[i64]>::to_vec)
            .unwrap_or_else(|| vec![0; sequence_length]);
        let types = Tensor::from_array(([1, sequence_length], values))?;
        inputs.push(("token_type_ids".into(), types.into()));
    }
    if input_names.contains(&"position_ids") {
        let values = (0..sequence_length)
            .map(|position| position as i64)
            .collect::<Vec<_>>();
        let position_ids = Tensor::from_array(([1, sequence_length], values))?;
        inputs.push(("position_ids".into(), position_ids.into()));
    }
    let outputs = session.run(inputs)?;

    let output = outputs
        .get(output_name)
        .ok_or_else(|| OnnxRunError::MissingNamedOutput(output_name.to_owned()))?;
    let (shape, values) = output.try_extract_tensor::<f32>()?;
    Ok((shape.to_vec(), values.to_vec()))
}
