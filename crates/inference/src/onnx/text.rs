use super::OnnxRunError;
use super::text_inputs::create_text_inputs;
use ort::session::Session;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub values: Vec<f32>,
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

pub fn run_encoder_output(
    session: &mut Session,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
) -> Result<Vec<f32>, OnnxRunError> {
    let output_name = [
        "last_hidden_state",
        "sentence_embedding",
        "pooler_output",
        "logits",
    ]
    .iter()
    .find(|name| has_output(session, name))
    .copied()
    .ok_or(OnnxRunError::MissingOutput)?;
    let (_, values) = run_text_output(
        session,
        &input_ids,
        Some(&attention_mask),
        Some(&token_type_ids),
        output_name,
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

pub fn run_decoder_output(
    session: &mut Session,
    input_ids: &[i64],
) -> Result<Vec<f32>, OnnxRunError> {
    let output_name = ["logits", "last_hidden_state", "sentence_embedding"]
        .iter()
        .find(|name| has_output(session, name))
        .copied()
        .ok_or(OnnxRunError::MissingOutput)?;
    let (_, values) = run_text_output(session, input_ids, None, None, output_name)?;
    Ok(values)
}

pub fn has_output(session: &Session, name: &str) -> bool {
    session.outputs().iter().any(|output| output.name() == name)
}

pub fn run_text_output(
    session: &mut Session,
    input_ids: &[i64],
    attention_mask: Option<&[i64]>,
    token_type_ids: Option<&[i64]>,
    output_name: &str,
) -> Result<(Vec<i64>, Vec<f32>), OnnxRunError> {
    let input_names = text_input_names(session)?;
    let inputs = create_text_inputs(
        &input_names,
        input_ids,
        attention_mask,
        token_type_ids,
        None,
    )?;
    let outputs = session.run(inputs)?;
    let output = outputs
        .get(output_name)
        .ok_or_else(|| OnnxRunError::MissingNamedOutput(output_name.to_owned()))?;
    let (shape, values) = output.try_extract_tensor::<f32>()?;
    Ok((shape.to_vec(), values.to_vec()))
}

pub fn run_text_outputs(
    session: &mut Session,
    input_ids: &[i64],
    attention_mask: Option<&[i64]>,
    token_type_ids: Option<&[i64]>,
    decoder_input_ids: Option<&[i64]>,
) -> Result<Vec<TensorOutput>, OnnxRunError> {
    let input_names = text_input_names(session)?;
    let inputs = create_text_inputs(
        &input_names,
        input_ids,
        attention_mask,
        token_type_ids,
        decoder_input_ids,
    )?;
    let outputs = session.run(inputs)?;
    outputs
        .iter()
        .map(|(name, output)| {
            let (shape, values) = output.try_extract_tensor::<f32>()?;
            Ok(TensorOutput {
                name: name.to_owned(),
                shape: shape.to_vec(),
                values: values.to_vec(),
            })
        })
        .collect()
}

fn text_input_names(session: &Session) -> Result<Vec<String>, OnnxRunError> {
    let input_names = session
        .inputs()
        .iter()
        .map(|input| input.name().to_owned())
        .collect::<Vec<_>>();
    Ok(input_names)
}
