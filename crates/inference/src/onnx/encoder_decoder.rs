use super::{OnnxRunError, text::run_text_output};
use ort::{session::Session, value::Tensor};

pub fn run_encoder_decoder_logits(
    session: &mut Session,
    input_ids: &[i64],
    decoder_input_ids: &[i64],
) -> Result<Vec<f32>, OnnxRunError> {
    let inputs = session
        .inputs()
        .iter()
        .map(|input| input.name())
        .collect::<Vec<_>>();
    validate_decoder_inputs(&inputs)?;

    let mut values = ort::inputs![
        "input_ids" => Tensor::from_array(([1, input_ids.len()], input_ids.to_vec()))?,
        "decoder_input_ids" => Tensor::from_array(([1, decoder_input_ids.len()], decoder_input_ids.to_vec()))?
    ];
    if inputs.contains(&"attention_mask") {
        values.push((
            "attention_mask".into(),
            Tensor::from_array(([1, input_ids.len()], vec![1i64; input_ids.len()]))?.into(),
        ));
    }
    if inputs.contains(&"decoder_attention_mask") {
        values.push((
            "decoder_attention_mask".into(),
            Tensor::from_array((
                [1, decoder_input_ids.len()],
                vec![1i64; decoder_input_ids.len()],
            ))?
            .into(),
        ));
    }
    let outputs = session.run(values)?;
    last_token_logits(&outputs)
}

pub fn run_split_encoder_decoder_logits(
    encoder: &mut Session,
    decoder: &mut Session,
    input_ids: &[i64],
    decoder_input_ids: &[i64],
) -> Result<Vec<f32>, OnnxRunError> {
    let (hidden_shape, hidden_values) =
        run_text_output(encoder, input_ids, None, None, "last_hidden_state")?;
    let decoder_inputs = decoder
        .inputs()
        .iter()
        .map(|input| input.name())
        .collect::<Vec<_>>();
    let token_input = if decoder_inputs.contains(&"decoder_input_ids") {
        "decoder_input_ids"
    } else {
        "input_ids"
    };
    let token_ids = Tensor::from_array(([1, decoder_input_ids.len()], decoder_input_ids.to_vec()))?;
    let hidden_states = Tensor::from_array((hidden_shape, hidden_values))?;
    let mut values =
        ort::inputs![token_input => token_ids, "encoder_hidden_states" => hidden_states];
    if decoder_inputs.contains(&"encoder_attention_mask") {
        values.push((
            "encoder_attention_mask".into(),
            Tensor::from_array(([1, input_ids.len()], vec![1i64; input_ids.len()]))?.into(),
        ));
    }
    if decoder_inputs.contains(&"decoder_attention_mask") {
        values.push((
            "decoder_attention_mask".into(),
            Tensor::from_array((
                [1, decoder_input_ids.len()],
                vec![1i64; decoder_input_ids.len()],
            ))?
            .into(),
        ));
    }
    let outputs = decoder.run(values)?;
    last_token_logits(&outputs)
}

fn validate_decoder_inputs(inputs: &[&str]) -> Result<(), OnnxRunError> {
    const SUPPORTED: &[&str] = &[
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ];
    if let Some(name) = inputs.iter().find(|name| !SUPPORTED.contains(name)) {
        return Err(OnnxRunError::UnsupportedSignature((*name).to_owned()));
    }
    for required in ["input_ids", "decoder_input_ids"] {
        if !inputs.contains(&required) {
            return Err(OnnxRunError::UnsupportedSignature(format!(
                "missing {required}"
            )));
        }
    }
    Ok(())
}

fn last_token_logits(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<f32>, OnnxRunError> {
    let output = outputs
        .get("logits")
        .ok_or_else(|| OnnxRunError::MissingNamedOutput("logits".into()))?;
    let (shape, values) = output.try_extract_tensor::<f32>()?;
    let vocabulary_size = *shape.last().ok_or(OnnxRunError::MissingOutput)? as usize;
    if vocabulary_size == 0 || values.len() < vocabulary_size {
        return Err(OnnxRunError::MissingOutput);
    }
    Ok(values[values.len() - vocabulary_size..].to_vec())
}
