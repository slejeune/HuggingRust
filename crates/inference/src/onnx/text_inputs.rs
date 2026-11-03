use super::OnnxRunError;
use ort::value::Tensor;

pub(super) fn create_text_inputs(
    input_names: &[String],
    input_ids: &[i64],
    attention_mask: Option<&[i64]>,
    token_type_ids: Option<&[i64]>,
    decoder_input_ids: Option<&[i64]>,
) -> Result<
    Vec<(
        std::borrow::Cow<'static, str>,
        ort::session::SessionInputValue<'static>,
    )>,
    OnnxRunError,
> {
    validate_input_names(input_names)?;
    let mut inputs = ort::inputs![
        "input_ids" => Tensor::from_array(([1, input_ids.len()], input_ids.to_vec()))?
    ];
    for name in input_names
        .iter()
        .filter(|name| name.as_str() != "input_ids")
    {
        let (sequence_length, values) = match name.as_str() {
            "attention_mask" => (
                input_ids.len(),
                attention_mask.map_or_else(|| vec![1; input_ids.len()], <[i64]>::to_vec),
            ),
            "token_type_ids" => (
                input_ids.len(),
                token_type_ids.map_or_else(|| vec![0; input_ids.len()], <[i64]>::to_vec),
            ),
            "position_ids" => (input_ids.len(), (0..input_ids.len() as i64).collect()),
            "decoder_input_ids" => {
                let ids = decoder_input_ids.ok_or_else(|| {
                    OnnxRunError::UnsupportedSignature("missing decoder_input_ids".into())
                })?;
                (ids.len(), ids.to_vec())
            }
            "decoder_attention_mask" => {
                let ids = decoder_input_ids.ok_or_else(|| {
                    OnnxRunError::UnsupportedSignature("missing decoder_input_ids".into())
                })?;
                (ids.len(), vec![1; ids.len()])
            }
            _ => return Err(OnnxRunError::UnsupportedSignature(name.clone())),
        };
        inputs.push((
            name.clone().into(),
            Tensor::from_array(([1, sequence_length], values))?.into(),
        ));
    }
    Ok(inputs)
}

fn validate_input_names(input_names: &[String]) -> Result<(), OnnxRunError> {
    const SUPPORTED: &[&str] = &[
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "position_ids",
        "decoder_input_ids",
        "decoder_attention_mask",
    ];
    if let Some(name) = input_names
        .iter()
        .find(|name| !SUPPORTED.contains(&name.as_str()))
    {
        return Err(OnnxRunError::UnsupportedSignature(name.clone()));
    }
    if !input_names.iter().any(|name| name == "input_ids") {
        return Err(OnnxRunError::UnsupportedSignature(
            "missing input_ids".into(),
        ));
    }
    Ok(())
}
