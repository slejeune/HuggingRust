use crate::InferenceError;

pub(crate) fn greedy_generate<F>(
    input_ids: &mut Vec<i64>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    mut next_token_logits: F,
) -> Result<Vec<u32>, InferenceError>
where
    F: FnMut(&[i64]) -> Result<Vec<f32>, InferenceError>,
{
    let mut generated = Vec::new();
    for _ in 0..max_new_tokens {
        let logits = next_token_logits(input_ids)?;
        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or(crate::onnx::OnnxRunError::MissingOutput)?;
        if eos_token_ids.contains(&next_token) {
            break;
        }
        input_ids.push(i64::from(next_token));
        generated.push(next_token);
    }
    Ok(generated)
}
