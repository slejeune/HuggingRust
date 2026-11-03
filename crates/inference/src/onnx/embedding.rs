use super::{OnnxRunError, text::TensorOutput};

pub fn last_token_embedding(
    output: &TensorOutput,
    attention_mask: &[u32],
) -> Result<Vec<f32>, OnnxRunError> {
    if output.shape.len() != 3 || output.shape[0] != 1 {
        return Err(OnnxRunError::UnsupportedSignature(format!(
            "cannot pool tensor '{}' with shape {:?}",
            output.name, output.shape
        )));
    }
    let sequence_length = output.shape[1] as usize;
    let hidden_size = output.shape[2] as usize;
    let Some(last_token) = attention_mask.iter().rposition(|&mask| mask != 0) else {
        return Err(OnnxRunError::MissingOutput);
    };
    if last_token >= sequence_length || output.values.len() < sequence_length * hidden_size {
        return Err(OnnxRunError::MissingOutput);
    }
    let start = last_token * hidden_size;
    let mut embedding = output.values[start..start + hidden_size].to_vec();
    normalize(&mut embedding);
    Ok(embedding)
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter_mut().for_each(|value| *value /= norm);
    }
}
