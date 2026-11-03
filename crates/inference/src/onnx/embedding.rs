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

pub fn mean_embedding(
    output: &TensorOutput,
    attention_mask: &[u32],
) -> Result<Vec<f32>, OnnxRunError> {
    if output.shape.len() != 3 || output.shape[0] != 1 {
        return Err(OnnxRunError::UnsupportedSignature(format!(
            "cannot mean-pool tensor '{}' with shape {:?}",
            output.name, output.shape
        )));
    }
    let sequence_length = output.shape[1] as usize;
    let hidden_size = output.shape[2] as usize;
    if sequence_length == 0
        || hidden_size == 0
        || output.values.len() != sequence_length * hidden_size
    {
        return Err(OnnxRunError::MissingOutput);
    }
    let mut embedding = vec![0.0; hidden_size];
    let mut token_count = 0usize;
    for (index, &mask) in attention_mask.iter().take(sequence_length).enumerate() {
        if mask == 0 {
            continue;
        }
        token_count += 1;
        let start = index * hidden_size;
        for (sum, value) in embedding
            .iter_mut()
            .zip(&output.values[start..start + hidden_size])
        {
            *sum += value;
        }
    }
    if token_count == 0 {
        return Err(OnnxRunError::MissingOutput);
    }
    embedding
        .iter_mut()
        .for_each(|value| *value /= token_count as f32);
    normalize(&mut embedding);
    Ok(embedding)
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        values.iter_mut().for_each(|value| *value /= norm);
    }
}
