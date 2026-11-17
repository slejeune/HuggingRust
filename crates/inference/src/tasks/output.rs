use super::Task;
use crate::{InferenceError, onnx};

pub(super) fn tensor_output<'a>(
    outputs: &'a [onnx::TensorOutput],
    task: Task,
    names: &[&str],
) -> Result<&'a onnx::TensorOutput, InferenceError> {
    outputs
        .iter()
        .find(|output| names.contains(&output.name.as_str()))
        .ok_or_else(|| InferenceError::TaskOutputMissing {
            task: task.name().to_owned(),
            expected: names.join(" or "),
        })
}

pub(super) fn vector_output(
    outputs: &[onnx::TensorOutput],
    task: Task,
    names: &[&str],
) -> Result<Vec<f32>, InferenceError> {
    let output = tensor_output(outputs, task, names)?;
    if output.values.is_empty()
        || output.shape.len() > 2
        || output.shape.first().is_some_and(|&size| size != 1)
        || output.values.iter().any(|value| !value.is_finite())
    {
        return Err(InferenceError::TaskOutputInvalid {
            task: task.name().to_owned(),
            detail: format!(
                "expected one score vector; received shape {:?}",
                output.shape
            ),
        });
    }
    Ok(output.values.clone())
}

pub(super) fn sequence_scores(
    output: &onnx::TensorOutput,
    task: Task,
) -> Result<Vec<Vec<f32>>, InferenceError> {
    if output.shape.len() != 3 || output.shape[0] != 1 {
        return Err(InferenceError::TaskOutputInvalid {
            task: task.name().to_owned(),
            detail: format!(
                "expected [1, sequence, classes] scores; received {:?}",
                output.shape
            ),
        });
    }
    let sequence_length = output.shape[1] as usize;
    let class_count = output.shape[2] as usize;
    if sequence_length == 0
        || class_count == 0
        || output.values.len() != sequence_length * class_count
        || output.values.iter().any(|value| !value.is_finite())
    {
        return Err(InferenceError::TaskOutputInvalid {
            task: task.name().to_owned(),
            detail: "score tensor shape does not match its values".into(),
        });
    }
    Ok(output
        .values
        .chunks_exact(class_count)
        .map(<[f32]>::to_vec)
        .collect())
}

pub(super) fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(index, _)| index)
}

pub(super) fn best_span(start: &[f32], end: &[f32]) -> Option<(usize, usize)> {
    if start.is_empty() || start.len() != end.len() {
        return None;
    }
    let mut best = (0, 0, f32::NEG_INFINITY);
    for (start_index, &start_score) in start.iter().enumerate() {
        for (end_index, &end_score) in end.iter().enumerate().skip(start_index).take(30) {
            let score = start_score + end_score;
            if score > best.2 {
                best = (start_index, end_index, score);
            }
        }
    }
    Some((best.0, best.1))
}

pub(super) fn softmax_candidates(candidates: Vec<(String, f32)>) -> Vec<(String, f32)> {
    let max = candidates
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::NEG_INFINITY, f32::max);
    let denominator = candidates
        .iter()
        .map(|(_, score)| (*score - max).exp())
        .sum::<f32>();
    candidates
        .into_iter()
        .map(|(label, score)| (label, (score - max).exp() / denominator))
        .collect()
}
