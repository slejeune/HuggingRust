use crate::{InferenceError, InferenceFamily, onnx};

#[derive(Debug, Clone, PartialEq)]
pub enum TaskOutput {
    GeneratedText(String),
    Embedding(Vec<f32>),
    Tensors(Vec<onnx::TensorOutput>),
}

pub(crate) fn run(
    inference: &mut InferenceFamily,
    task: Option<&str>,
    text: &str,
    max_new_tokens: usize,
) -> Result<TaskOutput, InferenceError> {
    if task == Some("feature-extraction")
        && let InferenceFamily::DecoderOnly(model) = inference
    {
        return model.embed(text).map(TaskOutput::Embedding);
    }

    if uses_generation(task, inference) {
        return inference
            .generate(text, max_new_tokens)
            .map(TaskOutput::GeneratedText);
    }
    inference.run_outputs(text).map(TaskOutput::Tensors)
}

fn uses_generation(task: Option<&str>, inference: &InferenceFamily) -> bool {
    task.is_some_and(is_generation_task)
        || task.is_none()
            && matches!(
                inference,
                InferenceFamily::DecoderOnly(_) | InferenceFamily::EncoderDecoder(_)
            )
}

fn is_generation_task(task: &str) -> bool {
    matches!(
        task,
        "text-generation"
            | "text2text-generation"
            | "translation"
            | "summarization"
            | "conversational"
    )
}
