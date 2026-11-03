use crate::{InferenceError, InferenceFamily, onnx};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Task {
    FeatureExtraction,
    Generation,
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    FillMask,
    ZeroShotClassification,
    TextSimilarity,
}

impl Task {
    fn parse(name: &str) -> Option<Self> {
        if name.starts_with("translation_") {
            return Some(Self::Generation);
        }
        Some(match name {
            "feature-extraction" | "sentence-embeddings" => Self::FeatureExtraction,
            "text-generation"
            | "text2text-generation"
            | "text-to-text-generation"
            | "translation"
            | "summarization"
            | "conversational" => Self::Generation,
            "text-classification" | "sentiment-analysis" => Self::TextClassification,
            "token-classification" | "ner" => Self::TokenClassification,
            "question-answering" => Self::QuestionAnswering,
            "fill-mask" => Self::FillMask,
            "zero-shot-classification" => Self::ZeroShotClassification,
            "text-similarity" | "sentence-similarity" | "text-ranking" => Self::TextSimilarity,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::FeatureExtraction => "feature-extraction",
            Self::Generation => "text generation",
            Self::TextClassification => "text-classification",
            Self::TokenClassification => "token-classification",
            Self::QuestionAnswering => "question-answering",
            Self::FillMask => "fill-mask",
            Self::ZeroShotClassification => "zero-shot-classification",
            Self::TextSimilarity => "text-similarity",
        }
    }
}

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
    let Some(task_name) = task else {
        return match inference {
            InferenceFamily::DecoderOnly(_) | InferenceFamily::EncoderDecoder(_) => inference
                .generate(text, max_new_tokens)
                .map(TaskOutput::GeneratedText),
            InferenceFamily::EncoderOnly(_) => inference.run_outputs(text).map(TaskOutput::Tensors),
        };
    };
    let task = Task::parse(task_name).ok_or_else(|| InferenceError::UnsupportedTask {
        task: task_name.to_owned(),
    })?;

    if task == Task::Generation {
        return inference
            .generate(text, max_new_tokens)
            .map(TaskOutput::GeneratedText);
    }
    if !supports(task, inference) {
        return Err(InferenceError::TaskUnsupportedForFamily {
            task: task.name().to_owned(),
            family: family_name(inference).to_owned(),
        });
    }
    if task == Task::FeatureExtraction {
        return inference.embed(text).map(TaskOutput::Embedding);
    }
    if matches!(task, Task::QuestionAnswering | Task::ZeroShotClassification) {
        return Err(InferenceError::TaskRequiresPair {
            task: task.name().to_owned(),
        });
    }
    let outputs = inference.run_outputs(text)?;
    let required: &[&str] = match task {
        Task::TextClassification | Task::FillMask | Task::TokenClassification => &["logits"],
        Task::TextSimilarity => &["sentence_embedding", "last_hidden_state", "pooler_output"],
        _ => &[],
    };
    if !required.is_empty()
        && !outputs
            .iter()
            .any(|output| required.contains(&output.name.as_str()))
    {
        return Err(InferenceError::TaskOutputMissing {
            task: task.name().to_owned(),
            expected: required.join(" or "),
        });
    }
    Ok(TaskOutput::Tensors(outputs))
}

pub(crate) fn run_pair(
    inference: &mut InferenceFamily,
    task_name: &str,
    first: &str,
    second: &str,
) -> Result<TaskOutput, InferenceError> {
    let task = Task::parse(task_name).ok_or_else(|| InferenceError::UnsupportedTask {
        task: task_name.to_owned(),
    })?;
    if !matches!(
        task,
        Task::QuestionAnswering
            | Task::ZeroShotClassification
            | Task::TextClassification
            | Task::TextSimilarity
    ) {
        return Err(InferenceError::TaskDoesNotAcceptPair {
            task: task.name().to_owned(),
        });
    }
    if !supports(task, inference) {
        return Err(InferenceError::TaskUnsupportedForFamily {
            task: task.name().to_owned(),
            family: family_name(inference).to_owned(),
        });
    }
    if task == Task::QuestionAnswering
        && matches!(
            inference,
            InferenceFamily::DecoderOnly(_) | InferenceFamily::EncoderDecoder(_)
        )
    {
        let prompt = format!("Context: {second}\nQuestion: {first}\nAnswer:");
        return inference
            .generate(&prompt, 64)
            .map(TaskOutput::GeneratedText);
    }
    let outputs = inference.run_pair_outputs(first, second)?;
    let required: &[&str] = if task == Task::QuestionAnswering {
        &["start_logits", "end_logits"]
    } else {
        &[
            "logits",
            "sentence_embedding",
            "last_hidden_state",
            "pooler_output",
        ]
    };
    if !outputs
        .iter()
        .any(|output| required.contains(&output.name.as_str()))
    {
        return Err(InferenceError::TaskOutputMissing {
            task: task.name().to_owned(),
            expected: required.join(" or "),
        });
    }
    Ok(TaskOutput::Tensors(outputs))
}

fn supports(task: Task, inference: &InferenceFamily) -> bool {
    match task {
        Task::Generation => matches!(
            inference,
            InferenceFamily::DecoderOnly(_) | InferenceFamily::EncoderDecoder(_)
        ),
        Task::FeatureExtraction
        | Task::TextClassification
        | Task::TokenClassification
        | Task::QuestionAnswering
        | Task::TextSimilarity => true,
        Task::FillMask => matches!(inference, InferenceFamily::EncoderOnly(_)),
        Task::ZeroShotClassification => matches!(inference, InferenceFamily::EncoderOnly(_)),
    }
}

fn family_name(inference: &InferenceFamily) -> &'static str {
    match inference {
        InferenceFamily::EncoderOnly(_) => "encoder-only",
        InferenceFamily::DecoderOnly(_) => "decoder-only",
        InferenceFamily::EncoderDecoder(_) => "encoder-decoder",
    }
}
