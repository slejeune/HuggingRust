use crate::{InferenceError, InferenceFamily};
mod output;
use output::{
    argmax, best_span, sequence_scores, softmax_candidates, tensor_output, vector_output,
};

#[derive(Debug, Clone, PartialEq)]
pub enum TaskInput {
    Text(String),
    Pair {
        first: String,
        second: String,
    },
    CandidateLabels {
        text: String,
        labels: Vec<String>,
        hypothesis_template: String,
    },
    Choices {
        context: String,
        choices: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskOutput {
    GeneratedText(String),
    Embedding(Vec<f32>),
    ClassificationScores(Vec<f32>),
    TokenPredictions(Vec<TokenPrediction>),
    QuestionAnswering {
        start_token: usize,
        end_token: usize,
    },
    FillMaskPrediction {
        token: String,
        score: f32,
    },
    ZeroShotScores(Vec<(String, f32)>),
    Similarity(f32),
    MultipleChoiceScores(Vec<f32>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenPrediction {
    pub token_index: usize,
    pub class_index: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Task {
    FeatureExtraction,
    Generation,
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    FillMask,
    ZeroShotClassification,
    TextSimilarity,
    MultipleChoice,
}

impl Task {
    fn parse(name: &str) -> Option<Self> {
        Some(match canonical_task_name(name)? {
            "feature-extraction" => Self::FeatureExtraction,
            "text-generation" | "text2text-generation" => Self::Generation,
            "text-classification" => Self::TextClassification,
            "token-classification" => Self::TokenClassification,
            "question-answering" => Self::QuestionAnswering,
            "fill-mask" => Self::FillMask,
            "zero-shot-classification" => Self::ZeroShotClassification,
            "text-similarity" => Self::TextSimilarity,
            "multiple-choice" => Self::MultipleChoice,
            _ => return None,
        })
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::FeatureExtraction => "feature-extraction",
            Self::Generation => "text generation",
            Self::TextClassification => "text-classification",
            Self::TokenClassification => "token-classification",
            Self::QuestionAnswering => "question-answering",
            Self::FillMask => "fill-mask",
            Self::ZeroShotClassification => "zero-shot-classification",
            Self::TextSimilarity => "text-similarity",
            Self::MultipleChoice => "multiple-choice",
        }
    }
}

/// Maps supported Hugging Face task tags and common aliases to a canonical task.
pub fn canonical_task_name(name: &str) -> Option<&'static str> {
    if name.starts_with("translation_") {
        return Some("text2text-generation");
    }
    Some(match name {
        "feature-extraction" | "sentence-embeddings" => "feature-extraction",
        "text-generation" | "conversational" => "text-generation",
        "text2text-generation" | "text-to-text-generation" | "translation" | "summarization" => {
            "text2text-generation"
        }
        "text-classification" | "sentiment-analysis" => "text-classification",
        "token-classification" | "ner" => "token-classification",
        "question-answering" => "question-answering",
        "fill-mask" => "fill-mask",
        "zero-shot-classification" => "zero-shot-classification",
        "sentence-similarity" | "text-similarity" | "text-ranking" => "text-similarity",
        "multiple-choice" => "multiple-choice",
        _ => return None,
    })
}

pub(crate) fn run(
    inference: &mut InferenceFamily,
    task_name: &str,
    input: TaskInput,
    max_new_tokens: usize,
) -> Result<TaskOutput, InferenceError> {
    let task = parse_task(task_name)?;
    if !supports(task, inference) {
        return Err(InferenceError::TaskUnsupportedForFamily {
            task: task.name().to_owned(),
            family: family_name(inference).to_owned(),
        });
    }
    match (task, input) {
        (Task::Generation, TaskInput::Text(text)) => inference
            .generate(&text, max_new_tokens)
            .map(TaskOutput::GeneratedText),
        (Task::FeatureExtraction, TaskInput::Text(text)) => {
            inference.embed(&text).map(TaskOutput::Embedding)
        }
        (Task::TextClassification, TaskInput::Text(text)) => {
            run_classification(inference, task, &text)
        }
        (Task::TextClassification, TaskInput::Pair { first, second }) => {
            run_pair_classification(inference, task, &first, &second)
        }
        (Task::TokenClassification, TaskInput::Text(text)) => {
            run_token_classification(inference, task, &text)
        }
        (Task::FillMask, TaskInput::Text(text)) => run_fill_mask(inference, task, &text),
        (Task::QuestionAnswering, TaskInput::Pair { first, second }) => {
            run_question_answering(inference, task, &first, &second, max_new_tokens)
        }
        (
            Task::ZeroShotClassification,
            TaskInput::CandidateLabels {
                text,
                labels,
                hypothesis_template,
            },
        ) => run_zero_shot(inference, task, &text, labels, &hypothesis_template),
        (Task::TextSimilarity, TaskInput::Pair { first, second }) => {
            run_similarity(inference, task, &first, &second)
        }
        (Task::MultipleChoice, TaskInput::Choices { context, choices }) => {
            run_multiple_choice(inference, task, &context, &choices)
        }
        (expected, input) => Err(input_error(expected, input)),
    }
}

fn run_classification(
    inference: &mut InferenceFamily,
    task: Task,
    text: &str,
) -> Result<TaskOutput, InferenceError> {
    let outputs = inference.run_outputs(text)?;
    Ok(TaskOutput::ClassificationScores(vector_output(
        &outputs,
        task,
        &["logits"],
    )?))
}

fn run_pair_classification(
    inference: &mut InferenceFamily,
    task: Task,
    first: &str,
    second: &str,
) -> Result<TaskOutput, InferenceError> {
    let outputs = inference.run_pair_outputs(first, second)?;
    Ok(TaskOutput::ClassificationScores(vector_output(
        &outputs,
        task,
        &["logits"],
    )?))
}

fn run_token_classification(
    inference: &mut InferenceFamily,
    task: Task,
    text: &str,
) -> Result<TaskOutput, InferenceError> {
    let outputs = inference.run_outputs(text)?;
    let output = tensor_output(&outputs, task, &["logits"])?;
    let rows = sequence_scores(output, task)?;
    let predictions = rows
        .iter()
        .enumerate()
        .map(|(token_index, scores)| {
            let class_index = argmax(scores);
            TokenPrediction {
                token_index,
                class_index,
                score: scores[class_index],
            }
        })
        .collect();
    Ok(TaskOutput::TokenPredictions(predictions))
}

fn run_fill_mask(
    inference: &mut InferenceFamily,
    task: Task,
    text: &str,
) -> Result<TaskOutput, InferenceError> {
    let mask_position = inference.mask_position(text)?;
    let outputs = inference.run_outputs(text)?;
    let output = tensor_output(&outputs, task, &["logits"])?;
    let rows = sequence_scores(output, task)?;
    let scores = rows
        .get(mask_position)
        .ok_or_else(|| InferenceError::TaskOutputInvalid {
            task: task.name().to_owned(),
            detail: format!("mask token position {mask_position} is outside the logits sequence"),
        })?;
    let token_id = argmax(scores);
    Ok(TaskOutput::FillMaskPrediction {
        token: inference.decode_token(token_id)?,
        score: scores[token_id],
    })
}

fn run_question_answering(
    inference: &mut InferenceFamily,
    task: Task,
    question: &str,
    context: &str,
    max_new_tokens: usize,
) -> Result<TaskOutput, InferenceError> {
    if matches!(
        inference,
        InferenceFamily::DecoderOnly(_) | InferenceFamily::EncoderDecoder(_)
    ) {
        let prompt = format!("Context: {context}\nQuestion: {question}\nAnswer:");
        return inference
            .generate(&prompt, max_new_tokens)
            .map(TaskOutput::GeneratedText);
    }
    let outputs = inference.run_pair_outputs(question, context)?;
    let start = vector_output(&outputs, task, &["start_logits"])?;
    let end = vector_output(&outputs, task, &["end_logits"])?;
    let (start_token, end_token) =
        best_span(&start, &end).ok_or_else(|| InferenceError::TaskOutputInvalid {
            task: task.name().to_owned(),
            detail: "start and end logits do not share a valid token span".into(),
        })?;
    Ok(TaskOutput::QuestionAnswering {
        start_token,
        end_token,
    })
}

fn run_zero_shot(
    inference: &mut InferenceFamily,
    task: Task,
    text: &str,
    labels: Vec<String>,
    hypothesis_template: &str,
) -> Result<TaskOutput, InferenceError> {
    if labels.is_empty() {
        return Err(InferenceError::TaskInputInvalid {
            task: task.name().into(),
            detail: "at least one candidate label is required".into(),
        });
    }
    if !hypothesis_template.contains("{}") {
        return Err(InferenceError::TaskInputInvalid {
            task: task.name().into(),
            detail: "hypothesis template must contain '{}' for the candidate label".into(),
        });
    }
    let entailment_index =
        inference
            .entailment_class_index()
            .ok_or_else(|| InferenceError::TaskLabelMissing {
                task: task.name().into(),
                label: "entailment".into(),
            })?;
    let mut logits = Vec::with_capacity(labels.len());
    for label in labels {
        let hypothesis = hypothesis_template.replace("{}", &label);
        let outputs = inference.run_pair_outputs(text, &hypothesis)?;
        let label_scores = vector_output(&outputs, task, &["logits"])?;
        let score = *label_scores.get(entailment_index).ok_or_else(|| {
            InferenceError::TaskOutputInvalid {
                task: task.name().to_owned(),
                detail: format!(
                    "entailment class index {entailment_index} exceeds {} output classes",
                    label_scores.len()
                ),
            }
        })?;
        logits.push((label, score));
    }
    Ok(TaskOutput::ZeroShotScores(softmax_candidates(logits)))
}

fn run_similarity(
    inference: &mut InferenceFamily,
    task: Task,
    first_text: &str,
    second_text: &str,
) -> Result<TaskOutput, InferenceError> {
    let first = inference.embed(first_text)?;
    let second = inference.embed(second_text)?;
    if first.len() != second.len() {
        return Err(InferenceError::TaskOutputInvalid {
            task: task.name().into(),
            detail: "the two embeddings have different sizes".into(),
        });
    }
    let dot = first.iter().zip(&second).map(|(a, b)| a * b).sum::<f32>();
    let first_norm = first.iter().map(|value| value * value).sum::<f32>().sqrt();
    let second_norm = second.iter().map(|value| value * value).sum::<f32>().sqrt();
    if first_norm == 0.0 || second_norm == 0.0 {
        return Err(InferenceError::TaskOutputInvalid {
            task: task.name().into(),
            detail: "cannot compare a zero-length embedding".into(),
        });
    }
    Ok(TaskOutput::Similarity(dot / (first_norm * second_norm)))
}

fn run_multiple_choice(
    inference: &mut InferenceFamily,
    task: Task,
    context: &str,
    choices: &[String],
) -> Result<TaskOutput, InferenceError> {
    if choices.is_empty() {
        return Err(InferenceError::TaskInputInvalid {
            task: task.name().into(),
            detail: "at least one choice is required".into(),
        });
    }
    let outputs = inference.run_multiple_choice_outputs(context, choices)?;
    let scores = vector_output(&outputs, task, &["logits"])?;
    if scores.len() != choices.len() {
        return Err(InferenceError::TaskOutputInvalid {
            task: task.name().into(),
            detail: format!(
                "received {} scores for {} choices",
                scores.len(),
                choices.len()
            ),
        });
    }
    Ok(TaskOutput::MultipleChoiceScores(scores))
}

fn parse_task(name: &str) -> Result<Task, InferenceError> {
    Task::parse(name).ok_or_else(|| InferenceError::UnsupportedTask {
        task: name.to_owned(),
    })
}

fn input_error(task: Task, input: TaskInput) -> InferenceError {
    let expected = match task {
        Task::Generation | Task::FeatureExtraction | Task::TokenClassification | Task::FillMask => {
            "text"
        }
        Task::TextClassification => "text or a text pair",
        Task::QuestionAnswering | Task::TextSimilarity => "a text pair",
        Task::ZeroShotClassification => "text and candidate labels",
        Task::MultipleChoice => "a context and choices",
    };
    InferenceError::TaskInputMismatch {
        task: task.name().to_owned(),
        expected: expected.to_owned(),
        received: match input {
            TaskInput::Text(_) => "text",
            TaskInput::Pair { .. } => "a text pair",
            TaskInput::CandidateLabels { .. } => "text and candidate labels",
            TaskInput::Choices { .. } => "a context and choices",
        }
        .to_owned(),
    }
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
        Task::MultipleChoice => matches!(inference, InferenceFamily::EncoderOnly(_)),
        Task::FillMask | Task::ZeroShotClassification => {
            matches!(inference, InferenceFamily::EncoderOnly(_))
        }
    }
}

fn family_name(inference: &InferenceFamily) -> &'static str {
    match inference {
        InferenceFamily::EncoderOnly(_) => "encoder-only",
        InferenceFamily::DecoderOnly(_) => "decoder-only",
        InferenceFamily::EncoderDecoder(_) => "encoder-decoder",
    }
}
