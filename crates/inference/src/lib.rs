mod decoder_only;
mod encoder_decoder;
mod encoder_only;
mod generation;
mod generation_config;
pub mod models;
pub mod onnx;
mod tasks;
pub mod tokenisers;

use decoder_only::DecoderOnly;
use encoder_decoder::EncoderDecoder;
use encoder_only::EncoderOnly;
pub use models::{ModelArtifact, ModelFamily};
pub use tasks::{TaskInput, TaskOutput, TokenPrediction, canonical_task_name};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("Failed to load ONNX model {path}: {source}")]
    OnnxLoad {
        path: std::path::PathBuf,
        #[source]
        source: ort::Error,
    },
    #[error("Failed to load tokenizer {path}: {source}")]
    TokenizerLoad {
        path: std::path::PathBuf,
        #[source]
        source: tokenizers::Error,
    },
    #[error("Failed to tokenize input text: {0}")]
    Tokenize(#[from] tokenizers::Error),
    #[error("ONNX inference failed: {0}")]
    Onnx(#[from] onnx::OnnxRunError),
    #[error("Text generation is not supported for encoder-only models")]
    GenerationUnsupported,
    #[error("Could not determine a supported model family from the model config")]
    UnknownModelFamily,
    #[error("Unsupported pipeline task '{task}'")]
    UnsupportedTask { task: String },
    #[error("Task '{task}' is not supported by the {family} model family")]
    TaskUnsupportedForFamily { task: String, family: String },
    #[error("Task '{task}' requires an ONNX output named {expected}")]
    TaskOutputMissing { task: String, expected: String },
    #[error("Task '{task}' expects {expected}, but received {received}")]
    TaskInputMismatch {
        task: String,
        expected: String,
        received: String,
    },
    #[error("Task '{task}' received invalid input: {detail}")]
    TaskInputInvalid { task: String, detail: String },
    #[error("Task '{task}' returned invalid output: {detail}")]
    TaskOutputInvalid { task: String, detail: String },
    #[error("The tokenizer does not define a supported mask token")]
    MaskTokenUnavailable,
    #[error("Fill-mask input must contain the tokenizer's mask token")]
    MaskTokenMissing,
    #[error("Task '{task}' requires the model config to identify the '{label}' class in id2label")]
    TaskLabelMissing { task: String, label: String },
}

pub enum InferenceFamily {
    EncoderOnly(EncoderOnly),
    DecoderOnly(DecoderOnly),
    EncoderDecoder(EncoderDecoder),
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceOutput {
    HiddenStates(Vec<f32>),
    NextTokenLogits(Vec<f32>),
}

impl InferenceFamily {
    pub fn load_encoder_only(model: ModelArtifact) -> Result<Self, InferenceError> {
        Ok(Self::EncoderOnly(EncoderOnly::load(model)?))
    }

    pub fn load_decoder_only(model: ModelArtifact) -> Result<Self, InferenceError> {
        Ok(Self::DecoderOnly(DecoderOnly::load(model)?))
    }

    pub fn load_encoder_decoder(model: ModelArtifact) -> Result<Self, InferenceError> {
        Ok(Self::EncoderDecoder(EncoderDecoder::load(model)?))
    }

    pub fn run_typed(&mut self, text: &str) -> Result<InferenceOutput, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run(text).map(|values| {
                if inference.returns_hidden_states() {
                    InferenceOutput::HiddenStates(values)
                } else {
                    InferenceOutput::NextTokenLogits(values)
                }
            }),
            Self::DecoderOnly(inference) => inference.run(text).map(|values| {
                if inference.returns_hidden_states() {
                    InferenceOutput::HiddenStates(values)
                } else {
                    InferenceOutput::NextTokenLogits(values)
                }
            }),
            Self::EncoderDecoder(inference) => {
                inference.run(text).map(InferenceOutput::NextTokenLogits)
            }
        }
    }

    pub fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run(text),
            Self::DecoderOnly(inference) => inference.run(text),
            Self::EncoderDecoder(inference) => inference.run(text),
        }
    }

    /// Runs the model and returns every named tensor produced by its ONNX graph.
    /// This preserves task-specific outputs such as start/end logits for QA and
    /// per-token logits for token classification.
    pub fn run_outputs(&mut self, text: &str) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run_outputs(text),
            Self::DecoderOnly(inference) => inference.run_outputs(text),
            Self::EncoderDecoder(inference) => inference.run_outputs(text),
        }
    }

    /// Runs pair-input text tasks such as question answering or text-pair
    /// classification and preserves every named graph output.
    pub fn run_pair_outputs(
        &mut self,
        first: &str,
        second: &str,
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run_pair_outputs(first, second),
            Self::DecoderOnly(inference) => inference.run_pair_outputs(first, second),
            Self::EncoderDecoder(inference) => inference.run_pair_outputs(first, second),
        }
    }

    pub(crate) fn run_multiple_choice_outputs(
        &mut self,
        context: &str,
        choices: &[String],
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        match self {
            Self::EncoderOnly(model) => model.run_multiple_choice(context, choices),
            Self::DecoderOnly(_) => Err(InferenceError::TaskUnsupportedForFamily {
                task: "multiple-choice".into(),
                family: "decoder-only".into(),
            }),
            Self::EncoderDecoder(_) => Err(InferenceError::TaskUnsupportedForFamily {
                task: "multiple-choice".into(),
                family: "encoder-decoder".into(),
            }),
        }
    }

    /// Runs the ONNX task named by the model metadata. Generation tasks return
    /// decoded text; other text tasks return all named graph outputs.
    pub fn run_task(
        &mut self,
        task: &str,
        input: TaskInput,
        max_new_tokens: usize,
    ) -> Result<TaskOutput, InferenceError> {
        tasks::run(self, task, input, max_new_tokens)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String, InferenceError> {
        match self {
            Self::DecoderOnly(inference) => inference.generate(prompt, max_new_tokens),
            Self::EncoderDecoder(inference) => inference.generate(prompt, max_new_tokens),
            Self::EncoderOnly(_) => Err(InferenceError::GenerationUnsupported),
        }
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.embed(text),
            Self::DecoderOnly(inference) => inference.embed(text),
            Self::EncoderDecoder(inference) => inference.embed(text),
        }
    }

    pub(crate) fn mask_position(&self, text: &str) -> Result<usize, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.mask_position(text),
            Self::DecoderOnly(_) => Err(InferenceError::TaskUnsupportedForFamily {
                task: "fill-mask".into(),
                family: "decoder-only".into(),
            }),
            Self::EncoderDecoder(_) => Err(InferenceError::TaskUnsupportedForFamily {
                task: "fill-mask".into(),
                family: "encoder-decoder".into(),
            }),
        }
    }

    pub(crate) fn entailment_class_index(&self) -> Option<usize> {
        match self {
            Self::EncoderOnly(inference) => inference.entailment_class_index(),
            _ => None,
        }
    }

    pub(crate) fn decode_token(&self, token_id: usize) -> Result<String, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.decode_token(token_id),
            Self::DecoderOnly(_) => Err(InferenceError::TaskUnsupportedForFamily {
                task: "fill-mask".into(),
                family: "decoder-only".into(),
            }),
            Self::EncoderDecoder(_) => Err(InferenceError::TaskUnsupportedForFamily {
                task: "fill-mask".into(),
                family: "encoder-decoder".into(),
            }),
        }
    }
}
