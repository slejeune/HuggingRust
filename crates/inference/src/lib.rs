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
pub use models::ModelArtifact;
pub use tasks::TaskOutput;
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
    #[error("Unsupported ONNX model signature: {0}")]
    UnsupportedOnnxSignature(String),
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

    /// Runs the ONNX task named by the model metadata. Generation tasks return
    /// decoded text; other text tasks return all named graph outputs.
    pub fn run_task(
        &mut self,
        task: Option<&str>,
        text: &str,
        max_new_tokens: usize,
    ) -> Result<TaskOutput, InferenceError> {
        tasks::run(self, task, text, max_new_tokens)
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
}
