mod decoder_only;
mod encoder_only;
pub mod models;
pub mod onnx;
pub mod tokenisers;

use decoder_only::DecoderOnly;
use encoder_only::EncoderOnly;
pub use models::ModelArtifact;
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
    #[error("Text generation is only supported for decoder-only models")]
    GenerationUnsupported,
    #[error("Unsupported ONNX model signature: {0}")]
    UnsupportedOnnxSignature(String),
}

pub enum InferenceFamily {
    EncoderOnly(EncoderOnly),
    DecoderOnly(DecoderOnly),
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

    pub fn run_typed(&mut self, text: &str) -> Result<InferenceOutput, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run(text).map(InferenceOutput::HiddenStates),
            Self::DecoderOnly(inference) => {
                inference.run(text).map(InferenceOutput::NextTokenLogits)
            }
        }
    }

    pub fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run(text),
            Self::DecoderOnly(inference) => inference.run(text),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String, InferenceError> {
        match self {
            Self::DecoderOnly(inference) => inference.generate(prompt, max_new_tokens),
            Self::EncoderOnly(_) => Err(InferenceError::GenerationUnsupported),
        }
    }
}
