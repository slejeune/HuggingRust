mod encoder_only;
pub mod models;
pub mod onnx;
pub mod tokenisers;

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
}

pub enum InferenceFamily {
    EncoderOnly(EncoderOnly),
}

impl InferenceFamily {
    pub fn load_encoder_only(model: ModelArtifact) -> Result<Self, InferenceError> {
        Ok(Self::EncoderOnly(EncoderOnly::load(model)?))
    }

    pub fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        match self {
            Self::EncoderOnly(inference) => inference.run(text),
        }
    }
}
