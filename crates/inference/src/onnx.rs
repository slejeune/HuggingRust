mod embedding;
mod encoder_decoder;
mod text;
mod text_inputs;

use crate::ModelArtifact;
use ort::session::Session;
use thiserror::Error;

pub use embedding::last_token_embedding;
pub use encoder_decoder::{run_encoder_decoder_logits, run_split_encoder_decoder_logits};
pub use text::{
    TensorOutput, has_output, run_decoder_logits, run_decoder_output, run_encoder_hidden_states,
    run_encoder_output, run_text_output, run_text_outputs,
};

#[derive(Debug, Error)]
pub enum OnnxRunError {
    #[error("ONNX Runtime error: {0}")]
    Runtime(#[from] ort::Error),
    #[error("ONNX model did not return the required output tensor")]
    MissingOutput,
    #[error("ONNX model did not return the required '{0}' output")]
    MissingNamedOutput(String),
    #[error("Unsupported ONNX input signature: {0}")]
    UnsupportedSignature(String),
}

pub fn load_model(model: &ModelArtifact) -> ort::Result<Session> {
    load_model_path(model.onnx_path())
}

pub fn load_model_path(path: impl AsRef<std::path::Path>) -> ort::Result<Session> {
    Session::builder()?.commit_from_file(path)
}
