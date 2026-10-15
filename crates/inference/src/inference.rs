use crate::{onnx, tokenisers};

use crate::ModelArtifact;
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
    Onnx(#[from] crate::onnx::OnnxRunError),
}

pub struct Inference {
    model: ModelArtifact,
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl Inference {
    pub fn load(model: ModelArtifact) -> Result<Self, InferenceError> {
        let onnx_path = model.onnx_path();
        let session = onnx::load_model(&model).map_err(|source| InferenceError::OnnxLoad {
            path: onnx_path,
            source,
        })?;

        let tokenizer_path = model.tokenizer_path();
        let tokenizer =
            tokenisers::load(&tokenizer_path).map_err(|source| InferenceError::TokenizerLoad {
                path: tokenizer_path,
                source,
            })?;

        Ok(Self {
            model,
            session,
            tokenizer,
        })
    }

    pub fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();

        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| i64::from(mask))
            .collect();

        let token_type_ids: Vec<i64> = encoding
            .get_type_ids()
            .iter()
            .map(|&type_id| i64::from(type_id))
            .collect();

        let sequence_length = input_ids.len();

        Ok(onnx::run_sentence_embedding(
            &mut self.session,
            input_ids,
            attention_mask,
            token_type_ids,
            sequence_length,
        )?)
    }

    pub fn model(&self) -> &ModelArtifact {
        &self.model
    }
}
