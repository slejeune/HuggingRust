use crate::{onnx, tokenisers};

use crate::ModelArtifact;

pub struct Inference {
    model: ModelArtifact,
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl Inference {
    pub fn load(model: ModelArtifact) -> ort::Result<Self> {
        let session = onnx::load_model(&model)?;

        let tokenizer = tokenisers::load(model.tokenizer_path())?;

        Ok(Self {
            model,
            session,
            tokenizer,
        })
    }

    pub fn run(&mut self, text: &str) -> ort::Result<Vec<f32>> {
        let encoding = tokenisers::encode(&self.tokenizer, text).expect("tokenisation failed");

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

        onnx::run_sentence_embedding(
            &mut self.session,
            input_ids,
            attention_mask,
            token_type_ids,
            sequence_length,
        )
    }

    pub fn model(&self) -> &ModelArtifact {
        &self.model
    }
}
