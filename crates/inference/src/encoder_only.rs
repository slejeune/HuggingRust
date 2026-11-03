use crate::{InferenceError, ModelArtifact, onnx, tokenisers};

pub struct EncoderOnly {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl EncoderOnly {
    pub(crate) fn load(model: ModelArtifact) -> Result<Self, InferenceError> {
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

        Ok(Self { session, tokenizer })
    }

    pub(crate) fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let inputs = tokenisers::model_inputs(&encoding);

        Ok(onnx::run_encoder_output(
            &mut self.session,
            inputs.input_ids,
            inputs.attention_mask,
            inputs.token_type_ids,
        )?)
    }

    pub(crate) fn returns_hidden_states(&self) -> bool {
        onnx::has_output(&self.session, "last_hidden_state")
            || onnx::has_output(&self.session, "sentence_embedding")
    }

    pub(crate) fn run_outputs(
        &mut self,
        text: &str,
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let inputs = tokenisers::model_inputs(&encoding);
        Ok(onnx::run_text_outputs(
            &mut self.session,
            &inputs.input_ids,
            Some(&inputs.attention_mask),
            Some(&inputs.token_type_ids),
            None,
        )?)
    }

    pub(crate) fn embed(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let inputs = tokenisers::model_inputs(&encoding);
        let outputs = onnx::run_text_outputs(
            &mut self.session,
            &inputs.input_ids,
            Some(&inputs.attention_mask),
            Some(&inputs.token_type_ids),
            None,
        )?;
        if let Some(output) = outputs
            .iter()
            .find(|output| output.name == "sentence_embedding" || output.name == "pooler_output")
        {
            if output.shape.len() == 2 && output.shape[0] == 1 {
                return Ok(output.values.clone());
            }
        }
        let hidden = outputs
            .iter()
            .find(|output| output.name == "last_hidden_state")
            .ok_or_else(|| onnx::OnnxRunError::MissingNamedOutput("last_hidden_state".into()))?;
        Ok(onnx::mean_embedding(hidden, encoding.get_attention_mask())?)
    }

    pub(crate) fn run_pair_outputs(
        &mut self,
        first: &str,
        second: &str,
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        let encoding = tokenisers::encode_pair(&self.tokenizer, first, second)?;
        let inputs = tokenisers::model_inputs(&encoding);
        Ok(onnx::run_text_outputs(
            &mut self.session,
            &inputs.input_ids,
            Some(&inputs.attention_mask),
            Some(&inputs.token_type_ids),
            None,
        )?)
    }
}
