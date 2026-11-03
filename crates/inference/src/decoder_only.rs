use crate::{InferenceError, ModelArtifact, generation, generation_config, onnx, tokenisers};
use ort::session::Session;

pub struct DecoderOnly {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    eos_token_ids: Vec<u32>,
}

impl DecoderOnly {
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

        let eos_token_ids = generation_config::read_eos_token_ids(&model.artifact_path, &tokenizer);
        Ok(Self {
            session,
            tokenizer,
            eos_token_ids,
        })
    }

    pub(crate) fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let inputs = tokenisers::model_inputs(&encoding);
        Ok(onnx::run_decoder_output(
            &mut self.session,
            &inputs.input_ids,
        )?)
    }

    pub(crate) fn returns_hidden_states(&self) -> bool {
        !onnx::has_output(&self.session, "logits")
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
        if let Some(embedding) = outputs
            .iter()
            .find(|output| output.name == "sentence_embedding")
        {
            return Ok(embedding.values.clone());
        }
        let hidden = outputs
            .iter()
            .find(|output| output.name == "last_hidden_state")
            .ok_or(onnx::OnnxRunError::MissingNamedOutput(
                "last_hidden_state".into(),
            ))?;
        Ok(onnx::last_token_embedding(
            hidden,
            encoding.get_attention_mask(),
        )?)
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

    pub(crate) fn generate(
        &mut self,
        text: &str,
        max_new_tokens: usize,
    ) -> Result<String, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let mut input_ids = tokenisers::model_inputs(&encoding).input_ids;

        if input_ids.is_empty() || max_new_tokens == 0 {
            return self
                .tokenizer
                .decode(
                    &input_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                    false,
                )
                .map_err(InferenceError::Tokenize);
        }

        let eos_token_ids = self.eos_token_ids.clone();
        let generated = generation::greedy_generate(
            &mut input_ids,
            max_new_tokens,
            &eos_token_ids,
            |tokens| self.next_token_logits(tokens),
        )?;
        let continuation = self
            .tokenizer
            .decode(&generated, false)
            .map_err(InferenceError::Tokenize)?;

        Ok(format!("{text}{continuation}"))
    }

    fn next_token_logits(&mut self, input_ids: &[i64]) -> Result<Vec<f32>, InferenceError> {
        Ok(onnx::run_decoder_logits(&mut self.session, input_ids)?)
    }
}
