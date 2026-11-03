use crate::{InferenceError, ModelArtifact, generation, generation_config, onnx, tokenisers};
use ort::session::Session;

pub struct EncoderDecoder {
    graph: EncoderDecoderGraph,
    tokenizer: tokenizers::Tokenizer,
    decoder_start_token_id: i64,
    eos_token_ids: Vec<u32>,
}

enum EncoderDecoderGraph {
    Single(Session),
    Split { encoder: Session, decoder: Session },
}

impl EncoderDecoder {
    pub(crate) fn load(model: ModelArtifact) -> Result<Self, InferenceError> {
        let onnx_path = model.onnx_path();
        let graph = if let Some(encoder_path) = model.encoder_onnx_path() {
            let encoder = onnx::load_model_path(&encoder_path).map_err(|source| {
                InferenceError::OnnxLoad {
                    path: encoder_path,
                    source,
                }
            })?;
            let decoder = onnx::load_model(&model).map_err(|source| InferenceError::OnnxLoad {
                path: onnx_path,
                source,
            })?;
            EncoderDecoderGraph::Split { encoder, decoder }
        } else {
            EncoderDecoderGraph::Single(onnx::load_model(&model).map_err(|source| {
                InferenceError::OnnxLoad {
                    path: onnx_path,
                    source,
                }
            })?)
        };
        let tokenizer_path = model.tokenizer_path();
        let tokenizer =
            tokenisers::load(&tokenizer_path).map_err(|source| InferenceError::TokenizerLoad {
                path: tokenizer_path,
                source,
            })?;
        let decoder_start_token_id =
            generation_config::read_token_id(&model.artifact_path, "decoder_start_token_id")
                .or_else(|| generation_config::read_token_id(&model.artifact_path, "bos_token_id"))
                .unwrap_or(0);
        let eos_token_ids = generation_config::read_eos_token_ids(&model.artifact_path, &tokenizer);
        Ok(Self {
            graph,
            tokenizer,
            decoder_start_token_id,
            eos_token_ids,
        })
    }

    pub(crate) fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let input_ids = self.encode(text)?;
        self.next_token_logits(&input_ids, &[self.decoder_start_token_id])
    }

    pub(crate) fn run_outputs(
        &mut self,
        text: &str,
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        let input_ids = self.encode(text)?;
        self.run_outputs_for(&input_ids)
    }

    pub(crate) fn run_pair_outputs(
        &mut self,
        first: &str,
        second: &str,
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        let encoding = tokenisers::encode_pair(&self.tokenizer, first, second)?;
        let input_ids = tokenisers::model_inputs(&encoding).input_ids;
        self.run_outputs_for(&input_ids)
    }

    fn run_outputs_for(
        &mut self,
        input_ids: &[i64],
    ) -> Result<Vec<onnx::TensorOutput>, InferenceError> {
        let decoder_ids = [self.decoder_start_token_id];
        match &mut self.graph {
            EncoderDecoderGraph::Single(session) => Ok(onnx::run_text_outputs(
                session,
                input_ids,
                None,
                None,
                Some(&decoder_ids),
            )?),
            EncoderDecoderGraph::Split { encoder, decoder } => {
                let values = onnx::run_split_encoder_decoder_logits(
                    encoder,
                    decoder,
                    input_ids,
                    &decoder_ids,
                )?;
                Ok(vec![onnx::TensorOutput {
                    name: "logits".into(),
                    shape: vec![1, 1, values.len() as i64],
                    values,
                }])
            }
        }
    }

    pub(crate) fn generate(
        &mut self,
        text: &str,
        max_new_tokens: usize,
    ) -> Result<String, InferenceError> {
        let input_ids = self.encode(text)?;
        let mut output_ids = vec![self.decoder_start_token_id];
        let eos_token_ids = self.eos_token_ids.clone();
        let generated = generation::greedy_generate(
            &mut output_ids,
            max_new_tokens,
            &eos_token_ids,
            |tokens| self.next_token_logits(&input_ids, tokens),
        )?;
        self.tokenizer
            .decode(&generated, false)
            .map_err(InferenceError::Tokenize)
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        Ok(tokenisers::model_inputs(&encoding).input_ids)
    }

    fn next_token_logits(
        &mut self,
        input_ids: &[i64],
        decoder_input_ids: &[i64],
    ) -> Result<Vec<f32>, InferenceError> {
        let logits = match &mut self.graph {
            EncoderDecoderGraph::Single(session) => {
                onnx::run_encoder_decoder_logits(session, input_ids, decoder_input_ids)?
            }
            EncoderDecoderGraph::Split { encoder, decoder } => {
                onnx::run_split_encoder_decoder_logits(
                    encoder,
                    decoder,
                    input_ids,
                    decoder_input_ids,
                )?
            }
        };
        Ok(logits)
    }
}
