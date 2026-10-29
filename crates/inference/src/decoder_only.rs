use crate::{InferenceError, ModelArtifact, onnx, tokenisers};
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

        let eos_token_ids = load_eos_token_ids(&model.artifact_path, &tokenizer);
        Ok(Self {
            session,
            tokenizer,
            eos_token_ids,
        })
    }

    pub(crate) fn run(&mut self, text: &str) -> Result<Vec<f32>, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        self.next_token_logits(&input_ids)
    }

    pub(crate) fn generate(
        &mut self,
        text: &str,
        max_new_tokens: usize,
    ) -> Result<String, InferenceError> {
        let encoding = tokenisers::encode(&self.tokenizer, text)?;
        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
        let prompt_length = input_ids.len();

        if input_ids.is_empty() || max_new_tokens == 0 {
            return self
                .tokenizer
                .decode(
                    &input_ids.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                    false,
                )
                .map_err(InferenceError::Tokenize);
        }

        for _ in 0..max_new_tokens {
            let logits = self.next_token_logits(&input_ids)?;
            let vocab_size = logits.len();
            if vocab_size == 0 {
                return Err(InferenceError::Onnx(onnx::OnnxRunError::MissingOutput));
            }

            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .expect("non-empty logits") as i64;

            if self.eos_token_ids.contains(&(next_token as u32)) {
                break;
            }
            input_ids.push(next_token);
        }

        let generated: Vec<u32> = input_ids[prompt_length..]
            .iter()
            .map(|&id| id as u32)
            .collect();
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

fn load_eos_token_ids(model_dir: &std::path::Path, tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    for filename in ["generation_config.json", "config.json"] {
        let Ok(bytes) = std::fs::read(model_dir.join(filename)) else {
            continue;
        };
        let Ok(config) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
            continue;
        };
        let Some(eos) = config.get("eos_token_id") else {
            continue;
        };
        match eos {
            serde_json::Value::Number(value) => {
                if let Some(id) = value.as_u64().and_then(|id| u32::try_from(id).ok()) {
                    ids.push(id);
                }
            }
            serde_json::Value::Array(values) => {
                ids.extend(
                    values
                        .iter()
                        .filter_map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok())),
                );
            }
            _ => {}
        }
        if !ids.is_empty() {
            break;
        }
    }

    if ids.is_empty() {
        let vocab = tokenizer.get_vocab(true);
        ids.extend(
            ["<|endoftext|>", "</s>", "<|im_end|>", "<|end_of_text|>"]
                .iter()
                .filter_map(|token| vocab.get(*token).copied()),
        );
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}
