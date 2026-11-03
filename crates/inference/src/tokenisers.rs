use std::path::Path;

use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ModelInputs {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
}

pub fn load(path: impl AsRef<Path>) -> tokenizers::Result<Tokenizer> {
    Tokenizer::from_file(path)
}

pub fn encode(tokenizer: &Tokenizer, text: &str) -> tokenizers::Result<tokenizers::Encoding> {
    tokenizer.encode(text, true)
}

pub fn encode_pair(
    tokenizer: &Tokenizer,
    first: &str,
    second: &str,
) -> tokenizers::Result<tokenizers::Encoding> {
    tokenizer.encode((first, second), true)
}

pub fn model_inputs(encoding: &tokenizers::Encoding) -> ModelInputs {
    ModelInputs {
        input_ids: encoding.get_ids().iter().map(|&id| i64::from(id)).collect(),
        attention_mask: encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| i64::from(mask))
            .collect(),
        token_type_ids: encoding
            .get_type_ids()
            .iter()
            .map(|&token_type| i64::from(token_type))
            .collect(),
    }
}
