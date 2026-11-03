use std::path::Path;

use tokenizers::Tokenizer;

pub(crate) fn read_token_id(directory: &Path, key: &str) -> Option<i64> {
    ["generation_config.json", "config.json"]
        .iter()
        .find_map(|filename| {
            let bytes = std::fs::read(directory.join(filename)).ok()?;
            let config: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
            config.get(key)?.as_i64()
        })
}

pub(crate) fn read_eos_token_ids(directory: &Path, tokenizer: &Tokenizer) -> Vec<u32> {
    configured_eos_token_ids(directory).unwrap_or_else(|| tokenizer_eos_token_ids(tokenizer))
}

fn configured_eos_token_ids(directory: &Path) -> Option<Vec<u32>> {
    ["generation_config.json", "config.json"]
        .iter()
        .find_map(|filename| {
            let bytes = std::fs::read(directory.join(filename)).ok()?;
            let config: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
            let eos = config.get("eos_token_id")?;
            let ids = match eos {
                serde_json::Value::Number(id) => id
                    .as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .into_iter()
                    .collect(),
                serde_json::Value::Array(ids) => ids
                    .iter()
                    .filter_map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                    .collect(),
                _ => Vec::new(),
            };
            (!ids.is_empty()).then_some(ids)
        })
}

fn tokenizer_eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    let vocab = tokenizer.get_vocab(true);
    let mut ids = ["<|endoftext|>", "</s>", "<|im_end|>", "<|end_of_text|>"]
        .iter()
        .filter_map(|token| vocab.get(*token).copied())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids
}
