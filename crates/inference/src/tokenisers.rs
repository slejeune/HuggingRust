use std::path::Path;

use tokenizers::Tokenizer;

pub fn load(path: impl AsRef<Path>) -> tokenizers::Result<Tokenizer> {
    Tokenizer::from_file(path)
}

pub fn encode(tokenizer: &Tokenizer, text: &str) -> tokenizers::Result<tokenizers::Encoding> {
    tokenizer.encode(text, true)
}
