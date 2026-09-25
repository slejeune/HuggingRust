use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    /// Directory containing every artifact belonging to this model: the ONNX
    /// graph, tokenizer files, and any model configuration files.
    pub artifact_path: PathBuf,
    pub model_type: ModelType,
}

impl Model {
    pub fn onnx_path(&self) -> PathBuf {
        self.artifact_path.join("model.onnx")
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.artifact_path.join("tokenizer.json")
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    SentenceEmbedding,
}
