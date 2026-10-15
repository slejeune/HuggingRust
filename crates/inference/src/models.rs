use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub artifact_path: PathBuf,
}

impl ModelArtifact {
    pub fn new(files: Vec<PathBuf>) -> Self {
        let artifact_path = files
            .first()
            .and_then(|path| path.parent())
            .map(PathBuf::from)
            .unwrap_or_default();
        Self { artifact_path }
    }

    pub fn onnx_path(&self) -> PathBuf {
        self.artifact_path.join("model.onnx")
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.artifact_path.join("tokenizer.json")
    }
}
