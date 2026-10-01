use std::path::PathBuf;

use importer::ModelMetadata;

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub metadata: ModelMetadata,
    pub artifact_path: PathBuf,
}

impl ModelArtifact {
    pub fn onnx_path(&self) -> PathBuf {
        self.artifact_path.join("model.onnx")
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.artifact_path.join("tokenizer.json")
    }
}
