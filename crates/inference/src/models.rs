use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub artifact_path: PathBuf,
}

impl ModelArtifact {
    pub fn new(files: Vec<PathBuf>) -> Result<Self, std::io::Error> {
        let artifact_path = files
            .first()
            .and_then(|path| path.parent())
            .map(PathBuf::from)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "model download returned no files",
                )
            })?;
        Ok(Self { artifact_path })
    }

    pub fn onnx_path(&self) -> PathBuf {
        self.artifact_path.join("model.onnx")
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        self.artifact_path.join("tokenizer.json")
    }

    pub fn from_directory(path: impl Into<PathBuf>) -> Self {
        let artifact_path = path.into();
        Self { artifact_path }
    }
}
