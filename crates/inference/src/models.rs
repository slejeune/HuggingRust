use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelTask {
    SentenceEmbedding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub repository: String,
    pub task: ModelTask,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    pub path: PathBuf,
}
