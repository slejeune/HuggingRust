use std::path::PathBuf;

use inference::models::{ModelArtifact, ModelSpec, ModelTask};

fn main() {
    let spec = ModelSpec {
        repository: "BAAI/bge-small-en-v1.5".to_owned(),
        task: ModelTask::SentenceEmbedding,
    };

    let artifact = ModelArtifact {
        path: PathBuf::from("/onnx_models/bge-small-en-v1.5"),
    };
}
