use std::path::PathBuf;

use crate::{InferenceError, InferenceFamily};

const ONNX_FILE_PRIORITY: &[&str] = &[
    "model.onnx",
    "onnx/model.onnx",
    "encoder_decoder_model.onnx",
    "onnx/encoder_decoder_model.onnx",
    "decoder_model.onnx",
    "onnx/decoder_model.onnx",
    "model_quantized.onnx",
    "onnx/model_quantized.onnx",
    "decoder_model_quantized.onnx",
    "onnx/decoder_model_quantized.onnx",
];

/// Broad execution shape inferred from a model's Hugging Face `model_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    EncoderOnly,
    DecoderOnly,
    EncoderDecoder,
    Unknown,
}

impl ModelFamily {
    pub fn from_model_type(model_type: Option<&str>) -> Self {
        match model_type {
            Some("bert") | Some("roberta") | Some("distilbert") | Some("albert")
            | Some("deberta") | Some("deberta-v2") | Some("electra") | Some("xlm-roberta") => {
                Self::EncoderOnly
            }
            Some("gpt2") | Some("gpt_neo") | Some("gptj") | Some("gpt_neox") | Some("llama")
            | Some("mistral") | Some("phi") | Some("opt") | Some("qwen2") | Some("qwen2_moe")
            | Some("qwen3") => Self::DecoderOnly,
            Some("t5") | Some("mt5") | Some("bart") | Some("mbart") | Some("pegasus")
            | Some("marian") | Some("blenderbot") | Some("long-t5") => Self::EncoderDecoder,
            _ => Self::Unknown,
        }
    }

    pub fn load(self, artifact: ModelArtifact) -> Result<InferenceFamily, InferenceError> {
        match self {
            Self::EncoderOnly => InferenceFamily::load_encoder_only(artifact),
            Self::DecoderOnly => InferenceFamily::load_decoder_only(artifact),
            Self::EncoderDecoder => InferenceFamily::load_encoder_decoder(artifact),
            Self::Unknown => Err(InferenceError::UnknownModelFamily),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub artifact_path: PathBuf,
    onnx_file: Option<PathBuf>,
}

impl ModelArtifact {
    pub fn new(files: Vec<PathBuf>) -> Result<Self, std::io::Error> {
        let anchor = files
            .iter()
            .find(|path| {
                path.file_name()
                    .is_some_and(|name| name == "tokenizer.json")
            })
            .or_else(|| {
                files
                    .iter()
                    .find(|path| path.file_name().is_some_and(|name| name == "config.json"))
            })
            .or_else(|| {
                files
                    .iter()
                    .find(|path| path.extension().is_some_and(|ext| ext == "onnx"))
            });
        let artifact_path = anchor
            .and_then(|path| path.parent())
            .map(|parent| {
                if parent.file_name().is_some_and(|name| name == "onnx") {
                    parent.parent().unwrap_or(parent)
                } else {
                    parent
                }
            })
            .map(PathBuf::from)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "model download returned no files",
                )
            })?;
        let onnx_file = ONNX_FILE_PRIORITY
            .iter()
            .find_map(|name| files.iter().find(|path| path.ends_with(name)))
            .cloned()
            .or_else(|| {
                files
                    .iter()
                    .find(|path| path.extension().is_some_and(|ext| ext == "onnx"))
                    .cloned()
            });
        Ok(Self {
            artifact_path,
            onnx_file,
        })
    }

    pub fn onnx_path(&self) -> PathBuf {
        if let Some(path) = &self.onnx_file {
            return path.clone();
        }
        let candidates: Vec<_> = ONNX_FILE_PRIORITY
            .iter()
            .map(|name| self.artifact_path.join(name))
            .collect();
        candidates
            .iter()
            .find(|path| path.is_file() && has_external_data(path))
            .or_else(|| candidates.iter().find(|path| path.is_file()))
            .cloned()
            .or_else(|| find_file(&self.artifact_path, "onnx"))
            .unwrap_or_else(|| self.artifact_path.join("model.onnx"))
    }

    pub fn encoder_onnx_path(&self) -> Option<PathBuf> {
        ["encoder_model.onnx", "onnx/encoder_model.onnx"]
            .iter()
            .map(|name| self.artifact_path.join(name))
            .find(|path| path.is_file())
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        ["tokenizer.json", "onnx/tokenizer.json"]
            .iter()
            .map(|name| self.artifact_path.join(name))
            .find(|path| path.is_file())
            .or_else(|| find_file(&self.artifact_path, "tokenizer.json"))
            .unwrap_or_else(|| self.artifact_path.join("tokenizer.json"))
    }

    pub fn from_directory(path: impl Into<PathBuf>) -> Self {
        let artifact_path = path.into();
        Self {
            artifact_path,
            onnx_file: None,
        }
    }
}

fn has_external_data(model: &std::path::Path) -> bool {
    let Some(name) = model.file_name() else {
        return false;
    };
    model
        .with_file_name(format!("{}.data", name.to_string_lossy()))
        .is_file()
        || model
            .with_file_name(format!("{}_data", name.to_string_lossy()))
            .is_file()
}

fn find_file(directory: &std::path::Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(directory).ok()?.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|file| file == name) {
            return Some(path);
        }
        if path.is_dir()
            && let Some(found) = find_file(&path, name)
        {
            return Some(found);
        }
    }
    None
}
