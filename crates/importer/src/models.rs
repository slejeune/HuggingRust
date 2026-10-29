use hf_hub::repository::ModelInfo;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("Hub returned invalid model ID {id:?}; expected author/model")]
    InvalidId { id: String },
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub author: String,
    pub model: String,
    pub revision: Option<String>,
    pub library: Option<String>,
    pub pipeline: Option<String>,
    pub files: Vec<ModelFile>,
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
}

impl ModelMetadata {
    pub fn from_huggingface(info: &ModelInfo) -> Result<Self, MetadataError> {
        let (author, model) = info
            .id
            .split_once('/')
            .filter(|(author, model)| !author.is_empty() && !model.is_empty())
            .ok_or_else(|| MetadataError::InvalidId {
                id: info.id.clone(),
            })?;

        let files = info
            .siblings
            .iter()
            .flatten()
            .map(|file| ModelFile {
                path: file.rfilename.clone(),
            })
            .collect();

        Ok(Self {
            id: info.id.clone(),
            author: author.to_string(),
            model: model.to_string(),
            revision: info.sha.clone(),
            library: info.library_name.clone(),
            pipeline: info.pipeline_tag.clone(),
            files,
            model_type: None,
            architectures: Vec::new(),
        })
    }

    pub fn has_onnx(&self) -> bool {
        self.selected_onnx().is_some()
    }

    pub fn selected_onnx(&self) -> Option<&ModelFile> {
        [
            "model.onnx",
            "onnx/model.onnx",
            "decoder_model.onnx",
            "onnx/decoder_model.onnx",
            "model_quantized.onnx",
            "onnx/model_quantized.onnx",
            "decoder_model_quantized.onnx",
            "onnx/decoder_model_quantized.onnx",
        ]
        .iter()
        .find_map(|name| self.files.iter().find(|file| file.path == *name))
        .or_else(|| self.files.iter().find(|file| file.path.ends_with(".onnx")))
    }

    pub fn inference_files(&self) -> impl Iterator<Item = &ModelFile> {
        let selected_onnx = self.selected_onnx().map(|file| file.path.as_str());
        self.files.iter().filter(move |file| {
            let path = file.path.to_ascii_lowercase();

            if path.ends_with(".onnx") && Some(file.path.as_str()) != selected_onnx {
                return false;
            }

            !path.ends_with(".bin")
                && !path.ends_with(".safetensors")
                && !path.ends_with(".pt")
                && !path.ends_with(".pth")
                && !path.ends_with(".ckpt")
                && !path.ends_with(".msgpack")
                && !path.ends_with(".h5")
                && !path.ends_with(".ot")
        })
    }
}

#[derive(Debug, Clone)]
pub struct ModelFile {
    pub path: String,
}
