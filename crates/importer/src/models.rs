use hf_hub::repository::ModelInfo;

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub author: String,
    pub model: String,
    pub revision: Option<String>,
    pub library: Option<String>,
    pub pipeline: Option<String>,
    pub files: Vec<ModelFile>,
}

impl ModelMetadata {
    pub fn from_huggingface(info: &ModelInfo) -> Self {
        let (author, model) = info.id.split_once('/').unwrap_or(("", &info.id));

        let files = info
            .siblings
            .as_ref()
            .map(|files| {
                files
                    .iter()
                    .map(|file| ModelFile {
                        path: file.rfilename.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            id: info.id.clone(),
            author: author.to_string(),
            model: model.to_string(),
            revision: info.sha.clone(),
            library: info.library_name.clone(),
            pipeline: info.pipeline_tag.clone(),
            files,
        }
    }

    pub fn has_onnx(&self) -> bool {
        self.files.iter().any(|f| f.path.ends_with(".onnx"))
    }

    pub fn inference_files(&self) -> impl Iterator<Item = &str> {
        self.files.iter().map(|f| f.path.as_str()).filter(|path| {
            let path = path.to_ascii_lowercase();

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
