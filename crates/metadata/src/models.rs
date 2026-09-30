use hf_hub::repository::ModelInfo;

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub id: String,
    pub revision: Option<String>,
    pub author: Option<String>,
    pub library: Option<String>,
    pub pipeline: Option<String>,
    pub files: Vec<ModelFile>,
    pub onnx: Vec<String>,
}

impl ModelMetadata {
    pub fn from_huggingface(info: &ModelInfo) -> Self {
        let files: Vec<_> = info
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

        let onnx = files
            .iter()
            .filter(|file| file.path.ends_with(".onnx"))
            .map(|file| file.path.clone())
            .collect();

        Self {
            id: info.id.clone(),
            revision: info.sha.clone(),
            author: info.author.clone(),
            library: info.library_name.clone(),
            pipeline: info.pipeline_tag.clone(),
            files,
            onnx,
        }
    }

    pub fn has_onnx(&self) -> bool {
        !self.onnx.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ModelFile {
    pub path: String,
}
