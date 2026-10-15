use importer::ModelMetadata;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceFamily {
    EncoderOnly,
    Unknown,
}

#[derive(Debug, Error)]
#[error(
    "Model '{model}' is not supported (architecture: {architecture}). Supported: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, ELECTRA, XLM-RoBERTa."
)]
pub struct UnsupportedModel {
    model: String,
    architecture: String,
}

pub fn infer_inference_family(metadata: &ModelMetadata) -> InferenceFamily {
    match metadata.model_type.as_deref() {
        Some("bert") | Some("roberta") | Some("distilbert") | Some("albert") | Some("deberta")
        | Some("deberta-v2") | Some("electra") | Some("xlm-roberta") => {
            InferenceFamily::EncoderOnly
        }

        _ => InferenceFamily::Unknown,
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub metadata: ModelMetadata,
    pub inference_family: InferenceFamily,
}

impl Model {
    pub fn new(metadata: ModelMetadata, inference_family: InferenceFamily) -> Self {
        Self {
            metadata,
            inference_family,
        }
    }

    pub fn from_metadata(metadata: ModelMetadata) -> Self {
        let inference_family = infer_inference_family(&metadata);
        Self::new(metadata, inference_family)
    }
}

impl From<ModelMetadata> for Model {
    fn from(metadata: ModelMetadata) -> Self {
        Self::from_metadata(metadata)
    }
}

impl InferenceFamily {
    pub fn ensure_supported(self, metadata: &ModelMetadata) -> Result<(), UnsupportedModel> {
        if self == Self::Unknown {
            let architecture = metadata
                .model_type
                .as_deref()
                .or_else(|| metadata.architectures.first().map(String::as_str))
                .unwrap_or("unknown");
            return Err(UnsupportedModel {
                model: metadata.id.clone(),
                architecture: architecture.to_owned(),
            });
        }
        Ok(())
    }
}
