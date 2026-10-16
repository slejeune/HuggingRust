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

pub fn inference_family(metadata: &ModelMetadata) -> InferenceFamily {
    match metadata.model_type.as_deref() {
        Some("bert") | Some("roberta") | Some("distilbert") | Some("albert") | Some("deberta")
        | Some("deberta-v2") | Some("electra") | Some("xlm-roberta") => {
            InferenceFamily::EncoderOnly
        }
        _ => InferenceFamily::Unknown,
    }
}

pub fn ensure_supported(metadata: &ModelMetadata) -> Result<InferenceFamily, UnsupportedModel> {
    let family = inference_family(metadata);
    match family {
        InferenceFamily::EncoderOnly => Ok(family),
        InferenceFamily::Unknown => Err(UnsupportedModel {
            model: metadata.id.clone(),
            architecture: metadata
                .model_type
                .as_deref()
                .or_else(|| metadata.architectures.first().map(String::as_str))
                .unwrap_or("unknown")
                .to_owned(),
        }),
    }
}

pub fn load_inference(
    family: InferenceFamily,
    artifact: inference::ModelArtifact,
) -> Result<inference::InferenceFamily, Box<dyn std::error::Error>> {
    match family {
        InferenceFamily::EncoderOnly => {
            Ok(inference::InferenceFamily::load_encoder_only(artifact)?)
        }
        InferenceFamily::Unknown => Err("Unsupported model family".into()),
    }
}
