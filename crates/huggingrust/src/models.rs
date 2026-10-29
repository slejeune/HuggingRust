use importer::ModelMetadata;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceFamily {
    EncoderOnly,
    DecoderOnly,
    Unknown,
}

#[derive(Debug, Error)]
#[error(
    "Model '{model}' is not supported (architecture: {architecture}). Supported: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa/DeBERTa-v2, ELECTRA, XLM-RoBERTa, GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, Mistral, Phi, OPT, Qwen2, Qwen2-MoE, and Qwen3."
)]
pub struct UnsupportedModel {
    model: String,
    architecture: String,
}

#[derive(Debug, Error)]
#[error(
    "Model '{model}' is tagged for '{task}', but this app supports decoder-only models for text generation. Embedding and other non-generation tasks are not supported."
)]
pub struct UnsupportedTask {
    model: String,
    task: String,
}

pub fn ensure_supported_task(
    model: &str,
    family: InferenceFamily,
    task: Option<&str>,
) -> Result<(), UnsupportedTask> {
    if family == InferenceFamily::DecoderOnly
        && let Some(task) = task.filter(|task| *task != "text-generation")
    {
        return Err(UnsupportedTask {
            model: model.to_owned(),
            task: task.to_owned(),
        });
    }
    Ok(())
}

pub fn inference_family_from_model_type(model_type: Option<&str>) -> InferenceFamily {
    match model_type {
        Some("bert") | Some("roberta") | Some("distilbert") | Some("albert") | Some("deberta")
        | Some("deberta-v2") | Some("electra") | Some("xlm-roberta") => {
            InferenceFamily::EncoderOnly
        }
        Some("gpt2") | Some("gpt_neo") | Some("gptj") | Some("gpt_neox") | Some("llama")
        | Some("mistral") | Some("phi") | Some("opt") | Some("qwen2") | Some("qwen2_moe")
        | Some("qwen3") => InferenceFamily::DecoderOnly,
        _ => InferenceFamily::Unknown,
    }
}

pub fn inference_family(metadata: &ModelMetadata) -> InferenceFamily {
    inference_family_from_model_type(metadata.model_type.as_deref())
}

pub fn ensure_supported(metadata: &ModelMetadata) -> Result<InferenceFamily, UnsupportedModel> {
    let family = inference_family(metadata);
    match family {
        InferenceFamily::EncoderOnly | InferenceFamily::DecoderOnly => Ok(family),
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
        InferenceFamily::DecoderOnly => {
            Ok(inference::InferenceFamily::load_decoder_only(artifact)?)
        }
        InferenceFamily::Unknown => Err("Unsupported model family".into()),
    }
}
