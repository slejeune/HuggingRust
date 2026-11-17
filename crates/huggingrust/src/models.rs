use importer::ModelMetadata;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceFamily {
    EncoderOnly,
    DecoderOnly,
    EncoderDecoder,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct TaskCapability {
    pub name: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub model_id: String,
    pub architecture: Option<String>,
    pub pipeline: Option<String>,
    pub task_tags: Vec<String>,
    pub tasks: Vec<TaskCapability>,
}

impl ModelCapabilities {
    pub fn from_metadata(metadata: &ModelMetadata, family: InferenceFamily) -> Self {
        Self::infer(
            metadata.id.clone(),
            family,
            metadata.model_type.as_deref(),
            &metadata.architectures,
            metadata.pipeline.as_deref(),
            &metadata.tags,
        )
    }

    pub fn infer(
        model_id: String,
        family: InferenceFamily,
        model_type: Option<&str>,
        architectures: &[String],
        pipeline: Option<&str>,
        tags: &[String],
    ) -> Self {
        let architecture = architectures
            .first()
            .map(String::as_str)
            .or(model_type)
            .map(str::to_owned);
        let mut tasks = Vec::new();
        let task_tags = tags
            .iter()
            .filter(|tag| Some(tag.as_str()) != pipeline && metadata_task(tag).is_some())
            .cloned()
            .collect::<Vec<_>>();
        if let Some(task) = pipeline.and_then(metadata_task) {
            add_task(&mut tasks, family, task);
        }
        for tag in tags {
            if let Some(task) = metadata_task(tag) {
                add_task(&mut tasks, family, task);
            }
        }
        if tasks.is_empty()
            && pipeline.is_none()
            && let Some(architecture) = architecture.as_deref()
        {
            for (marker, task) in [
                ("ForMaskedLM", "fill-mask"),
                ("ForSequenceClassification", "text-classification"),
                ("ForTokenClassification", "token-classification"),
                ("ForQuestionAnswering", "question-answering"),
                ("ForMultipleChoice", "multiple-choice"),
                ("ForCausalLM", "text-generation"),
                ("ForConditionalGeneration", "text2text-generation"),
            ] {
                if architecture.contains(marker) {
                    add_task(&mut tasks, family, task);
                }
            }
        }
        if tasks.is_empty() && pipeline.is_none() {
            let default_task = match family {
                InferenceFamily::EncoderOnly => "feature-extraction",
                InferenceFamily::DecoderOnly => "text-generation",
                InferenceFamily::EncoderDecoder => "text2text-generation",
                InferenceFamily::Unknown => {
                    return Self {
                        model_id,
                        architecture,
                        pipeline: pipeline.map(str::to_owned),
                        task_tags,
                        tasks,
                    };
                }
            };
            add_task(&mut tasks, family, default_task);
        }
        Self {
            model_id,
            architecture,
            pipeline: pipeline.map(str::to_owned),
            task_tags,
            tasks,
        }
    }
}

fn metadata_task(tag: &str) -> Option<&'static str> {
    Some(match tag {
        "feature-extraction" | "sentence-embeddings" => "feature-extraction",
        "text-generation" | "conversational" => "text-generation",
        "text2text-generation" | "text-to-text-generation" | "summarization" | "translation" => {
            "text2text-generation"
        }
        "text-classification" | "sentiment-analysis" => "text-classification",
        "token-classification" | "ner" => "token-classification",
        "question-answering" => "question-answering",
        "fill-mask" => "fill-mask",
        "zero-shot-classification" => "zero-shot-classification",
        "sentence-similarity" | "text-similarity" | "text-ranking" => "text-similarity",
        "multiple-choice" => "multiple-choice",
        _ => return None,
    })
}

fn add_task(tasks: &mut Vec<TaskCapability>, family: InferenceFamily, name: &'static str) {
    let name = match (family, name) {
        (InferenceFamily::EncoderDecoder, "text-generation") => "text2text-generation",
        (InferenceFamily::DecoderOnly, "text2text-generation") => "text-generation",
        _ => name,
    };
    let supported = match name {
        "text-generation" => matches!(family, InferenceFamily::DecoderOnly),
        "text2text-generation" => matches!(family, InferenceFamily::EncoderDecoder),
        "fill-mask" | "zero-shot-classification" | "multiple-choice" => {
            matches!(family, InferenceFamily::EncoderOnly)
        }
        _ => !matches!(family, InferenceFamily::Unknown),
    };
    if !supported || tasks.iter().any(|task| task.name == name) {
        return;
    }
    let description = match name {
        "feature-extraction" => "Create vector representations of text",
        "text-generation" => "Generate text from a prompt",
        "text2text-generation" => {
            "Run free-form text-to-text tasks using an instruction or task prefix"
        }
        "text-classification" => "Score text against the model's classes",
        "token-classification" => "Predict a class for each token",
        "question-answering" => "Answer a question from a context or prompt",
        "fill-mask" => "Predict the masked token",
        "zero-shot-classification" => "Rank candidate labels without task-specific training",
        "text-similarity" => "Compare two text embeddings",
        "multiple-choice" => "Score choices for a context",
        _ => "Run text inference",
    };
    tasks.push(TaskCapability { name, description });
}

#[derive(Debug, Error)]
#[error(
    "Model '{model}' is not supported (architecture: {architecture}). Supported: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa/DeBERTa-v2, ELECTRA, XLM-RoBERTa, GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, Mistral, Phi, OPT, Qwen2, Qwen2-MoE, Qwen3, T5, BART, mBART, and PEGASUS."
)]
pub struct UnsupportedModel {
    model: String,
    architecture: String,
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
        Some("t5") | Some("mt5") | Some("bart") | Some("mbart") | Some("pegasus")
        | Some("marian") | Some("blenderbot") | Some("long-t5") => InferenceFamily::EncoderDecoder,
        _ => InferenceFamily::Unknown,
    }
}

pub fn inference_family(metadata: &ModelMetadata) -> InferenceFamily {
    inference_family_from_model_type(metadata.model_type.as_deref())
}

pub fn ensure_supported(metadata: &ModelMetadata) -> Result<InferenceFamily, UnsupportedModel> {
    let family = inference_family(metadata);
    match family {
        InferenceFamily::EncoderOnly
        | InferenceFamily::DecoderOnly
        | InferenceFamily::EncoderDecoder => Ok(family),
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
        InferenceFamily::EncoderDecoder => {
            Ok(inference::InferenceFamily::load_encoder_decoder(artifact)?)
        }
        InferenceFamily::Unknown => Err("Unsupported model family".into()),
    }
}
