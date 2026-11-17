use importer::ModelMetadata;
use thiserror::Error;

pub use inference::ModelFamily;

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
    pub fn from_metadata(metadata: &ModelMetadata, family: ModelFamily) -> Self {
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
        family: ModelFamily,
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
                ModelFamily::EncoderOnly => "feature-extraction",
                ModelFamily::DecoderOnly => "text-generation",
                ModelFamily::EncoderDecoder => "text2text-generation",
                ModelFamily::Unknown => {
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
    inference::canonical_task_name(tag)
}

fn add_task(tasks: &mut Vec<TaskCapability>, family: ModelFamily, name: &'static str) {
    let name = match (family, name) {
        (ModelFamily::EncoderDecoder, "text-generation") => "text2text-generation",
        (ModelFamily::DecoderOnly, "text2text-generation") => "text-generation",
        _ => name,
    };
    let supported = match name {
        "text-generation" => matches!(family, ModelFamily::DecoderOnly),
        "text2text-generation" => matches!(family, ModelFamily::EncoderDecoder),
        "fill-mask" | "zero-shot-classification" | "multiple-choice" => {
            matches!(family, ModelFamily::EncoderOnly)
        }
        _ => !matches!(family, ModelFamily::Unknown),
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

pub fn inference_family(metadata: &ModelMetadata) -> ModelFamily {
    ModelFamily::from_model_type(metadata.model_type.as_deref())
}

pub fn ensure_supported(metadata: &ModelMetadata) -> Result<ModelFamily, UnsupportedModel> {
    let family = inference_family(metadata);
    match family {
        ModelFamily::EncoderOnly | ModelFamily::DecoderOnly | ModelFamily::EncoderDecoder => {
            Ok(family)
        }
        ModelFamily::Unknown => Err(UnsupportedModel {
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
