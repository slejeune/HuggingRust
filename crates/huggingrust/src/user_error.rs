use std::fmt;

pub struct UserError(pub Box<dyn std::error::Error>);

impl fmt::Display for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(error) = self.0.downcast_ref::<importer::hub::ImportError>() {
            return write!(f, "{}", friendly_import_error(error));
        }
        if let Some(error) = self.0.downcast_ref::<inference::InferenceError>() {
            return write!(f, "Inference failed: {error}");
        }
        if let Some(error) = self.0.downcast_ref::<super::models::UnsupportedModel>() {
            return write!(f, "{error}");
        }
        if let Some(error) = self.0.downcast_ref::<hf_hub::HFError>() {
            return write!(f, "{}", friendly_hub_error(error));
        }
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for UserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

fn friendly_import_error(error: &importer::hub::ImportError) -> String {
    use importer::hub::ImportError as E;
    match error {
        E::Hub(error) => friendly_hub_error(error),
        E::NoOnnx { model } => format!("Model '{model}' has no ONNX export."),
        E::InvalidConfig { model, .. } => format!("Model '{model}' has an invalid config.json."),
        _ => error.to_string(),
    }
}

fn friendly_hub_error(error: &hf_hub::HFError) -> String {
    match error {
        hf_hub::HFError::RepoNotFound { repo_id, .. } => format!(
            "Model repository '{repo_id}' was not found. Check the owner and model name, and make sure you can access it."
        ),
        hf_hub::HFError::EntryNotFound { path, repo_id, .. } => {
            format!("File '{path}' was not found in model repository '{repo_id}'.")
        }
        hf_hub::HFError::RevisionNotFound {
            revision, repo_id, ..
        } => {
            format!("Revision '{revision}' was not found in model repository '{repo_id}'.")
        }
        hf_hub::HFError::AuthRequired { .. } | hf_hub::HFError::Forbidden { .. } => {
            "Hugging Face access was denied. Sign in or request access to this model.".into()
        }
        hf_hub::HFError::RateLimited { .. } => {
            "Hugging Face rate limit reached. Please try again later.".into()
        }
        _ => format!("Hugging Face request failed: {error}"),
    }
}
