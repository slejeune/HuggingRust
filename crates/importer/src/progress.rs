use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub fn spinner(message: impl Into<String>) -> Result<ProgressBar, indicatif::style::TemplateError> {
    let pb = ProgressBar::new_spinner();

    pb.set_style(ProgressStyle::with_template("{spinner:.green} {msg}")?.tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "));

    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message(message.into());

    Ok(pb)
}
