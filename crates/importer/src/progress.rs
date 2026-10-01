use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub fn spinner(message: impl Into<String>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();

    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
    );

    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message(message.into());

    pb
}
