mod input;
mod models;
mod user_error;
mod workflow;

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (artifact, family, capabilities) =
        workflow::load_model(input::prompt_model_input()?).await?;
    let mut inference = family.load(artifact)?;
    let task = input::choose_task(&capabilities)?;

    let task_input = match task.as_str() {
        "question-answering" => inference::TaskInput::Pair {
            first: input::prompt_text("Question")?,
            second: input::prompt_text("Context")?,
        },
        "text-similarity" | "sentence-similarity" | "text-ranking" => inference::TaskInput::Pair {
            first: input::prompt_text("First text")?,
            second: input::prompt_text("Second text")?,
        },
        "zero-shot-classification" => inference::TaskInput::CandidateLabels {
            text: input::prompt_text("Text")?,
            labels: input::prompt_text("Candidate labels (comma-separated)")?
                .split(',')
                .map(str::trim)
                .filter(|label| !label.is_empty())
                .map(str::to_owned)
                .collect(),
            hypothesis_template: "This example is {}.".into(),
        },
        "multiple-choice" => inference::TaskInput::Choices {
            context: input::prompt_text("Context")?,
            choices: input::prompt_text("Choices (comma-separated)")?
                .split(',')
                .map(str::trim)
                .filter(|choice| !choice.is_empty())
                .map(str::to_owned)
                .collect(),
        },
        _ => inference::TaskInput::Text(input::prompt_task_text(family, &task)?),
    };
    let max_new_tokens = 128;
    let output = inference.run_task(&task, task_input, max_new_tokens)?;
    print_task_output(output);

    Ok(())
}

fn print_task_output(output: inference::TaskOutput) {
    match output {
        inference::TaskOutput::GeneratedText(text) => println!("generated text: {text}"),
        inference::TaskOutput::Embedding(values) => {
            println!("embedding values: {}", values.len())
        }
        inference::TaskOutput::ClassificationScores(scores) => println!("class scores: {scores:?}"),
        inference::TaskOutput::TokenPredictions(predictions) => {
            println!("token predictions: {} tokens", predictions.len())
        }
        inference::TaskOutput::QuestionAnswering {
            start_token,
            end_token,
        } => println!("answer token span: {start_token}..={end_token}"),
        inference::TaskOutput::FillMaskPrediction { token, score } => {
            println!("masked token prediction: {token} ({score})")
        }
        inference::TaskOutput::ZeroShotScores(scores) => println!("candidate scores: {scores:?}"),
        inference::TaskOutput::Similarity(score) => println!("similarity: {score}"),
        inference::TaskOutput::MultipleChoiceScores(scores) => {
            println!("choice scores: {scores:?}")
        }
    }
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("Error: {}", user_error::UserError(error));
        std::process::exit(1);
    }
}
