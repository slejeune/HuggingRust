# HuggingRust

HuggingRust is a Rust workspace for running text models that are already exported to ONNX. It can load a local model directory or resolve and download an ONNX model from Hugging Face, then run supported tasks through ONNX Runtime.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `huggingrust` | Interactive CLI, prompts, model resolution workflow, and user-facing errors |
| `importer` | Hugging Face metadata, ONNX file selection, and local model downloads |
| `inference` | Tokenization, ONNX Runtime execution, model-family adapters, and task handling |

Model files used by the CLI are stored under `onnx_models/` by default. The directory is ignored by Git because model weights are large; local model directories can also be provided directly.

## Run

```sh
cargo run -p huggingrust
```

Enter either a local model directory or a Hugging Face identifier such as `owner/model`. Hub models need a compatible ONNX export and tokenizer files. Local directories need `config.json`, `tokenizer.json`, and an ONNX model. The CLI lists tasks inferred from model metadata and prompts for the inputs each task expects.

## Current scope

The runtime has adapters for encoder-only, decoder-only, and encoder-decoder text models. A recognized architecture alone does not guarantee compatibility: model graph input/output names, tokenizer features, and task-specific outputs must also match supported signatures. The project downloads existing ONNX exports; it does not currently export or convert models.

