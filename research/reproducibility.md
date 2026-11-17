# Reproducibility and Data Governance

## Run manifest

Every experiment run must save a machine-readable manifest containing:

- Git commit SHA and dirty/clean working-tree state.
- UTC timestamp and run ID.
- Full resolved configuration and configuration hash.
- Python, OS, CPU, RAM, GPU, driver, CUDA, PyTorch, Transformers and relevant library versions.
- Model IDs, resolved model/tokenizer commit hashes and model configuration.
- Dataset name/config/version, split, license/source URL, sample IDs and preprocessing.
- Random seeds and deterministic-kernel settings.
- Command line, exit code, stdout/stderr paths.
- Output file hashes and metric summaries.

## Data separation

- Representation calibration examples are not final evaluation examples.
- Matching thresholds and score weights are chosen without final test labels/outcomes.
- SST-2 validation is used for model selection; final test evaluation remains untouched until analysis is frozen.
- If a public dataset does not provide a hidden test split through the selected loader, explicitly document which official split is used and do not describe validation as test.
- Distribution-shift calibration and evaluation samples must be separate.

## Artifact policy

Commit source code, small configs, metadata, citations, curated CSV tables and compact figures. Do not commit:
- model checkpoints or model weights;
- cached Hugging Face datasets;
- full copyrighted publisher PDFs;
- credentials, tokens or personal data;
- large raw activation dumps.

For large outputs, retain a manifest, summary metrics and reproducible generation command. Store full artifacts in an appropriate external storage location if needed, and record checksums/links.

## Reproducibility target

A second machine should be able to:
1. Clone the repository.
2. Create an isolated Python environment.
3. Install the documented dependencies.
4. Download public model/dataset artifacts through documented libraries.
5. Run unit tests.
6. Execute a small smoke experiment before the full configuration.
7. Regenerate tables and figures from saved result files.

## Hardware-aware execution

For a laptop GPU with limited VRAM/RAM:
- Load models sequentially.
- Use batch size 1–2 and short sequence lengths for initial extraction.
- Move cached activations to CPU and write chunked arrays to disk.
- Avoid retaining computational graphs during representation extraction.
- Use `torch.inference_mode()` for inference-only passes.
- Provide CPU fallback; expect it to be slower.
- Do not use quantization for the initial scientific comparison unless it is a separately controlled condition.
