# Environment Setup

The repository contains Stage 2 representation and Stage 3 SST-2 causal pilot entry points.

## Requirements

- Python 3.10 or 3.11 recommended.
- Git.
- Internet access for downloading public model and dataset artifacts.
- Optional NVIDIA GPU with a compatible PyTorch/CUDA installation.

## Windows PowerShell

```powershell
git clone https://github.com/gourav-ai-engineer/endsem.git
cd endsem
git checkout research/foundation
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

## Linux/macOS

```bash
git clone https://github.com/gourav-ai-engineer/endsem.git
cd endsem
git checkout research/foundation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

## Optional model dependencies

Install model dependencies after selecting the correct PyTorch build for your machine:

```bash
pip install -e ".[models]"
```

For NVIDIA systems, follow the official PyTorch installation selector so the CUDA wheel matches the installed driver/runtime. Record the exact installation command and versions in the run manifest.

## Current status

The Stage 2 representation-only pilot is available as `python -m cra_hp.experiments.representation_pilot --config configs/feasibility.yaml`. The Stage 3 SST-2 fine-tuning and causal intervention pilot is available as `python -m cra_hp.experiments.causal_pilot --config configs/stage3_pilot.yaml`. Install model extras first. Stage 3 downloads the pinned SST-2 dataset, BERT and GPT-2; trains sequentially; then performs exact head ablations and gate-gradient measurements. It is compute intensive, especially on CPU. Its model checkpoints are written under the run directory and should not be committed. A labeled official validation split is used as the untouched outer evaluation set because the GLUE SST-2 test split has no labels.

The current Windows environment blocks the high-level `pyarrow.dataset` binary. The loader falls back to checksum-verified Parquet files from the pinned Hub revision using PyArrow's low-level reader, and caches those files under ignored `data/raw/`. Use `--seed 20261118` to run a predeclared replication seed; this changes both the training and train-partition seeds and records the effective config in that run's folder. If an earlier run stops after saving a model checkpoint, resume with `--resume-encoder-checkpoint <path-to-checkpoints/bert>`; the checkpoint's saved config must exactly match the requested pilot config. No experiment result should be inferred from implementation status alone.
