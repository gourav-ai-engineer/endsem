# Environment Setup

The repository foundation is documentation/configuration-first. The model-experiment entry points will be added in Stage 2.

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

At Stage 1, only repository-foundation smoke tests are defined. Model extraction and CKA commands do not exist yet; they will be implemented and tested in Stage 2. Do not interpret a passing foundation test as evidence that model experiments have run.
