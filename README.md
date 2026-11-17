# Cross-Architecture Causal Attention-Head Alignment and Pruning

Research repository for studying whether task-conditioned causal correspondence between attention heads in bidirectional encoder Transformers and causal decoder Transformers can transfer pruning utility.

**Status:** Stage 1 — research foundation in progress. No model experiments have been run yet. All numerical results must come from recorded experiment outputs; none are assumed.

## Research question

Can task-conditioned causal correspondence between heads in bidirectional encoder and causal decoder Transformers transfer pruning utility across architectures, and how reliable is that transfer under calibration and task-distribution shifts?

## Working hypothesis

Heads that exhibit similar task-conditioned causal effects may provide a more useful cross-architecture transfer signal than representation similarity or head-index matching alone. This is a hypothesis to test, not an established result.

## Planned stages

1. Research foundation: proposal, literature inventory, provisional novelty audit, definitions, protocol and reproducibility configuration.
2. Representation-only pilot: BERT-base-uncased and GPT-2 small; 256 shared texts; per-head representations; CKA; partial matching; bootstrap stability.
3. Predeclared go/no-go decision based on held-out stability and baseline comparisons.
4. SST-2 fine-tuning and head-level causal interventions.
5. Importance-transfer evaluation against target-specific and non-transfer baselines.
6. Physical structured head pruning and hardware benchmarking only if earlier evidence justifies it.

## Repository map

- `research/`: proposal, questions, hypotheses, novelty audit, protocol and literature.
- `configs/`: reproducible experiment settings.
- `src/`: implementation modules.
- `experiments/`: staged experiment entry points and outputs.
- `tests/`: unit and integration tests.
- `results/`: generated reports, tables and figures (large generated artifacts may be excluded from Git).
- `paper/`: manuscript outline and writing materials.

## Reproducibility principles

- Record model identifiers, tokenizer revisions, dataset versions, split identifiers, seeds, package versions and hardware.
- Never overwrite raw outputs; identify each run with a configuration hash or run ID.
- Keep calibration, matching, model-selection and final evaluation data separate.
- Use sequential model loading and small batches; support CPU fallback.
- Do not commit model checkpoints, private credentials, or copyrighted full-text papers.
- Distinguish implemented code, executed experiments and independently verified findings.

## Quick start

The representation-only pilot implementation is now present. Install model extras and run `python -m cra_hp.experiments.representation_pilot --config configs/feasibility.yaml`. The first full run downloads WikiText-2, BERT-base and GPT-2; it has **not yet been executed or validated on real model activations**. See [setup](docs/setup.md), [the research protocol](research/research_protocol.md) and [the feasibility configuration](configs/feasibility.yaml).

## Current status

| Stage | Status |
|---|---|
| 1. Research foundation | <span>Completed and merged to main; CI passed</span> |
| 2. Representation pilot | Implementation merged; full model run pending |
| 3. Go/no-go | Not started |
| 4. Causal correspondence | Not started |
| 5. Importance transfer | Not started |
| 6. Physical pruning | Not started |

## Research integrity

A positive CKA score is not evidence of causal equivalence. A stable assignment is not evidence of transferable pruning utility. Importance rankings must be validated against target-model interventions and held-out task performance. Physical speedups must be measured after structural removal on specified hardware.
