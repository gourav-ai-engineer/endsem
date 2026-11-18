# Cross-Architecture Causal Attention-Head Alignment and Pruning

Research repository for studying whether task-conditioned causal correspondence between attention heads in bidirectional encoder Transformers and causal decoder Transformers can transfer pruning utility.

**Status:** First Stage 3 causal pilot completed on one seed. Multi-seed reproducibility and pruning-utility transfer remain untested.

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

The Stage 2 analysis and first Stage 3 SST-2 causal pilot results are recorded. Install model extras and run `python -m cra_hp.experiments.causal_pilot --config configs/stage3_pilot.yaml` for a new seed. See [setup](docs/setup.md), [the research protocol](research/research_protocol.md), [the Stage 3 pilot configuration](configs/stage3_pilot.yaml), the [Stage 2 analysis](research/stage2_corrected_analysis.md), and the [Stage 3 analysis](research/stage3_pilot_analysis.md).

## Current status

| Stage | Status |
|---|---|
| 1. Research foundation | Completed and merged to main |
| 2. Representation pilot | Corrected pilot completed; analysis recorded |
| 3. Go/no-go | Passed for exploratory Stage 3 only; full mapping is unstable |
| 4. Causal correspondence | One exploratory seed completed; multi-seed exit condition not met |
| 5. Importance transfer | Not started |
| 6. Physical pruning | Not started |

## Research integrity

A positive CKA score is not evidence of causal equivalence. A stable assignment is not evidence of transferable pruning utility. Importance rankings must be validated against target-model interventions and held-out task performance. Physical speedups must be measured after structural removal on specified hardware.
