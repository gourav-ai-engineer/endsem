# Experimental Protocol and Decision Rules

**Protocol status:** Version 0.1, to be frozen before confirmatory runs.

## General controls

- Models: `bert-base-uncased` and `gpt2`; record exact Hugging Face revision and tokenizer revision.
- Initial expected architecture: 12 layers × 12 heads; assert actual config.
- Device: record CPU/GPU, driver, CUDA, PyTorch and Transformers versions.
- Seed: default 20261117; run multiple seeds for confirmatory work.
- Deterministic settings: set Python, NumPy and PyTorch seeds; record whether deterministic kernels are enabled and any performance implications.
- Sequential loading: do not keep both models resident on a 4 GB GPU simultaneously.
- Store raw run config, stdout/stderr, metrics JSON/CSV and environment manifest for every run.
- No dataset/test leakage into matching or threshold selection.

## Stage 1 — Research foundation

Deliverables:
- Proposal, research questions, hypotheses, provisional novelty audit.
- Machine-readable literature inventory and bibliography.
- Mathematical definitions and protocol.
- Feasibility and main experiment configurations.
- Dependency/environment plan and tests.

Exit condition: all methods, primary metrics, data splits and go/no-go rules are documented; unresolved literature items are labeled.

## Stage 2 — Representation-only feasibility pilot

### Data

- 256 shared raw text examples from a documented public corpus (initial candidate WikiText-2).
- Stable example IDs and deterministic sampling.
- Use identical raw texts for both models. Tokenization, truncation and masks are model-specific and saved.
- Maximum sequence length 64 for the first pilot.
- Keep any final evaluation set separate.

### Extraction

- Extract every attention head's output contribution for each example.
- Initial extraction target: 144 heads per model; verify actual model config.
- Save pooled representations with metadata: model/revision, layer, head, sample ID, pooling, dtype and token count.
- Compare masked mean pooling first. Run designated readout and alternative pooling as sensitivity analyses.
- Do not equate the head context vector with its post-output-projection contribution; document the exact hook location and use one consistent definition per experiment.

### Similarity and matching

- Compute the 144 × 144 linear CKA matrix.
- Run partial global assignment with dummy unmatched nodes.
- Baselines: same-index, same-layer constrained, random and sample-permutation null.
- Save raw similarity matrix, assignment, unmatched heads and thresholds.
- Plot heatmap with layer/head axes and matched-pair overlays.

### Bootstrap

- Debug: 5 replicates.
- Pilot report: at least 50 paired-example bootstrap replicates.
- Main report: target 200 if resources permit.
- Report match frequency, assignment stability, Jaccard and uncertainty.

### Representation-only go/no-go

Proceed to causal experiment only if:
1. The pipeline passes shape, determinism and numerical tests.
2. Similarity/matching structure exceeds a permutation/random null under a predeclared statistic.
3. At least some matches have repeatable bootstrap support (report distribution; do not cherry-pick a single threshold).
4. Results are not wholly explained by same-layer or same-index baselines.

This is a feasibility gate, not proof of causal transfer. If these conditions fail, diagnose representation extraction and pooling before changing the hypothesis. Record negative outcomes.

## Stage 3 — Causal correspondence on SST-2

- Use official SST-2 train/validation/test splits; record dataset revision and preprocessing.
- Train/fine-tune BERT and GPT-2 under a documented common classification protocol.
- The GPT-2 readout (e.g. last non-padding token) and BERT readout (e.g. CLS) are architecturally different; report this as a limitation and include a readout/pooling sensitivity study.
- Match optimization budget, training examples, evaluation examples and selection criteria as closely as possible. Record any unavoidable differences.
- Compute per-example exact head-ablation effects on validation/calibration examples; compute gradient proxies separately.
- Compare causal signature similarity against representation-only matching.
- Validate on held-out examples. Never use test data to form matches.

Exit condition: causal signatures are reproducible across seeds and exact interventions; otherwise report failure and do not claim causal correspondence.

## Stage 4 — Importance transfer

Compare:
1. Random target head ranking.
2. Same-index source-to-target ranking.
3. Unaligned source ranking.
4. CKA-aligned source importance.
5. Causal-aligned source importance (proposed).
6. Target-specific gradient importance.
7. HIES or another justified target-specific baseline, if implementation and citation are verified.
8. Target exact-ablation ranking as an oracle/reference, not a deployable method.

Metrics:
- Spearman/Kendall correlation with target single-head utility.
- Top-k overlap and precision/recall against target oracle ranking.
- Held-out target loss/accuracy/F1 after removing selected sets at fixed budgets.
- Confidence calibration and abstention coverage/risk.
- Group-removal utility on a selected subset to quantify interactions.

Use calibration budgets 128, 512 and 2048 where dataset size permits. Evaluate in-domain and shifted calibration separately. All ranking construction must precede target test evaluation.

Go/no-go: proceed to physical pruning only if proposed transfer beats relevant baselines on held-out target utility with uncertainty reported and does not rely on test leakage.

## Stage 5 — Physical structured pruning

- Remove head-specific slices from query/key/value projections and corresponding output-projection input dimensions; update model configuration and validate tensor shapes.
- Account for architecture-specific implementations; BERT and GPT-2 projection layouts differ.
- Test numerical equivalence before pruning and expected output change after pruning.
- Compare gated ablation against physically pruned model outputs.
- Evaluate multiple pruning ratios (10%, 20%, 30%, 40%) only if justified by prior results.
- Optional recovery fine-tuning must be a separate condition and use training data only.

Measure:
- Accuracy, F1, validation/test loss.
- Parameter count and FLOPs (tool and version recorded).
- Batch-1 and larger-batch latency, throughput and peak memory.
- Fixed sequence lengths, warm-up count, repetitions, synchronization and device details.
- Report medians and dispersion; do not claim speedup from parameter reduction alone.

## Stage 6 — Reporting

- Freeze final configs and code commit.
- Generate tables/figures from machine-readable results.
- Include negative results, confidence intervals, limitations and exact hardware.
- Provide commands to reproduce every figure/table.
- Archive run manifests and data provenance. Do not commit model weights or copyrighted PDFs.

## Run naming

Use `<stage>_<modelpair>_<dataset>_<seed>_<short-config-hash>`. Store a JSON manifest alongside every output. Never overwrite an existing run directory.
