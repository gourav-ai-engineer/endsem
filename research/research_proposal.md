# Research Proposal

## Working title

**Causal-Response Alignment for Cross-Architecture Attention-Head Pruning (CRA-HP)**

The acronym and title are provisional and should be revisited after the literature audit.

## Abstract

Attention-head pruning can reduce Transformer computation, but head importance is architecture-, task- and data-dependent. This project investigates whether task-conditioned causal effects of attention heads can be aligned across two distinct language-model architectures—a bidirectional encoder (BERT) and an autoregressive decoder (GPT-2)—and whether the resulting correspondence transfers pruning utility more reliably than representation-only or index-based matching. The study proceeds from representation similarity to causal intervention signatures, then to importance transfer and, only if justified, physical structured pruning. It explicitly evaluates calibration-data size and distribution shift. The central claim will be considered supported only if cross-architecture transfer improves over relevant baselines on held-out target data and survives uncertainty analysis.

## Motivation

Prior work has established that many attention heads can be pruned, that importance varies across tasks and datasets, and that head alignment or transfer has been explored in related settings. These findings make broad claims such as "attention heads are redundant" or "head pruning transfers across domains" insufficiently novel. The narrower question here is whether *task-conditioned causal response* can provide useful correspondence between bidirectional encoder and causal decoder heads, and whether that correspondence transfers pruning decisions.

## Research question

Can task-conditioned causal correspondence between heads in bidirectional encoder and causal decoder Transformers transfer pruning utility across architectures, and how reliable is that transfer under calibration and task-distribution shifts?

## Objectives

1. Build a reproducible literature and methods foundation.
2. Measure representation similarity between all 144 heads of BERT-base-uncased and GPT-2 small on common text inputs.
3. Test whether partial head assignments are stable under resampling and outperform simple matching baselines.
4. Construct task-conditioned causal-response signatures on SST-2 using a consistent classification protocol.
5. Test whether source-model head importance, transferred through correspondence, predicts target-model pruning utility.
6. Evaluate physical structured pruning only if the transfer results justify it.

## Scope

### Initial models

- `bert-base-uncased` (12 layers, 12 heads per layer).
- `gpt2` (12 layers, 12 heads per layer).

The expected 144 heads per model must be confirmed from the loaded model configuration at runtime; code must not silently assume this for arbitrary checkpoints.

### Initial datasets

- 256 shared English text examples for representation-only feasibility, sampled deterministically from a documented public corpus such as WikiText-2.
- SST-2 for a shared classification task.
- A held-out domain-shift dataset (candidate: IMDb or Yelp Polarity) only after the in-domain pipeline is validated.

Dataset licenses, versions, preprocessing and split provenance must be recorded.

## Main methodological idea

For each head, collect (a) task-relevant head-output representations and (b) task-conditioned effects of intervening on that head. Use representation similarity as an exploratory signal and causal-response similarity as the primary candidate correspondence signal. Solve a partial assignment problem that allows unmatched heads. Transfer source importance only through sufficiently confident correspondences. Evaluate transferred rankings by actually removing or ablating target heads and measuring held-out loss/performance changes.

## Hypotheses

- H1: Representation similarity yields nontrivial cross-architecture structure, but it is not by itself sufficient to establish causal correspondence.
- H2: Causal-response signatures are more predictive of target head-ablation effects than representation-only matching.
- H3: Confidence-aware partial matching transfers pruning utility better than same-index, random and unaligned-source baselines.
- H4: Transfer quality decreases as calibration distribution shift increases; confidence estimates should identify some unreliable matches.
- H5: Utility transfer does not guarantee hardware speedup; measured speedup depends on physical implementation and runtime.

These hypotheses may be rejected. No result is assumed.

## Key risks and mitigations

| Risk | Mitigation |
|---|---|
| BERT and GPT-2 have different tokenization/attention masks | Use identical raw texts, record tokenization separately, and compare only explicitly defined pooled/head-output representations. |
| Different readout semantics | Use a common classification protocol and report pooling/readout variants as separate experiments. |
| Forced one-to-one matching | Use partial assignment with dummy unmatched nodes and a predeclared rejection threshold. |
| Small bootstrap count | Use 5 resamples only for debugging; use at least 50 for pilot inference and 200 for final stability estimates if compute permits. |
| Gradient importance differs from actual ablation utility | Validate against exact single-head ablation and group-removal measurements. |
| Head interactions | Include pairwise/group ablation analysis on a subset. |
| Calibration leakage | Keep calibration, validation and final test sets strictly separated. |
| Claimed speedup from zero gates | Physically slice projection parameters and benchmark the resulting architecture. |
| Limited compute | Sequential model loading, small batches, short sequence lengths, CPU fallback and staged execution. |

## Expected contributions (conditional)

The project may contribute:
1. A controlled empirical study of head correspondence between BERT and GPT-2.
2. A task-conditioned causal-response signature and confidence-aware partial matching procedure.
3. An evaluation of cross-architecture pruning-utility transfer under calibration shift.
4. A reproducible implementation and, if supported, a physical pruning benchmark.

These are intended contributions, not novelty claims. The final manuscript must be adjusted to what the completed literature audit and experiments actually establish.

## Success criteria

A successful scientific outcome does not require positive results. A reproducible negative result showing that correspondence does not transfer utility is still useful if baselines, uncertainty and limitations are rigorous. The project succeeds methodologically if each stage has auditable configs, tests, raw metrics, logs and a clear go/no-go decision.
