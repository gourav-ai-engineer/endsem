# Research Questions and Hypotheses

## Primary research question

Can task-conditioned causal correspondence between heads in bidirectional encoder and causal decoder Transformers transfer pruning utility across architectures, and how reliable is that transfer under calibration and task-distribution shifts?

## Subquestions

- RQ1 — Representation: Do head-output representations from BERT-base-uncased and GPT-2 small exhibit reproducible cross-model similarity on shared inputs?
- RQ2 — Assignment: Does partial global matching outperform same-index, layer-constrained and random assignments?
- RQ3 — Causality: Do matched heads exhibit similar per-example effects on a common downstream classification margin?
- RQ4 — Transfer: Does transferring source importance through causal correspondence predict target head-ablation utility better than representation-only or index-based transfer?
- RQ5 — Shift: How do matching confidence and transfer performance change with calibration sample size and distribution shift?
- RQ6 — Compression: Does the selected target-head set yield useful accuracy/compute trade-offs after actual structural pruning?

## Hypotheses (preliminary)

- H1: Cross-model representation similarity is structured above a random-matching null, but the signal may be weak or layer-dependent.
- H2: Causal-response similarity is more predictive of target ablation effects than representation CKA alone.
- H3: Confidence-weighted partial transfer outperforms same-index and unaligned-source importance baselines on held-out target data.
- H4: Transfer utility degrades under increasing calibration shift; match confidence and abstention reduce harmful transfer.
- H5: Structural pruning can reduce parameter/FLOP counts, but latency improvement is hardware- and implementation-dependent.

## Falsification conditions

- H1 is unsupported if observed matching scores do not exceed random/permuted null distributions or are unstable across resamples.
- H2 is unsupported if causal matching does not improve held-out prediction of target ablation effects over representation-only matching.
- H3 is unsupported if transferred rankings fail to beat random, same-index or unaligned-source baselines within uncertainty.
- H4 is unsupported if confidence does not predict transfer error or if abstention does not help under shift.
- H5 is unsupported if physical pruning does not improve measured efficiency at an acceptable quality cost.

## Primary outcomes

1. CKA matrix and partial assignment score.
2. Bootstrap match frequency and assignment stability.
3. Correlation between source-predicted and target-observed per-head ablation utility.
4. Top-k overlap and target loss/metric change after removing selected heads.
5. Accuracy/F1, parameters, FLOPs, latency, throughput and peak memory after structural pruning.

## Analysis discipline

All thresholds, primary comparisons, calibration splits and decision rules must be frozen in `research/research_protocol.md` before confirmatory experiments. Exploratory analyses must be labeled as such.
