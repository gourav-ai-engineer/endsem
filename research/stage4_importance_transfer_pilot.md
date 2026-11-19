# Stage 4 importance-transfer pilot

**Run:** `transfer_seed20261121_sst2_20261119T075016_873545Z`  
**Analysis code revision:** `7509900e6ff27513933c92480f2f3788e7724b6f`  
**Source/target models:** BERT-base-uncased → GPT-2  
**Status:** One low-power held-out diagnostic completed; utility transfer was not established.

## Design

The source BERT ranking and target GPT-2 gradient ranking were frozen from the seed 20261121 calibration artifacts. The causal alignment covered 23 of 144 target heads; the Stage 2 pretrained-representation CKA map covered 143 of 144. Target exact single-head effects were measured on a stratified 128-example sample (63 negative, 65 positive) from the pinned labeled SST-2 validation split. The validation sample was not used to build the source rankings or mappings. A fixed 10% group-removal budget removed 14 GPT-2 heads. All grouped ablations were gated interventions, not physically pruned models.

The official dataset revision was `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`; the repository's `main` SHA was checked to match before fetching the rows. The selected validation IDs and snapshot hash are recorded in the run manifest. The ranking feature was the signed mean loss increase after removing one head; lower scores were treated as more removable. The target exact-ablation ranking uses this same validation sample and is reported only as a non-deployable oracle reference.

## Ranking agreement with target held-out utility

| Ranking/mapping | Coverage | Spearman ρ (95% bootstrap interval) | Kendall τ | Least-utility top-10% overlap |
|---|---:|---:|---:|---:|
| Causal-aligned BERT source | 23/144 (16.0%) | **-0.286** [-0.401, -0.152] | -0.217 | 0.50 (2 heads) |
| Same-layer constrained mapping | 23/144 (16.0%) | -0.348 [-0.469, -0.253] | -0.217 | 0.00 (2 heads) |
| Stage 2 representation-CKA mapping | 143/144 (99.3%) | -0.014 [-0.077, 0.060] | -0.009 | 0.143 (14 heads) |
| Same-index mapping | 144/144 (100%) | -0.060 [-0.191, 0.041] | -0.044 | 0.071 (14 heads) |
| Target-gradient proxy | 144/144 (100%) | +0.159 [-0.059, 0.218] | +0.132 | 0.286 (14 heads) |

Intervals use 500 label-stratified bootstrap resamples of the 128 held-out examples. At fixed cardinality 23, random source-target mappings had mean Spearman ρ = -0.0046 (SD 0.2289; empirical 95% interval [-0.409, 0.454]) and mean least-utility top-10% overlap 0.085 (SD 0.214). The causal-aligned ranking was negatively associated with target held-out utility in this run; it did not outperform the random mapping distribution or target-gradient comparison.

## Joint 14-head ablation on the held-out sample

| Removed-head selection | Coverage | Mean loss | Accuracy | Positive F1 |
|---|---:|---:|---:|---:|
| No ablation | — | 0.3625 | 0.8594 | 0.8500 |
| Causal-aligned source ranking | 23 | 0.3639 | 0.8438 | 0.8305 |
| Same-layer constrained source ranking | 23 | 0.3755 | 0.8672 | 0.8571 |
| Representation-CKA source ranking | 143 | 0.3631 | 0.8438 | 0.8333 |
| Same-index source ranking | 144 | 0.6663 | 0.6641 | 0.5057 |
| Target-gradient ranking | 144 | 0.4025 | 0.8047 | 0.8031 |
| Random target sets, 20 draws (mean ± SD) | 144 | 0.4264 ± 0.1103 | 0.8125 ± 0.0667 | 0.7759 ± 0.1238 |
| Target exact-ablation oracle† | 144 | 0.2919 | 0.8672 | 0.8722 |

†The oracle selects the 14 individually most removable heads using exact target effects computed on this same sample. It is an optimistic reference, not a deployable ranking. Group effects are non-additive, so the oracle's joint result need not match the sum of individual effects.

The causal-aligned group slightly increased loss (+0.0013 versus no ablation) and reduced accuracy by 1.56 percentage points. Its loss was marginally worse than the CKA-aligned group, and no uncertainty interval for grouped performance was estimated from this small single run. Same-index removal was particularly damaging. These results do not show useful causal-ranking transfer.

## Decision and next experiment

This Stage 4 pilot does not meet the transfer go/no-go condition. Stage 3 pair identities did not recur across seeds, and the single Stage 4 causal ranking was inversely associated with target validation utility. Do not proceed to physical pruning or speed benchmarking.

If continuing the research, repeat the ranking-transfer diagnostic across the predeclared Stage 3 seeds and evaluate the planned 512- and 2,048-example calibration budgets where data and compute permit. Freeze all rankings before evaluating the same labeled validation sample, report paired uncertainty across examples and seeds, and retain the random, representation, same-index, same-layer, target-gradient, and target-oracle comparisons. The current 128-example result is a pilot, not a final confirmatory test.

## Artifacts

The full inputs, mappings, per-head effects, group-ablation predictions, summary metrics, and dataset manifest are under `results/04_importance_transfer/transfer_seed20261121_sst2_20261119T075016_873545Z/`.
