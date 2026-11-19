# Stage 4 importance-transfer pilot

**Run:** `transfer_seed20261121_sst2_20261119T075016_873545Z`  
**Analysis code revision:** `7509900e6ff27513933c92480f2f3788e7724b6f`  
**Source/target models:** BERT-base-uncased → GPT-2  
**Status:** Three low-power held-out diagnostics completed; utility transfer was not established. Two predeclared seed runs remain.

## Design

The source BERT ranking and target GPT-2 gradient ranking were frozen from the seed 20261121 calibration artifacts. The causal alignment covered 23 of 144 target heads; the Stage 2 pretrained-representation CKA map covered 143 of 144. Target exact single-head effects were measured on a stratified 128-example sample (63 negative, 65 positive) from the pinned labeled SST-2 validation split. The validation sample was not used to build the source rankings or mappings. A fixed 10% group-removal budget removed 14 GPT-2 heads. All grouped ablations were gated interventions, not physically pruned models.

The official dataset revision was `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`; the repository's `main` SHA was checked to match before fetching the rows. The selected validation IDs and snapshot hash are recorded in the run manifest. The ranking feature was the signed mean loss increase after removing one head; lower scores were treated as more removable. The target exact-ablation ranking uses this same validation sample and is reported only as a non-deployable oracle reference.

## Ranking agreement with target held-out utility

Each source seed uses the same 128 validation IDs and deterministic bootstrap seed. Intervals below quantify example-sampling uncertainty within a seed; they are not cross-seed intervals.

| Ranking/mapping | Coverage | Spearman ρ (95% bootstrap interval) | Kendall τ | Least-utility top-10% overlap |
|---|---:|---:|---:|---:|
| Causal-aligned BERT source | 23/144 (16.0%) | **-0.286** [-0.401, -0.152] | -0.217 | 0.50 (2 heads) |
| Same-layer constrained mapping | 23/144 (16.0%) | -0.348 [-0.469, -0.253] | -0.217 | 0.00 (2 heads) |
| Stage 2 representation-CKA mapping | 143/144 (99.3%) | -0.014 [-0.077, 0.060] | -0.009 | 0.143 (14 heads) |
| Same-index mapping | 144/144 (100%) | -0.060 [-0.191, 0.041] | -0.044 | 0.071 (14 heads) |
| Target-gradient proxy | 144/144 (100%) | +0.159 [-0.059, 0.218] | +0.132 | 0.286 (14 heads) |

Intervals use 500 label-stratified bootstrap resamples of the 128 held-out examples. For seed 20261121, at fixed cardinality 23, random source-target mappings had mean Spearman ρ = -0.0046 (SD 0.2289; empirical 95% interval [-0.409, 0.454]) and mean least-utility top-10% overlap 0.085 (SD 0.214). The causal-aligned ranking was negatively associated with target held-out utility in this run; it did not outperform the random mapping distribution or target-gradient comparison.

### Seed 20261120

**Run:** `transfer_seed20261120_sst2_20261119T080117_537534Z`  
**Analysis code revision:** `34023078f63e98d7ee213f69fc3cf511df2e8d45`

The same held-out IDs were evaluated against the seed 20261120 BERT/GPT-2 pair. The causal map covered 16 heads (11.1%). Its Spearman correlation was **-0.300** (95% interval [-0.418, -0.018]); same-layer was -0.453 [-0.512, -0.082]. The representation-CKA and same-index baselines were near zero (+0.046 and +0.098, both intervals include zero). The target-gradient proxy was positively correlated at +0.566 [0.111, 0.650]. Random fixed-cardinality mappings had mean Spearman +0.020 (SD 0.262; empirical 95% interval [-0.471, 0.460]).

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

For seed 20261120, the no-ablation GPT-2 baseline was loss 0.3297, accuracy 0.8672, and positive F1 0.8741. The causal-aligned group had loss 0.3066 with the same accuracy and F1 0.8722; the target-gradient group had loss 0.3019, accuracy 0.8594, and F1 0.8525. Random groups averaged loss 0.3345 ± 0.0455, accuracy 0.8730 ± 0.0156, and F1 0.8810 ± 0.0116. The target exact-ablation oracle group raised loss to 0.3704 and lowered accuracy to 0.8203, despite selecting heads by best individual effects on the same sample. This contrast with seed 20261121's oracle result shows that individual-head utility does not reliably predict joint-removal effects in this pilot.

| Seed 20261120 grouped selection | Mean loss | Accuracy | Positive F1 |
|---|---:|---:|---:|
| No ablation | 0.3297 | 0.8672 | 0.8741 |
| Causal-aligned source ranking | 0.3066 | 0.8672 | 0.8722 |
| Same-layer constrained source ranking | 0.3291 | 0.8672 | 0.8741 |
| Representation-CKA source ranking | 0.4175 | 0.7891 | 0.8085 |
| Same-index source ranking | 0.2920 | 0.8750 | 0.8769 |
| Target-gradient ranking | 0.3019 | 0.8594 | 0.8525 |
| Random target sets, 20 draws (mean ± SD) | 0.3345 ± 0.0455 | 0.8730 ± 0.0156 | 0.8810 ± 0.0116 |
| Target exact-ablation oracle† | 0.3704 | 0.8203 | 0.8099 |

### Seed 20261119

**Run:** `transfer_seed20261119_sst2_20261119T113123_685828Z`  
**Analysis code revision:** `523defd2ff186784697aa73a4192d99e3e75917b`

On this seed, causal-aligned ranking coverage was 40/144 (27.8%) and Spearman ρ was **-0.111** (95% interval [-0.225, +0.139]), which includes zero. Same-layer constrained ranking was -0.083 ([-0.338, +0.096]). Representation CKA was -0.128 ([-0.185, -0.014]); same-index was +0.056 ([-0.087, +0.117]); target-gradient was +0.090 ([-0.079, +0.177]). The causal-aligned result is therefore not uniformly negative at the interval level across seeds. Random mappings had mean ρ -0.013 (SD 0.152; empirical interval [-0.306, +0.249]).

The no-ablation group had loss 0.2974, accuracy 0.8906, and positive F1 0.8889. The causal-aligned 14-head group had loss 0.3309, accuracy 0.8516, and F1 0.8455; the same-layer group had loss 0.3225, accuracy 0.8594, and F1 0.8500. Random groups averaged loss 0.3319 ± 0.0303, accuracy 0.8586 ± 0.0154, and F1 0.8538 ± 0.0183. The target exact-ablation oracle had loss 0.2626, accuracy 0.8984, and F1 0.8960, again illustrating that its same-sample selection is optimistic and not a transferable result.

| Seed 20261119 grouped selection | Mean loss | Accuracy | Positive F1 |
|---|---:|---:|---:|
| No ablation | 0.2974 | 0.8906 | 0.8889 |
| Causal-aligned source ranking | 0.3309 | 0.8516 | 0.8455 |
| Same-layer constrained source ranking | 0.3225 | 0.8594 | 0.8500 |
| Representation-CKA source ranking | 0.3872 | 0.8516 | 0.8430 |
| Same-index source ranking | 0.3129 | 0.8594 | 0.8525 |
| Target-gradient ranking | 0.4472 | 0.8125 | 0.7931 |
| Random target sets, 20 draws (mean ± SD) | 0.3319 ± 0.0303 | 0.8586 ± 0.0154 | 0.8538 ± 0.0183 |
| Target exact-ablation oracle† | 0.2626 | 0.8984 | 0.8960 |

## Decision and next experiment

These three Stage 4 runs do not meet the transfer go/no-go condition. Stage 3 pair identities did not recur across seeds. Causal-aligned rank correlations were negative in all three point estimates, but the third seed's interval includes zero, so the association is not consistently distinguishable from zero at this pilot scale. The target-gradient baseline is also inconsistent, and group-ablation outcomes vary. This is not evidence of useful transfer and is not yet a confirmatory negative result. Do not proceed to physical pruning or speed benchmarking.

Repeat the ranking-transfer diagnostic across the remaining Stage 3 seeds (`20261117`, `20261118`) and evaluate the planned 512- and 2,048-example calibration budgets where data and compute permit. Freeze all rankings before evaluating the same labeled validation sample, report paired uncertainty across examples and seeds, and retain the random, representation, same-index, same-layer, target-gradient, and target-oracle comparisons. These 128-example results are pilot evidence, not a final confirmatory test.

## Artifacts

The full inputs, mappings, per-head effects, group-ablation predictions, summary metrics, and dataset manifest are under `results/04_importance_transfer/transfer_seed20261121_sst2_20261119T075016_873545Z/`, `results/04_importance_transfer/transfer_seed20261120_sst2_20261119T080117_537534Z/`, and `results/04_importance_transfer/transfer_seed20261119_sst2_20261119T113123_685828Z/`.
