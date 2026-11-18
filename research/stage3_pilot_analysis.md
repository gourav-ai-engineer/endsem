# Stage 3 SST-2 causal pilot: cross-seed analysis

**Run:** `stage3_bert-gpt2_sst2_20261117_17024513_20261118T063637_767760Z`  
**Experiment code revision:** `81c5349e485eaf51b9e02b25ffbe8eaa5e6690ae`  
**Status:** Two exploratory seeds completed. The Stage 3 multi-seed exit condition is not yet met.

## Design and artifacts

The run used revision-pinned `nyu-mll/glue` SST-2 data, BERT and GPT-2 model/tokenizer revisions from `configs/stage3_pilot.yaml`, and a deterministic stratified split of the official labeled train set: 2,048 fit, 512 checkpoint-selection, and 128 causal-calibration examples. The official labeled validation split (872 examples) was reserved for outer evaluation. The public test split was not used.

Each model was fine-tuned for the configured two epochs. We measured exact per-example ablations for all 144 heads at the attention-head output before the output projection, plus a gate-gradient proxy. Causal signatures used each calibration example's loss change and true-class margin change. The signature-matching permutation shuffled example correspondence within sentiment-label strata.

The run artifacts include split IDs and hashes, per-example and per-head exact effects, gradient proxies, causal CKA, assignments, random-baseline draws, permutation draws, and the final summary. A process error occurred only while recording package-version metadata after these artifacts had been written. The summary was finalized from the saved models and outputs; training history was not preserved. Model selection and outer-validation metrics were recomputed from the saved checkpoints.

## Results

| Measure | BERT | GPT-2 |
|---|---:|---:|
| Checkpoint-selection accuracy (512 examples) | 0.8965 | 0.8125 |
| Outer SST-2 validation accuracy (872 examples) | 0.8842 | 0.8073 |
| Mean exact loss change after one-head ablation | -0.00180 | +0.00177 |
| Mean true-class margin decrease after ablation | +0.00912 | +0.02006 |
| Mean prediction-flip rate across head/example pairs | 0.14% | 0.66% |
| Pearson correlation: first-order gradient proxy vs exact mean loss change across heads | 0.902 | 0.814 |

Positive loss change means ablating a head raised loss; negative means loss fell. Positive margin decrease means ablation lowered the true-class margin. These are averages over this calibration set, not estimates of population-level utility.

| Causal-signature matching method | Mean CKA | Match count |
|---|---:|---:|
| Global partial matching | **0.2153** | 34 |
| Same-layer, fixed cardinality | 0.1690 | 34 |
| Same-index, fixed cardinality | 0.0995 | 34 |
| Random fixed-cardinality assignments, 500 draws | 0.0238 ± 0.0057 SD | 34 |
| Label-stratified example-permutation null, 100 draws | 0.1636 ± 0.0201 SD | 34 |

The observed global score exceeded every permutation draw, giving the minimum possible empirical p-value at 100 draws: 0.0099. The null's 95% empirical interval was [0.1329, 0.1996]. Global matching exceeded the same-layer constraint by 0.0463 CKA, but matched only 34 of 144 possible head pairs.

None of the 34 global causal-signature pairs exactly matched a pair in the Stage 2 pretrained-representation assignment (0 pair overlap). That divergence is a useful observation for the hypothesis, but a single seed and one small calibration split cannot establish that representation matches fail to transfer.

## Interpretation

This first run shows that a causal response signature can be computed for both fine-tuned models and that global matching in this run exceeds the tested random, same-index, same-layer, and label-stratified permutation baselines. The agreement between gradient proxies and exact mean loss changes is descriptive and encouraging for using the proxy in follow-up work.

At the time of the first run, the evidence was exploratory: 2,048 training examples, 128 calibration examples, and 100 permutation replicates, with no cross-seed stability estimate. BERT and GPT-2 also differ in their classification readouts. The pair mapping should not be presented as stable, and no pruning-utility transfer was tested.

## Next Stage 3 step

## Second seed: 20261118

**Run:** `stage3_bert-gpt2_sst2_20261118_77a4208b_20261118T070529_838582Z`

**Code revision:** `78c5964f6df8767193be6add94511b3b5d6d5f33`

This run completed without the metadata-finalization interruption seen in seed 20261117. It used independently seeded fit, selection, and calibration splits (2,048 / 512 / 128 examples) and the same pinned model and dataset revisions. Both checkpoints were selected from the configured two epochs; BERT selected epoch 2 and GPT-2 epoch 1.

| Measure | BERT | GPT-2 |
|---|---:|---:|
| Checkpoint-selection accuracy (512 examples) | 0.9160 | 0.8418 |
| Outer SST-2 validation accuracy (872 examples) | 0.8933 | 0.8406 |
| Mean exact loss change after one-head ablation | +0.00268 | +0.00053 |
| Mean true-class margin decrease after ablation | +0.01923 | +0.01172 |
| Mean prediction-flip rate across head/example pairs | 1.39% | 0.87% |
| Pearson correlation: gradient proxy vs exact mean loss change across heads | 0.864 | 0.814 |

| Causal-signature matching method | Mean CKA | Match count |
|---|---:|---:|
| Global partial matching | **0.2270** | 21 |
| Same-layer, fixed cardinality | 0.1651 | 21 |
| Same-index, fixed cardinality | 0.1014 | 21 |
| Random fixed-cardinality assignments, 500 draws | 0.0208 ± 0.0069 SD | 21 |
| Label-stratified example-permutation null, 100 draws | 0.2007 ± 0.0186 SD | 21 |

The empirical permutation p-value was 0.0693 (7 of 100 null scores at least as large as observed, with the +1 correction). Global matching exceeded the same-layer score by 0.0619 CKA. This is weaker permutation evidence than seed 20261117 (p = 0.0099), with fewer matched pairs (21 versus 34). Across the two seeds, 40 global pairs were shared among 180 distinct pairs (pair-set Jaccard 0.222). These differences point to substantial mapping variability; two seeds are not enough to estimate stable pair frequencies or score uncertainty.

The second seed therefore reproduces the *direction* of the descriptive baseline comparisons, but not the first seed's permutation significance or assignment set. It does not establish task-conditioned causal correspondence. Both runs still use a small calibration set and only 100 permutation replicates; the validation accuracies are outer-evaluation metrics and are not evidence about cross-model pruning transfer.

**Correction to the earlier pair-overlap calculation:** the previous report version selected the wrong columns from the matches CSV when comparing seed 20261117 and seed 20261118. Using the actual `source_head` and `target_head` columns, those runs share zero global pairs (Jaccard 0.000), not 40 pairs (Jaccard 0.222). The raw match artifacts and all per-run CKA and permutation statistics were unaffected.

## Third seed: 20261119

**Run:** `stage3_bert-gpt2_sst2_20261119_ec8f59fe_20261118T094954_357817Z`

| Measure | BERT | GPT-2 |
|---|---:|---:|
| Checkpoint-selection accuracy (512 examples) | 0.8926 | 0.8516 |
| Outer SST-2 validation accuracy (872 examples) | 0.8819 | 0.8349 |
| Mean exact loss change after one-head ablation | -0.00128 | +0.00059 |
| Mean true-class margin decrease after ablation | +0.01651 | +0.01407 |
| Mean prediction-flip rate across head/example pairs | 0.57% | 0.67% |
| Pearson correlation: gradient proxy vs exact mean loss change across heads | 0.948 | 0.545 |

| Causal-signature matching method | Mean CKA | Match count |
|---|---:|---:|
| Global partial matching | **0.2297** | 40 |
| Same-layer, fixed cardinality | 0.1545 | 40 |
| Same-index, fixed cardinality | 0.0770 | 40 |
| Random fixed-cardinality assignments, 500 draws | 0.0257 ± 0.0053 SD | 40 |
| Label-stratified example-permutation null, 100 draws | 0.1445 ± 0.0162 SD | 40 |

The empirical permutation p-value was 0.0099 (minimum attainable with 100 draws). Global matching exceeded the same-layer score by 0.0752 CKA. Across the three runs, pairwise exact global-pair overlap is zero: each pair of seeds has Jaccard 0.000, and there are 95 distinct pairs in total. This is direct evidence that the selected head-to-head assignments are seed-sensitive at this pilot scale, even though the observed global score exceeded the tested baselines in each run.

## Next Stage 3 step

## Fourth seed: 20261120

**Run:** `stage3_bert-gpt2_sst2_20261120_ba78d983_20261118T102517_770322Z`

| Measure | BERT | GPT-2 |
|---|---:|---:|
| Checkpoint-selection accuracy (512 examples) | 0.8867 | 0.8281 |
| Outer SST-2 validation accuracy (872 examples) | 0.8842 | 0.8452 |
| Mean exact loss change after one-head ablation | -0.00375 | +0.00012 |
| Mean true-class margin decrease after ablation | +0.00999 | +0.02331 |
| Mean prediction-flip rate across head/example pairs | 0.12% | 0.76% |
| Pearson correlation: gradient proxy vs exact mean loss change across heads | 0.919 | 0.883 |

| Causal-signature matching method | Mean CKA | Match count |
|---|---:|---:|
| Global partial matching | **0.2217** | 16 |
| Same-layer, fixed cardinality | 0.1790 | 16 |
| Same-index, fixed cardinality | 0.0827 | 16 |
| Random fixed-cardinality assignments, 500 draws | 0.0170 ± 0.0070 SD | 16 |
| Label-stratified example-permutation null, 100 draws | 0.1940 ± 0.0291 SD | 16 |

The empirical permutation p-value was 0.1386. Global matching exceeded the same-layer score by 0.0427 CKA, but only 16 pairs were selected. Across all four seeds, no exact global source-target head pair is shared between any two seeds; there are 111 distinct pairs in total. The changing match count and absent pair recurrence show that the head correspondence itself remains unstable, even though global scores exceed structural and random baselines in each run.

## Next Stage 3 step

Run the final predeclared seed (`20261121`) with an independent train/selection/calibration split. Then report pair-frequency and score uncertainty across all five seeds, alongside seed-wise permutation results. Only after that should we compare source causal rankings with target exact-ablation utility on held-out examples. Do not proceed to physical pruning from these pilot runs alone.
