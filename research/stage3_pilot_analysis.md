# Stage 3 SST-2 causal pilot: first-run analysis

**Run:** `stage3_bert-gpt2_sst2_20261117_17024513_20261118T063637_767760Z`  
**Experiment code revision:** `81c5349e485eaf51b9e02b25ffbe8eaa5e6690ae`  
**Status:** One exploratory seed completed. The Stage 3 multi-seed exit condition is not yet met.

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

The evidence remains exploratory. The models were trained on 2,048 examples for one seed; the calibration set had 128 examples; the permutation test had only 100 replicates; and causal-match bootstrap or cross-seed stability has not been measured. BERT and GPT-2 also differ in their classification readouts. The pair mapping should not be presented as stable, and no pruning-utility transfer was tested.

## Next Stage 3 step

Repeat the same pilot under the remaining predeclared seeds (`20261118`–`20261121`) with independent train/selection/calibration splits, then measure pair-frequency and score uncertainty across seeds. Only after that should we compare source causal rankings with target exact-ablation utility on held-out examples. Do not proceed to physical pruning from this first run alone.
