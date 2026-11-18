# Corrected Stage 2 representation pilot: analysis

**Run:** `repr_20261118T025519_863016Z`  
**Run commit:** `1090e65b6bf146265396fce67e6c2688f9a7fb41`  
**Decision:** Proceed to Stage 3 as a feasibility follow-up. Do not treat this pilot as evidence of causal correspondence or pruning-utility transfer.

## Artifacts inspected

The analysis uses the run's `summary.json`, `permutation_null.csv`, `random_baseline.csv`, `bootstrap_stability.csv`, the three fixed/partial match tables, `all_matches.csv`, `samples.json`, and `cka_heatmap.png`. The saved CKA matrix and representation tensors were also checked for their recorded dimensions/metadata. The run contains 256 paired examples, 144 heads per model, 64-dimensional pooled head vectors, and examples drawn from 12 WikiText API pages.

## Results

| Statistic | Result |
|---|---:|
| Global partial Hungarian, 143 pairs | 0.6851 mean CKA |
| Same-layer assignment, fixed at 143 pairs | 0.6514 |
| Same-index assignment, fixed at 143 pairs | 0.5549 |
| Random fixed-cardinality assignment, 500 draws | 0.5298 ± 0.0085 SD; 95% empirical interval [0.5135, 0.5461] |
| Paired-example permutation null, 100 draws | 0.0555 ± 0.0025 SD; 95% empirical interval [0.0514, 0.0613] |
| Permutation empirical p-value | 0.0099 (1/101; resolution limited by 100 draws) |
| Cluster-bootstrap pair-set Jaccard, 100 draws | mean 0.159; median 0.149 |

Global assignment scores exceed the random and same-index baselines, and exceed the same-layer constrained assignment by 0.0337 CKA. This indicates cross-example representation correspondence in this sample and some improvement from allowing cross-layer assignments. The heatmap has visible structure, but it is diffuse and includes strong row/column bands; this supports structured similarity, not a one-to-one semantic identity for every head. A same-layer assignment remains competitive, so the benefit of unrestricted global matching is modest.

Bootstrap evidence is mixed. Of 1,681 distinct pairs seen across replicates, 39 appeared in at least half of the 100 cluster-bootstrap draws, 8 in at least 75%, and one in at least 90%. Yet the average pair-set Jaccard is only 0.159. A small subset is repeatable; the full 143-pair mapping is not. The bootstrap resamples only 12 source-page clusters, so these uncertainty summaries are preliminary.

## Interpretation and limits

The corrected permutation statistic fixes the observed cardinality at 143 and re-optimizes the maximum-weight assignment after independently shuffling paired-example correspondence. The random baseline instead samples arbitrary one-to-one head pairs from the unshuffled CKA matrix. They test different nulls and should not be conflated. The permutation result is strong within this pilot, but 100 permutations give coarse p-value resolution and do not address dataset or model-selection uncertainty.

This remains a representation-only result. Extraction captures masked-mean pooled head context before the output projection, not the head's full causal contribution after downstream mixing. BERT's bidirectional and GPT-2's causal contexts differ, tokenization is model-specific, and the sampled examples are WikiText rather than task examples. Model revisions are recorded as resolved for BERT and GPT-2 in the run metadata, but tokenizer revision is null and the WikiText dataset revision is the moving `main` reference. These provenance gaps should be fixed in the next reproducible run.

## Stage decision

Proceed to Stage 3 with an explicitly exploratory, task-conditioned SST-2 experiment. This gate is met because the pipeline completed, the observed assignment exceeds both random and paired-example permutation nulls, and a nonempty subset of pairs recurs in cluster bootstrap. Same-layer matching is still close and the global map is unstable overall, so Stage 3 must compare the proposed causal correspondence against same-layer, same-index, representation-CKA, and target-specific baselines. Do not use this pilot's match map as a fixed truth or claim causal/utility transfer from it.

## Next Stage 3 work

1. Pin the SST-2 dataset revision and split hashes, and pin tokenizer revisions alongside model commits.
2. Define the common fine-tuning budget and validation-only checkpoint selection for BERT and GPT-2; keep the test split untouched.
3. Implement head gates at the pre-output-projection location and compare exact per-head ablations with gradient proxies on calibration examples.
4. Build task-conditioned response signatures on calibration data, then measure signature and ranking stability across seeds and held-out validation examples.
5. Compare causal matching against representation-only CKA, same-layer, same-index, random, and target-specific exact-ablation reference rankings before any physical pruning work.

The experiment plan in `research/research_protocol.md` remains the governing protocol; Stage 3 implementation and results are not yet present in this repository.
