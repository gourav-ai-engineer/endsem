# Stage 2 Status — Representation-only pilot

**Status:** First feasibility run completed; statistical correction implemented; corrected rerun pending.

## First run

- Run: `repr_20261117T152607_402097Z`
- Git revision: `2287082db1229f0388ec348781a8c41a61463ea1`
- Models: `bert-base-uncased` and `gpt2`, 144 attention heads each
- Inputs: 256 WikiText-2 raw rows, sampled from 12 API pages
- Device: CPU
- Raw representation arrays: 144 × 256 × 64 per model

The first run produced a global partial-Hungarian mean CKA of about 0.6851, compared with 0.6482 for same-layer, 0.5578 for same-index, and 0.5368 for one random assignment. These values are descriptive pilot observations only.

## Problems found

1. The permutation null used thresholded partial matching. When all shuffled matches fell below the threshold, the code converted the empty assignment into a mean score of zero. The resulting null distribution was degenerate, so its empirical p-value must not be interpreted.
2. The random baseline was one random assignment, not a distribution. It was too noisy to support a fair baseline comparison.
3. The bootstrap resampled individual rows although rows were drawn from a limited number of WikiText API pages. Within-page dependence was ignored.
4. The first sample covered only 12 pages (1,200 rows fetched) from a 36,718-row split. It is a page-clustered feasibility sample, not a corpus-wide representative sample.

## Corrections implemented on `fix/representation-statistics`

- Add exact fixed-cardinality maximum-weight Hungarian matching.
- Permutation test fixes k to the observed partial-match count and recomputes the maximum-weight k-matching for each shuffled pairing. The observed and null statistics now use the same statistic and cardinality.
- Compare same-index and same-layer constrained assignments at the same fixed match count as the global assignment.
- Estimate random assignment performance over 500 independent fixed-cardinality random assignments.
- Resample WikiText API pages as clusters in the bootstrap.
- Increase pilot bootstrap and permutation replicates to 100.
- Add tests for exact-cardinality matching and cluster bootstrap.
- Fix Ruff findings that caused the previous `main` CI workflow to fail before tests ran.

## What this does and does not establish

The corrected analysis can test whether paired-example representation alignment exceeds a matched-cardinality permutation null and whether candidate head pairs are stable under page-cluster resampling. It still does **not** establish causal correspondence, transferable pruning utility, or model efficiency gains.

## Next action

After the pull request CI passes, check out the correction branch, pull the changes, and rerun:

```powershell
git fetch origin
git checkout fix/representation-statistics
git pull
python -m ruff check src tests
python -m pytest -q
python -m cra_hp.experiments.representation_pilot --config configs/feasibility.yaml
```

Inspect the new `summary.json`, `permutation_null.csv`, `random_baseline.csv`, and `bootstrap_stability.csv`. Do not proceed to causal claims unless the corrected results support the predeclared feasibility criteria.