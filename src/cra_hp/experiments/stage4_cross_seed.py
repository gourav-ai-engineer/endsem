"""Seed-and-example bootstrap for Stage 4 ranking-transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

METHODS = (
    "causal_aligned",
    "causal_same_layer",
    "representation_cka",
    "same_index",
    "target_gradient",
)
GROUP_METHODS = (
    "causal_aligned",
    "causal_same_layer",
    "representation_cka",
    "same_index",
    "target_gradient",
    "random_target",
    "target_exact_oracle",
)


def _load_runs(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(root.glob("transfer_seed*_sst2_*")):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        source_run = Path(config["source_run"])
        if not source_run.is_absolute():
            source_run = root.parent.parent / source_run
        source_seed = summary.get("source_seed")
        if source_seed is None:
            source_summary = json.loads(
                (source_run / "summary.json").read_text(encoding="utf-8")
            )
            source_seed = source_summary["seed"]
        effects = pd.read_csv(run_dir / "target_heldout_per_head_effects.csv")
        groups = pd.read_csv(run_dir / "grouped_ablation_per_example.csv")
        pivot = effects.pivot(
            index="example_id", columns="head_id", values="loss_delta"
        )
        labels = effects.drop_duplicates("example_id").set_index("example_id")["label"]
        example_ids = list(labels.index)
        example_index = {example_id: index for index, example_id in enumerate(example_ids)}
        baseline = groups[
            (groups["method"] == "no_ablation") & (groups["replicate"] == 0)
        ].set_index("example_id").reindex(example_ids)
        group_arrays: dict[str, list[dict[str, np.ndarray]]] = {
            "no_ablation": [{
                "loss": baseline["loss"].to_numpy(),
                "prediction": baseline["prediction"].to_numpy(),
            }]
        }
        for method in GROUP_METHODS:
            if method == "random_target":
                selected_groups = groups[groups["method"] == method]
            else:
                selected_groups = groups[
                    (groups["method"] == method) & (groups["replicate"] == 0)
                ]
            group_arrays[method] = [
                {
                    "loss": replicate.set_index("example_id").reindex(example_ids)["loss"].to_numpy(),
                    "prediction": replicate.set_index("example_id").reindex(example_ids)["prediction"].to_numpy(),
                }
                for _, replicate in selected_groups.groupby("replicate")
            ]
        source_effects = pd.read_csv(source_run / "encoder_head_effects.csv")
        source_scores = source_effects.set_index("head_id")["mean_loss_delta"]
        gradient = pd.read_csv(source_run / "decoder_gradient_proxy.csv")
        gradient_scores = gradient.set_index("head_id")[
            "first_order_loss_increase_if_removed"
        ]
        mappings = {
            method: pd.read_csv(run_dir / f"mapping_{method}.csv")
            for method in METHODS
            if method != "target_gradient"
        }
        runs.append({
            "seed": int(source_seed),
            "run_id": run_dir.name,
            "pivot": pivot,
            "labels": labels,
            "source_scores": source_scores,
            "gradient_scores": gradient_scores,
            "mappings": mappings,
            "example_index": example_index,
            "group_labels": baseline["label"].to_numpy(),
            "group_arrays": group_arrays,
        })
    return sorted(runs, key=lambda item: item["seed"])


def _correlations(run: dict[str, Any], sampled_ids: list[str]) -> dict[str, float]:
    target = run["pivot"].loc[sampled_ids].mean(axis=0)
    values: dict[str, float] = {}
    for method in METHODS:
        if method == "target_gradient":
            source_scores = run["gradient_scores"]
            target_ids = sorted(set(target.index) & set(source_scores.index))
            left = source_scores.loc[target_ids]
        else:
            mapping = run["mappings"][method]
            source_ids = mapping["source_head"].astype(int)
            target_ids = mapping["target_head"].astype(int)
            valid = source_ids.isin(run["source_scores"].index) & target_ids.isin(target.index)
            mapping = mapping.loc[valid]
            target_ids = mapping["target_head"].astype(int).tolist()
            left = pd.Series(
                run["source_scores"].reindex(mapping["source_head"].astype(int)).to_numpy(),
                index=target_ids,
            )
        right = target.reindex(target_ids)
        if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
            values[method] = float("nan")
        else:
            values[method] = float(spearmanr(left, right).statistic)
    return values


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)


def _group_deltas(
    run: dict[str, Any], method: str, sampled_ids: list[str]
) -> dict[str, float]:
    indices = np.asarray([run["example_index"][example_id] for example_id in sampled_ids])
    labels = run["group_labels"][indices]
    baseline = run["group_arrays"]["no_ablation"][0]
    base_predictions = baseline["prediction"][indices]
    base_loss = baseline["loss"][indices]
    replicate_deltas = []
    for candidate in run["group_arrays"][method]:
        predictions = candidate["prediction"][indices]
        candidate_loss = candidate["loss"][indices]
        replicate_deltas.append({
            "mean_loss_delta": float(np.mean(candidate_loss - base_loss)),
            "accuracy_delta": float(
                np.mean(predictions == labels) - np.mean(base_predictions == labels)
            ),
            "positive_f1_delta": _f1(labels, predictions) - _f1(labels, base_predictions),
        })
    return {
        metric: float(np.mean([row[metric] for row in replicate_deltas]))
        for metric in ("mean_loss_delta", "accuracy_delta", "positive_f1_delta")
    }


def analyze(root: Path, *, replicates: int, seed: int) -> Path:
    runs = _load_runs(root)
    if len(runs) < 2:
        raise ValueError(f"Need at least two completed Stage 4 runs; found {len(runs)}")
    common_ids = sorted(set.intersection(*[set(run["labels"].index) for run in runs]))
    labels = runs[0]["labels"].reindex(common_ids)
    label_groups = {
        label: [example_id for example_id in common_ids if labels.loc[example_id] == label]
        for label in sorted(labels.unique())
    }
    rng = np.random.default_rng(seed)
    point_rows = []
    point_group_rows = []
    for run in runs:
        for method, rho in _correlations(run, common_ids).items():
            point_rows.append({"source_seed": run["seed"], "run_id": run["run_id"], "method": method, "spearman_rho": rho})
        for method in GROUP_METHODS:
            point_group_rows.append({
                "source_seed": run["seed"],
                "run_id": run["run_id"],
                "method": method,
                **_group_deltas(run, method, common_ids),
            })

    boot_values: dict[str, list[float]] = {method: [] for method in METHODS}
    boot_group_values: dict[str, dict[str, list[float]]] = {
        method: {
            metric: []
            for metric in ("mean_loss_delta", "accuracy_delta", "positive_f1_delta")
        }
        for method in GROUP_METHODS
    }
    for _ in range(replicates):
        sampled_runs = rng.choice(len(runs), size=len(runs), replace=True)
        sampled_ids = [
            example_id
            for group in label_groups.values()
            for example_id in rng.choice(group, size=len(group), replace=True).tolist()
        ]
        per_method: dict[str, list[float]] = {method: [] for method in METHODS}
        per_group: dict[str, dict[str, list[float]]] = {
            method: {
                metric: []
                for metric in ("mean_loss_delta", "accuracy_delta", "positive_f1_delta")
            }
            for method in GROUP_METHODS
        }
        for run_index in sampled_runs:
            selected_run = runs[int(run_index)]
            for method, rho in _correlations(selected_run, sampled_ids).items():
                if np.isfinite(rho):
                    per_method[method].append(rho)
            for method in GROUP_METHODS:
                deltas = _group_deltas(selected_run, method, sampled_ids)
                for metric, value in deltas.items():
                    per_group[method][metric].append(value)
        for method, values in per_method.items():
            if values:
                boot_values[method].append(float(np.mean(values)))
        for method, metrics in per_group.items():
            for metric, values in metrics.items():
                if values:
                    boot_group_values[method][metric].append(float(np.mean(values)))

    out_dir = root / "cross_seed_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(out_dir / "per_seed_rank_correlations.csv", index=False)
    summaries = []
    points = pd.DataFrame(point_rows)
    for method in METHODS:
        seed_values = points.loc[points["method"] == method, "spearman_rho"].dropna()
        boot = np.asarray(boot_values[method])
        summaries.append({
            "method": method,
            "completed_seeds": len(seed_values),
            "mean_seed_spearman_rho": float(seed_values.mean()),
            "seed_sd": float(seed_values.std(ddof=1)),
            "negative_seed_point_estimates": int((seed_values < 0).sum()),
            "hierarchical_bootstrap_95_low": float(np.quantile(boot, 0.025)),
            "hierarchical_bootstrap_95_high": float(np.quantile(boot, 0.975)),
            "bootstrap_replicates": len(boot),
            "bootstrap_seed": seed,
            "bootstrap_unit": "resample model seeds and stratified held-out examples; macro-average per-seed Spearman rho",
        })
    pd.DataFrame(summaries).to_csv(out_dir / "cross_seed_rank_summary.csv", index=False)
    group_points = pd.DataFrame(point_group_rows)
    group_summaries = []
    for method in GROUP_METHODS:
        selected = group_points[group_points["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "completed_seeds": len(selected),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        }
        for metric in ("mean_loss_delta", "accuracy_delta", "positive_f1_delta"):
            seed_values = selected[metric].dropna()
            boot = np.asarray(boot_group_values[method][metric])
            row[f"mean_{metric}"] = float(seed_values.mean())
            row[f"{metric}_hierarchical_95_low"] = float(np.quantile(boot, 0.025))
            row[f"{metric}_hierarchical_95_high"] = float(np.quantile(boot, 0.975))
        group_summaries.append(row)
    group_points.to_csv(out_dir / "per_seed_group_ablation_deltas.csv", index=False)
    pd.DataFrame(group_summaries).to_csv(out_dir / "cross_seed_group_summary.csv", index=False)
    metadata = {
        "completed_run_ids": [run["run_id"] for run in runs],
        "heldout_examples_per_run": len(common_ids),
        "label_counts": {str(label): len(group) for label, group in label_groups.items()},
        "bootstrap_replicates_requested": replicates,
        "bootstrap_seed": seed,
        "interval_interpretation": "Exploratory hierarchical percentile interval for the macro-average of per-seed rank correlations; resamples both model seeds and held-out examples.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/04_importance_transfer"))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    output = analyze(args.root, replicates=args.replicates, seed=args.seed)
    print(f"Stage 4 cross-seed analysis written to: {output}")


if __name__ == "__main__":
    main()
