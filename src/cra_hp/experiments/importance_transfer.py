"""Evaluate calibration-derived head rankings against held-out exact utility."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cra_hp.experiments.causal_pilot import (
    _batches,
    _encode_batch,
    _exact_ablation,
    _install_gates,
)
from cra_hp.task_data import load_sst2_pilot


def _correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right, method=method))


def _top_overlap(left: pd.Series, right: pd.Series, count: int) -> float:
    if count <= 0:
        return float("nan")
    left_top = set(left.nlargest(count).index)
    right_top = set(right.nlargest(count).index)
    return len(left_top & right_top) / count


def _sample_heldout(examples: list[Any], size: int, seed: int) -> list[Any]:
    if size >= len(examples):
        return examples
    labels = np.asarray([example.label for example in examples])
    selected, _ = train_test_split(
        np.arange(len(examples)),
        train_size=size,
        random_state=seed,
        stratify=labels,
    )
    sampled = [examples[int(index)] for index in sorted(selected.tolist())]
    if len(sampled) != size or len({item.label for item in sampled}) != len(np.unique(labels)):
        raise RuntimeError("Held-out sample does not satisfy the declared stratification")
    return sampled


def _bootstrap_correlations(
    per_example: pd.DataFrame,
    scores: dict[int, float],
    target_heads: list[int],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    frame = per_example[per_example["head_id"].isin(target_heads)]
    pivot = frame.pivot(index="example_id", columns="head_id", values="loss_delta")
    labels = frame.drop_duplicates("example_id").set_index("example_id")["label"]
    head_ids = [head for head in target_heads if head in pivot.columns and head in scores]
    if len(head_ids) < 3:
        return float("nan"), float("nan")
    labels = labels.loc[pivot.index]
    groups = [np.flatnonzero(labels.to_numpy() == label) for label in sorted(labels.unique())]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    source = pd.Series({head: scores[head] for head in head_ids})
    for _ in range(replicates):
        sample = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        target = pivot.iloc[sample][head_ids].mean(axis=0)
        value = _correlation(source, target, "spearman")
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _group_metrics(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[Any],
    architecture: str,
    removed_heads: list[int],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    method: str,
    replicate: int = 0,
) -> list[dict[str, Any]]:
    total_heads = int(model.config.n_head) * int(model.config.n_layer)
    gates = torch.ones(total_heads, device=device)
    if removed_heads:
        gates[removed_heads] = 0.0
    handles = _install_gates(model, architecture, gates)
    rows: list[dict[str, Any]] = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in _batches(examples, batch_size):
                inputs, labels = _encode_batch(tokenizer, batch, max_length, device)
                logits = model(**inputs).logits
                losses = F.cross_entropy(logits, labels, reduction="none")
                predictions = logits.argmax(dim=-1)
                for index, example in enumerate(batch):
                    rows.append({
                        "method": method,
                        "replicate": replicate,
                        "example_id": example.example_id,
                        "label": example.label,
                        "loss": float(losses[index]),
                        "prediction": int(predictions[index]),
                    })
    finally:
        for handle in handles:
            handle.remove()
    return rows


def _summarize_grouped(frame: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (method, replicate), group in frame.groupby(["method", "replicate"]):
        labels = group["label"].to_numpy()
        predictions = group["prediction"].to_numpy()
        tp = int(((labels == 1) & (predictions == 1)).sum())
        fp = int(((labels == 0) & (predictions == 1)).sum())
        fn = int(((labels == 1) & (predictions == 0)).sum())
        summaries.append({
            "method": method,
            "replicate": replicate,
            "mean_loss": float(group["loss"].mean()),
            "accuracy": float((labels == predictions).mean()),
            "positive_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "examples": len(group),
        })
    return pd.DataFrame(summaries)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_run = Path(config["source_run"])
    stage3_config = yaml.safe_load(
        (source_run / "config.yaml").read_text(encoding="utf-8")
    )
    data = load_sst2_pilot(stage3_config["dataset"])
    eval_cfg = config["evaluation"]
    heldout = _sample_heldout(
        data.heldout, int(eval_cfg["sample_size"]), int(eval_cfg["sample_seed"])
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(config["outputs"]["root"]) / f"transfer_seed20261121_sst2_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "dataset_revision": stage3_config["dataset"]["revision"],
        "source_run": str(source_run),
        "heldout_source_split": "official_labeled_validation",
        "heldout_sample_size": len(heldout),
        "heldout_sample_seed": int(eval_cfg["sample_seed"]),
        "heldout_example_ids": [example.example_id for example in heldout],
        "heldout_example_ids_sha256": hashlib.sha256(
            "\n".join(example.example_id for example in heldout).encode()
        ).hexdigest(),
        "label_counts": pd.Series([item.label for item in heldout]).value_counts().sort_index().to_dict(),
        "ranking_features": "mean calibration loss_delta; larger values indicate greater single-head removal cost",
        "target_utility": "mean exact single-head loss_delta on held-out validation sample",
        "heldout_usage": "single diagnostic evaluation; no ranking or threshold selection",
    }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    source_effects = pd.read_csv(source_run / "encoder_head_effects.csv")
    source_scores = source_effects.set_index("head_id")["mean_loss_delta"].to_dict()
    target_gradient = pd.read_csv(source_run / "decoder_gradient_proxy.csv")
    target_gradient_scores = target_gradient.set_index("head_id")[
        "first_order_loss_increase_if_removed"
    ].to_dict()
    stage3_matches = pd.read_csv(source_run / "causal_matches.csv")
    stage2_matches = pd.read_csv(config["representation_matches"])
    stage2_matches = stage2_matches[
        stage2_matches["method"] == config["representation_method"]
    ]
    mappings: dict[str, pd.DataFrame] = {
        "causal_aligned": stage3_matches[stage3_matches["method"] == "global_partial"],
        "causal_same_layer": stage3_matches[
            stage3_matches["method"] == "same_layer_fixed_k"
        ],
        "representation_cka": stage2_matches,
        "same_index": pd.DataFrame({"source_head": range(144), "target_head": range(144)}),
    }
    for method, mapping in mappings.items():
        mapping[["source_head", "target_head"]].to_csv(
            run_dir / f"mapping_{method}.csv", index=False
        )

    target_checkpoint = source_run / "checkpoints" / "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(target_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(target_checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_cfg = stage3_config["models"]
    exact_frame, exact_summary = _exact_ablation(
        model,
        tokenizer,
        heldout,
        "gpt2",
        batch_size=int(eval_cfg["batch_size"]),
        heads_per_forward=int(eval_cfg["exact_ablation_heads_per_forward"]),
        max_length=int(model_cfg["max_length"]),
        device=device,
    )
    exact_frame.to_csv(run_dir / "target_heldout_per_head_effects.csv", index=False)
    exact_summary.to_csv(run_dir / "target_heldout_head_utility.csv", index=False)
    target_utility = exact_summary.set_index("head_id")["mean_loss_delta"]

    ranking_rows: list[dict[str, Any]] = []
    mapped_scores: dict[str, dict[int, float]] = {}
    for method, mapping in mappings.items():
        pairs = mapping[["source_head", "target_head"]].drop_duplicates()
        scores = {
            int(row.target_head): float(source_scores[int(row.source_head)])
            for row in pairs.itertuples(index=False)
        }
        mapped_scores[method] = scores
        observed = sorted(set(scores) & set(target_utility.index))
        source_series = pd.Series({head: scores[head] for head in observed})
        target_series = target_utility.loc[observed]
        top_count = max(1, round(0.10 * len(observed)))
        correlation = _correlation(source_series, target_series, "spearman")
        low_overlap = _top_overlap(-source_series, -target_series, top_count)
        ci_low, ci_high = _bootstrap_correlations(
            exact_frame,
            scores,
            observed,
            replicates=int(eval_cfg["bootstrap_replicates"]),
            seed=int(eval_cfg["bootstrap_seed"]),
        )
        ranking_rows.append({
            "method": method,
            "coverage_heads": len(observed),
            "coverage_fraction": len(observed) / len(target_utility),
            "spearman_rho": correlation,
            "spearman_bootstrap_95_low": ci_low,
            "spearman_bootstrap_95_high": ci_high,
            "kendall_tau": _correlation(source_series, target_series, "kendall"),
            "least_important_top_10pct_overlap": low_overlap,
            "overlap_k_within_coverage": top_count,
        })

    gradient_ids = sorted(target_gradient_scores)
    gradient_series = pd.Series({head: target_gradient_scores[head] for head in gradient_ids})
    target_series = target_utility.loc[gradient_ids]
    ci_low, ci_high = _bootstrap_correlations(
        exact_frame,
        target_gradient_scores,
        gradient_ids,
        replicates=int(eval_cfg["bootstrap_replicates"]),
        seed=int(eval_cfg["bootstrap_seed"]),
    )
    ranking_rows.append({
        "method": "target_gradient",
        "coverage_heads": len(gradient_ids),
        "coverage_fraction": len(gradient_ids) / len(target_utility),
        "spearman_rho": _correlation(gradient_series, target_series, "spearman"),
        "spearman_bootstrap_95_low": ci_low,
        "spearman_bootstrap_95_high": ci_high,
        "kendall_tau": _correlation(gradient_series, target_series, "kendall"),
        "least_important_top_10pct_overlap": _top_overlap(
            -gradient_series, -target_series, max(1, round(0.10 * len(gradient_ids)))
        ),
        "overlap_k_within_coverage": max(1, round(0.10 * len(gradient_ids))),
    })

    rng = np.random.default_rng(int(eval_cfg["random_seed"]))
    k = len(mappings["causal_aligned"])
    random_rows: list[dict[str, Any]] = []
    for replicate in range(int(eval_cfg["random_mapping_replicates"])):
        source_ids = rng.choice(144, size=k, replace=False)
        target_ids = rng.choice(144, size=k, replace=False)
        random_scores = pd.Series(
            {int(target): float(source_scores[int(source)])
             for source, target in zip(source_ids, target_ids, strict=True)}
        )
        utility = target_utility.loc[random_scores.index]
        random_rows.append({
            "replicate": replicate,
            "coverage_heads": k,
            "spearman_rho": _correlation(random_scores, utility, "spearman"),
            "kendall_tau": _correlation(random_scores, utility, "kendall"),
            "least_important_top_10pct_overlap": _top_overlap(
                -random_scores, -utility, max(1, round(0.10 * k))
            ),
        })
    random_mapping = pd.DataFrame(random_rows)
    random_mapping.to_csv(run_dir / "random_mapping_metrics.csv", index=False)
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(run_dir / "ranking_metrics.csv", index=False)

    prune_count = max(1, round(0.10 * len(target_utility)))
    group_rows: list[dict[str, Any]] = []
    base_rows = _group_metrics(
        model, tokenizer, heldout, "gpt2", [], batch_size=int(eval_cfg["batch_size"]),
        max_length=int(model_cfg["max_length"]), device=device, method="no_ablation"
    )
    group_rows.extend(base_rows)
    selected_sets: dict[str, list[int]] = {}
    for method, scores in mapped_scores.items():
        pruneable = pd.Series(scores).dropna().nsmallest(min(prune_count, len(scores)))
        selected_sets[method] = [int(head) for head in pruneable.index]
    gradient_pruneable = gradient_series.nsmallest(prune_count)
    selected_sets["target_gradient"] = [int(head) for head in gradient_pruneable.index]
    oracle_pruneable = target_utility.nsmallest(prune_count)
    selected_sets["target_exact_oracle"] = [int(head) for head in oracle_pruneable.index]
    for method, heads in selected_sets.items():
        group_rows.extend(_group_metrics(
            model, tokenizer, heldout, "gpt2", heads,
            batch_size=int(eval_cfg["batch_size"]), max_length=int(model_cfg["max_length"]),
            device=device, method=method,
        ))
    random_target_sets = [
        sorted(rng.choice(144, size=prune_count, replace=False).tolist())
        for _ in range(int(eval_cfg["random_group_replicates"]))
    ]
    for replicate, heads in enumerate(random_target_sets):
        group_rows.extend(_group_metrics(
            model, tokenizer, heldout, "gpt2", heads,
            batch_size=int(eval_cfg["batch_size"]), max_length=int(model_cfg["max_length"]),
            device=device, method="random_target", replicate=replicate,
        ))
    grouped = pd.DataFrame(group_rows)
    grouped.to_csv(run_dir / "grouped_ablation_per_example.csv", index=False)
    group_summary = _summarize_grouped(grouped)
    group_summary.to_csv(run_dir / "grouped_ablation_summary.csv", index=False)
    random_summary = group_summary[group_summary["method"] == "random_target"]

    random_aggregate = {
        column: {
            "mean": float(random_summary[column].mean()),
            "std": float(random_summary[column].std(ddof=1)),
        }
        for column in ("mean_loss", "accuracy", "positive_f1")
    }
    summary = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_stage3_run": str(source_run),
        "target_model": "gpt2",
        "source_model": "bert-base-uncased",
        "device": str(device),
        "heldout_examples": len(heldout),
        "prune_budget_heads": prune_count,
        "ranking_metrics": ranking.to_dict(orient="records"),
        "random_mapping_metrics": {
            column: {
                "mean": float(random_mapping[column].mean()),
                "std": float(random_mapping[column].std(ddof=1)),
                "q025": float(random_mapping[column].quantile(0.025)),
                "q975": float(random_mapping[column].quantile(0.975)),
            }
            for column in ("spearman_rho", "kendall_tau", "least_important_top_10pct_overlap")
        },
        "random_group_ablation": random_aggregate,
        "interpretation_warning": (
            "Single-seed, 128-example diagnostic. Stage 3 exact pair recurrence failed. "
            "The target exact-ablation oracle is computed on this evaluation sample and "
            "is a non-deployable upper reference. No physical pruning or speed claims."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = run(args.config)
    print(f"Stage 4 importance-transfer artifacts written to: {output}")


if __name__ == "__main__":
    main()
