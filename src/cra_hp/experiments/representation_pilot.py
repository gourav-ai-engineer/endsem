"""Run the BERT/GPT-2 representation-only alignment pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from cra_hp.bootstrap import bootstrap_matching
from cra_hp.data import load_wikitext_2_raw
from cra_hp.extraction import extract_head_representations
from cra_hp.matching import (
    fixed_cardinality_hungarian,
    partial_hungarian,
    random_matching,
    same_index_matching,
    same_layer_matching,
)
from cra_hp.similarity import cka_similarity_matrix


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _save_matches(path: Path, matches: list[Any], source_meta: dict[str, Any],
                  target_meta: dict[str, Any], method: str) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        rows.append({
            "method": method,
            "source_head": match.source_head,
            "source_layer": match.source_head // source_meta["num_heads_per_layer"],
            "source_head_in_layer": match.source_head % source_meta["num_heads_per_layer"],
            "target_head": match.target_head,
            "target_layer": match.target_head // target_meta["num_heads_per_layer"],
            "target_head_in_layer": match.target_head % target_meta["num_heads_per_layer"],
            "score": match.score,
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repro = config["reproducibility"]
    seed = int(repro["seed"])
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        import transformers

        torch.manual_seed(seed)
        torch.set_num_threads(int(repro.get("num_threads", 4)))
        deterministic = bool(repro.get("deterministic_algorithms", False))
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
    except ImportError as exc:
        raise RuntimeError("Install optional model dependencies first") from exc

    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("repr_%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    texts, sample_ids, data_metadata = load_wikitext_2_raw(config["data"])
    sample_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    ]
    (run_dir / "samples.json").write_text(
        json.dumps(
            {"dataset": data_metadata, "sample_ids": sample_ids,
             "sample_sha256": sample_hashes},
            indent=2,
        ),
        encoding="utf-8",
    )

    model_cfg = config["models"]
    extraction_cfg = config["extraction"]
    device = model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Models are intentionally loaded and released one at a time.
    encoder = extract_head_representations(
        model_cfg["encoder"]["id"],
        texts,
        max_length=int(config["data"]["max_length"]),
        batch_size=int(extraction_cfg["batch_size"]),
        device=device,
        revision=model_cfg["encoder"].get("revision", "main"),
    )
    encoder_metadata = encoder.metadata
    np.savez_compressed(
        run_dir / "encoder_representations.npz",
        representations=encoder.representations,
    )
    del encoder.representations
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decoder = extract_head_representations(
        model_cfg["decoder"]["id"],
        texts,
        max_length=int(config["data"]["max_length"]),
        batch_size=int(extraction_cfg["batch_size"]),
        device=device,
        revision=model_cfg["decoder"].get("revision", "main"),
    )
    decoder_metadata = decoder.metadata
    np.savez_compressed(
        run_dir / "decoder_representations.npz",
        representations=decoder.representations,
    )

    with np.load(run_dir / "encoder_representations.npz") as data:
        source = data["representations"]
    target = decoder.representations

    if source.shape[1] != target.shape[1]:
        raise RuntimeError("Model representations do not share the same sample dimension")
    expected_heads = int(extraction_cfg["expected_heads"])
    if source.shape[0] != expected_heads or target.shape[0] != expected_heads:
        raise RuntimeError(
            f"Expected {expected_heads} heads per model; got {source.shape[0]} and {target.shape[0]}"
        )

    similarity = cka_similarity_matrix(
        source, target, epsilon=float(config["similarity"]["epsilon"])
    )
    np.save(run_dir / "cka_similarity.npy", similarity)
    pd.DataFrame(similarity).to_csv(run_dir / "cka_similarity.csv", index_label="source_head")

    matching_cfg = config["matching"]
    threshold = float(matching_cfg["unmatched_threshold"])
    global_matches = partial_hungarian(similarity, threshold=threshold)
    encoder_meta = {
        "num_heads_per_layer": int(encoder_metadata["num_heads_per_layer"]),
    }
    decoder_meta = {
        "num_heads_per_layer": int(decoder_metadata["num_heads_per_layer"]),
    }

    all_match_rows: list[dict[str, Any]] = []
    all_match_rows.extend(_save_matches(
        run_dir / "matches_global_partial.csv", global_matches,
        encoder_metadata, decoder_metadata, "global_partial_hungarian"
    ))
    same_index = same_index_matching(similarity, threshold=threshold)
    all_match_rows.extend(_save_matches(
        run_dir / "matches_same_index.csv", same_index,
        encoder_metadata, decoder_metadata, "same_index"
    ))
    same_layer = same_layer_matching(
        similarity,
        encoder_meta["num_heads_per_layer"],
        decoder_meta["num_heads_per_layer"],
        threshold=threshold,
    )
    all_match_rows.extend(_save_matches(
        run_dir / "matches_same_layer.csv", same_layer,
        encoder_metadata, decoder_metadata, "same_layer_hungarian"
    ))

    matched_count = len(global_matches)
    random_baseline = random_matching(similarity, matched_count, seed=seed)
    all_match_rows.extend(_save_matches(
        run_dir / "matches_random.csv", random_baseline,
        encoder_metadata, decoder_metadata, "random_fixed_cardinality_example"
    ))
    pd.DataFrame(all_match_rows).to_csv(run_dir / "all_matches.csv", index=False)

    # Compare the observed maximum-weight k-matching with maximum-weight
    # k-matchings after breaking paired-example correspondence. The cardinality
    # k is fixed from the observed partial assignment in every replicate.
    bootstrap_cfg = config["bootstrap"]
    permutation_replicates = int(bootstrap_cfg.get("permutation_replicates", 100))
    permutation_rng = np.random.default_rng(seed + 1)
    permutation_rows = []
    if matched_count:
        observed_fixed = fixed_cardinality_hungarian(similarity, matched_count)
        observed_fixed_mean = float(np.mean([item.score for item in observed_fixed]))
        for replicate in range(permutation_replicates):
            shuffled_indices = permutation_rng.permutation(source.shape[1])
            null_scores = cka_similarity_matrix(source, target[:, shuffled_indices, :])
            null_matches = fixed_cardinality_hungarian(null_scores, matched_count)
            null_mean = float(np.mean([item.score for item in null_matches]))
            permutation_rows.append({
                "replicate": replicate,
                "matched_count": matched_count,
                "mean_max_weight_score": null_mean,
            })
        null_values = np.asarray(
            [row["mean_max_weight_score"] for row in permutation_rows],
            dtype=np.float64,
        )
        permutation_summary = {
            "statistic": "mean score of maximum-weight fixed-cardinality assignment",
            "matched_count": matched_count,
            "replicates": permutation_replicates,
            "observed_statistic": observed_fixed_mean,
            "null_mean": float(np.mean(null_values)),
            "null_std": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
            "null_95_percentile_interval": [
                float(np.quantile(null_values, 0.025)),
                float(np.quantile(null_values, 0.975)),
            ],
            "empirical_p_value": float(
                (1 + np.sum(null_values >= observed_fixed_mean))
                / (1 + len(null_values))
            ),
        }
    else:
        permutation_summary = {
            "statistic": "not_estimable_no_observed_matches",
            "matched_count": 0,
            "replicates": 0,
            "observed_statistic": None,
            "null_mean": None,
            "null_std": None,
            "null_95_percentile_interval": None,
            "empirical_p_value": None,
        }
    pd.DataFrame(permutation_rows).to_csv(run_dir / "permutation_null.csv", index=False)

    # Random assignment baseline: estimate a distribution, not a single lucky
    # or unlucky random draw.
    random_replicates = int(bootstrap_cfg.get("random_replicates", 500))
    random_rng = np.random.default_rng(seed + 2)
    random_rows = []
    for replicate in range(random_replicates):
        random_matches = random_matching(
            similarity, matched_count, seed=int(random_rng.integers(0, 2**32 - 1))
        )
        random_mean = (
            float(np.mean([item.score for item in random_matches]))
            if random_matches else None
        )
        random_rows.append({
            "replicate": replicate,
            "matched_count": len(random_matches),
            "mean_score": random_mean,
        })
    pd.DataFrame(random_rows).to_csv(run_dir / "random_baseline.csv", index=False)
    random_values = np.asarray(
        [row["mean_score"] for row in random_rows if row["mean_score"] is not None],
        dtype=np.float64,
    )
    random_summary = {
        "replicates": random_replicates,
        "matched_count": matched_count,
        "mean": float(np.mean(random_values)) if len(random_values) else None,
        "std": float(np.std(random_values, ddof=1)) if len(random_values) > 1 else None,
        "percentile_95_interval": [
            float(np.quantile(random_values, 0.025)),
            float(np.quantile(random_values, 0.975)),
        ] if len(random_values) else None,
    }

    # WikiText rows are sampled from API pages. Resample whole pages, not
    # individual rows, to respect within-page dependence.
    page_size = int(data_metadata["page_size"])
    cluster_ids = np.asarray(sample_ids, dtype=np.int64) // page_size
    bootstrap_frame, bootstrap_summary = bootstrap_matching(
        source,
        target,
        replicates=int(bootstrap_cfg["pilot_replicates"]),
        threshold=threshold,
        seed=seed,
        cluster_ids=cluster_ids,
    )
    bootstrap_frame.to_csv(run_dir / "bootstrap_stability.csv", index=False)

    plt.figure(figsize=(16, 12))
    sns.heatmap(similarity, cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("GPT-2 head index (layer-major)")
    plt.ylabel("BERT head index (layer-major)")
    plt.title("BERT-base / GPT-2 small linear CKA by attention head")
    plt.tight_layout()
    plt.savefig(run_dir / "cka_heatmap.png", dpi=180)
    plt.close()

    def mean_match_score(matches: list[Any]) -> float | None:
        return float(np.mean([item.score for item in matches])) if matches else None

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dataset_metadata": data_metadata,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": device,
        "sample_count": len(texts),
        "sampled_page_count": len(np.unique(cluster_ids)),
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "threshold": threshold,
        "methods": {
            "global_partial_hungarian": {
                "matched_count": len(global_matches),
                "mean_matched_score": mean_match_score(global_matches),
            },
            "same_index": {
                "matched_count": len(same_index),
                "mean_matched_score": mean_match_score(same_index),
            },
            "same_layer_hungarian": {
                "matched_count": len(same_layer),
                "mean_matched_score": mean_match_score(same_layer),
            },
        },
        "random_baseline_distribution": random_summary,
        "bootstrap": bootstrap_summary,
        "permutation_null": permutation_summary,
        "encoder": encoder_metadata,
        "decoder": decoder_metadata,
        "config": config,
        "interpretation_warning": (
            "This remains a pretrained representation-only pilot. A significant "
            "fixed-cardinality permutation result does not establish causal "
            "equivalence or pruning-utility transfer."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feasibility.yaml")
    args = parser.parse_args()
    run_dir = run(args.config)
    print(f"Representation pilot artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
