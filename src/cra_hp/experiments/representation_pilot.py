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
from cra_hp.extraction import extract_head_representations
from cra_hp.matching import (
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


def _load_shared_texts(config: dict[str, Any]) -> tuple[list[str], list[int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install model extras with: pip install -e '.[models]'") from exc

    data_cfg = config["data"]
    dataset = load_dataset(
        data_cfg["dataset"],
        data_cfg.get("config"),
        split=data_cfg["split"],
    )
    text_field = data_cfg.get("text_field", "text")
    min_characters = int(data_cfg.get("min_characters", 1))
    candidates = [
        (index, str(row[text_field]).strip())
        for index, row in enumerate(dataset)
        if str(row.get(text_field, "")).strip()
        and len(str(row.get(text_field, "")).strip()) >= min_characters
    ]
    count = int(data_cfg["sample_count"])
    if len(candidates) < count:
        raise ValueError(f"Dataset has only {len(candidates)} usable rows; {count} requested")

    rng = np.random.default_rng(int(data_cfg["sampling_seed"]))
    selected_positions = np.sort(
        rng.choice(len(candidates), size=count, replace=False)
    )
    selected = [candidates[int(position)] for position in selected_positions]
    return [text for _, text in selected], [index for index, _ in selected]


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
        import datasets

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

    texts, sample_ids = _load_shared_texts(config)
    sample_hashes = [
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
    ]
    (run_dir / "samples.json").write_text(
        json.dumps(
            {"dataset": config["data"], "sample_ids": sample_ids,
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
    np.savez_compressed(run_dir / "encoder_representations.npz",
                        representations=encoder.representations)
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
    np.savez_compressed(run_dir / "decoder_representations.npz",
                        representations=decoder.representations)

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
        "num_heads_per_layer": int(encoder.metadata["num_heads_per_layer"]),
    }
    decoder_meta = {
        "num_heads_per_layer": int(decoder.metadata["num_heads_per_layer"]),
    }

    all_match_rows: list[dict[str, Any]] = []
    all_match_rows.extend(_save_matches(
        run_dir / "matches_global_partial.csv", global_matches,
        encoder.metadata, decoder.metadata, "global_partial_hungarian"
    ))
    same_index = same_index_matching(similarity, threshold=threshold)
    all_match_rows.extend(_save_matches(
        run_dir / "matches_same_index.csv", same_index,
        encoder.metadata, decoder.metadata, "same_index"
    ))
    same_layer = same_layer_matching(
        similarity,
        encoder_meta["num_heads_per_layer"],
        decoder_meta["num_heads_per_layer"],
        threshold=threshold,
    )
    all_match_rows.extend(_save_matches(
        run_dir / "matches_same_layer.csv", same_layer,
        encoder.metadata, decoder.metadata, "same_layer_hungarian"
    ))
    random_baseline = random_matching(
        similarity, len(global_matches), seed=seed
    )
    all_match_rows.extend(_save_matches(
        run_dir / "matches_random.csv", random_baseline,
        encoder.metadata, decoder.metadata, "random_fixed_cardinality"
    ))
    pd.DataFrame(all_match_rows).to_csv(run_dir / "all_matches.csv", index=False)

    # Break paired-example correspondence to form an empirical null distribution.
    bootstrap_cfg = config["bootstrap"]
    permutation_replicates = int(bootstrap_cfg.get("permutation_replicates", 50))
    permutation_rng = np.random.default_rng(seed + 1)
    observed_mean = (
        float(np.mean([item.score for item in global_matches])) if global_matches else 0.0
    )
    permutation_rows = []
    for replicate in range(permutation_replicates):
        shuffled_indices = permutation_rng.permutation(source.shape[1])
        null_scores = cka_similarity_matrix(source, target[:, shuffled_indices, :])
        null_matches = partial_hungarian(null_scores, threshold=threshold)
        null_mean = (
            float(np.mean([item.score for item in null_matches])) if null_matches else 0.0
        )
        permutation_rows.append({
            "replicate": replicate,
            "matched_count": len(null_matches),
            "mean_matched_score": null_mean,
        })
    pd.DataFrame(permutation_rows).to_csv(run_dir / "permutation_null.csv", index=False)
    null_values = [row["mean_matched_score"] for row in permutation_rows]
    permutation_summary = {
        "replicates": permutation_replicates,
        "observed_mean_matched_score": observed_mean,
        "null_mean": float(np.mean(null_values)) if null_values else None,
        "null_std": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else None,
        "empirical_p_value": (
            (1 + sum(value >= observed_mean for value in null_values))
            / (1 + len(null_values))
        ) if null_values else None,
    }

    bootstrap_frame, bootstrap_summary = bootstrap_matching(
        source,
        target,
        replicates=int(bootstrap_cfg["pilot_replicates"]),
        threshold=threshold,
        seed=seed,
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
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": device,
        "sample_count": len(texts),
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
            "random_fixed_cardinality": {
                "matched_count": len(random_baseline),
                "mean_matched_score": mean_match_score(random_baseline),
            },
        },
        "bootstrap": bootstrap_summary,
        "permutation_null": permutation_summary,
        "encoder": encoder.metadata,
        "decoder": decoder.metadata,
        "config": config,
        "interpretation_warning": (
            "Representation similarity is exploratory and does not establish causal "
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
