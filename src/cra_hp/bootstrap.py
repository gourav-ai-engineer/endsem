"""Bootstrap confidence and stability estimates for partial head matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cra_hp.matching import matches_to_pairs, partial_hungarian
from cra_hp.similarity import cka_similarity_matrix


def bootstrap_matching(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    replicates: int = 50,
    threshold: float = 0.15,
    seed: int = 0,
    cluster_ids: NDArray | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Estimate pairwise matching stability with iid or cluster resampling.

    If cluster_ids are supplied, entire clusters (e.g. WikiText API pages) are
    resampled with replacement. This avoids treating rows sampled from the same
    source page as independent observations.
    """
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target must share paired example dimension")
    n_examples = source.shape[1]
    if n_examples < 2:
        raise ValueError("At least two examples are required for bootstrap matching")

    groups: np.ndarray | None = None
    if cluster_ids is not None:
        groups = np.asarray(cluster_ids)
        if groups.ndim != 1 or len(groups) != n_examples:
            raise ValueError("cluster_ids must be one-dimensional with one ID per example")
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError("Cluster bootstrap requires at least two distinct clusters")
    else:
        unique_groups = np.array([], dtype=object)

    rng = np.random.default_rng(seed)
    pair_counts: dict[tuple[int, int], int] = {}
    pair_sets: list[set[tuple[int, int]]] = []
    sampled_example_counts: list[int] = []

    for _ in range(replicates):
        if groups is None:
            indices = rng.integers(0, n_examples, size=n_examples)
        else:
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            group_indices = [np.flatnonzero(groups == group) for group in sampled_groups]
            indices = np.concatenate(group_indices)
        if len(indices) < 2:
            raise RuntimeError("Bootstrap replicate contains fewer than two examples")
        sampled_example_counts.append(int(len(indices)))
        scores = cka_similarity_matrix(source[:, indices, :], target[:, indices, :])
        pairs = matches_to_pairs(partial_hungarian(scores, threshold=threshold))
        pair_sets.append(pairs)
        for pair in pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    rows = [
        {
            "source_head": i,
            "target_head": j,
            "match_frequency": count / replicates,
            "bootstrap_count": count,
            "replicates": replicates,
        }
        for (i, j), count in sorted(pair_counts.items())
    ]
    pair_frame = pd.DataFrame(
        rows,
        columns=[
            "source_head",
            "target_head",
            "match_frequency",
            "bootstrap_count",
            "replicates",
        ],
    )

    jaccards: list[float] = []
    for left in range(len(pair_sets)):
        for right in range(left + 1, len(pair_sets)):
            union = pair_sets[left] | pair_sets[right]
            if union:
                jaccards.append(len(pair_sets[left] & pair_sets[right]) / len(union))
            else:
                jaccards.append(1.0)

    summary: dict[str, float | str] = {
        "replicates": float(replicates),
        "resampling_method": "cluster" if groups is not None else "iid_examples",
        "cluster_count": float(len(unique_groups)) if groups is not None else 0.0,
        "mean_examples_per_replicate": float(np.mean(sampled_example_counts)),
        "unique_pairs_observed": float(len(pair_counts)),
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "median_pairwise_jaccard": float(np.median(jaccards)) if jaccards else 1.0,
        "mean_matches_per_replicate": float(np.mean([len(p) for p in pair_sets])),
    }
    return pair_frame, summary
