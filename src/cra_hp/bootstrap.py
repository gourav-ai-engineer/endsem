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
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Resample paired examples and estimate pairwise matching frequency."""
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target must share paired example dimension")

    n_examples = source.shape[1]
    rng = np.random.default_rng(seed)
    pair_counts: dict[tuple[int, int], int] = {}
    pair_sets: list[set[tuple[int, int]]] = []

    for _ in range(replicates):
        indices = rng.integers(0, n_examples, size=n_examples)
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

    summary = {
        "replicates": float(replicates),
        "unique_pairs_observed": float(len(pair_counts)),
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "median_pairwise_jaccard": float(np.median(jaccards)) if jaccards else 1.0,
        "mean_matches_per_replicate": float(np.mean([len(p) for p in pair_sets])),
    }
    return pair_frame, summary
