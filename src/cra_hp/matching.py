"""Partial one-to-one assignment with explicit unmatched heads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class Match:
    source_head: int
    target_head: int
    score: float


def partial_hungarian(
    scores: FloatArray,
    threshold: float = 0.15,
) -> list[Match]:
    """Maximize total score above threshold while allowing unmatched nodes.

    Dummy rows/columns have zero cost. Real edges have negative cost only when
    their score exceeds the threshold, so an unmatched assignment is always
    available and weak edges are never forced.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must contain only finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    n_source, n_target = matrix.shape
    if n_source == 0 or n_target == 0:
        return []

    size = n_source + n_target
    cost = np.zeros((size, size), dtype=np.float64)
    real_cost = -(matrix - threshold)
    real_cost[matrix <= threshold] = 0.0
    cost[:n_source, :n_target] = real_cost

    rows, cols = linear_sum_assignment(cost)
    matches: list[Match] = []
    for row, col in zip(rows, cols, strict=True):
        if row < n_source and col < n_target and matrix[row, col] > threshold:
            matches.append(Match(int(row), int(col), float(matrix[row, col])))
    return sorted(matches, key=lambda item: item.source_head)


def same_index_matching(scores: FloatArray, threshold: float = 0.15) -> list[Match]:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    return [
        Match(i, i, float(matrix[i, i]))
        for i in range(min(matrix.shape))
        if matrix[i, i] > threshold
    ]


def same_layer_matching(
    scores: FloatArray,
    source_heads_per_layer: int,
    target_heads_per_layer: int,
    threshold: float = 0.15,
) -> list[Match]:
    """Match only heads whose integer layer indices are equal."""
    matrix = np.asarray(scores, dtype=np.float64)
    constrained = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            source_layer = i // source_heads_per_layer
            target_layer = j // target_heads_per_layer
            if source_layer == target_layer:
                constrained[i, j] = matrix[i, j]
    return partial_hungarian(constrained, threshold=threshold)


def random_matching(
    scores: FloatArray, matched_count: int, seed: int = 0
) -> list[Match]:
    """Create a random one-to-one assignment with a fixed match count."""
    matrix = np.asarray(scores, dtype=np.float64)
    n_source, n_target = matrix.shape
    count = min(max(0, matched_count), n_source, n_target)
    rng = np.random.default_rng(seed)
    source_ids = rng.choice(n_source, size=count, replace=False)
    target_ids = rng.choice(n_target, size=count, replace=False)
    return [
        Match(int(i), int(j), float(matrix[i, j]))
        for i, j in zip(source_ids, target_ids, strict=True)
    ]


def matches_to_pairs(matches: list[Match]) -> set[tuple[int, int]]:
    return {(match.source_head, match.target_head) for match in matches}
