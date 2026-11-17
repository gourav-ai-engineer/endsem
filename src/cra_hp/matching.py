"""Partial and fixed-cardinality one-to-one head assignment utilities."""

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


def _validated_scores(scores: FloatArray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must contain only finite values")
    return matrix


def partial_hungarian(
    scores: FloatArray,
    threshold: float = 0.15,
) -> list[Match]:
    """Maximize total score above threshold while allowing unmatched nodes.

    This is the exploratory thresholded assignment used to discover candidate
    correspondences. It is not used as the statistic for permutation testing.
    """
    matrix = _validated_scores(scores)
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


def fixed_cardinality_hungarian(
    scores: FloatArray,
    matched_count: int,
    allowed_mask: NDArray[np.bool_] | None = None,
) -> list[Match]:
    """Return the maximum-weight one-to-one assignment with exactly k pairs.

    Unlike thresholded partial matching, this fixes assignment cardinality.
    It is appropriate for comparing an observed alignment with permutation
    nulls because every null replicate is evaluated at the same k. Scores are
    maximized without thresholding; callers should choose k independently of
    the permuted scores (for example, from the observed partial assignment).
    """
    matrix = _validated_scores(scores)
    n_source, n_target = matrix.shape
    if allowed_mask is not None:
        allowed_mask = np.asarray(allowed_mask, dtype=bool)
        if allowed_mask.shape != matrix.shape:
            raise ValueError("allowed_mask must have the same shape as scores")
        if int(allowed_mask.sum()) < matched_count:
            raise ValueError("allowed_mask has fewer than matched_count allowed edges")
    if not isinstance(matched_count, (int, np.integer)):
        raise TypeError("matched_count must be an integer")
    k = int(matched_count)
    if k < 0 or k > min(n_source, n_target):
        raise ValueError("matched_count must be between 0 and min(matrix dimensions)")
    if k == 0:
        return []

    # Add (n_source-k) dummy target columns and (n_target-k) dummy source
    # rows. A large dummy-to-dummy penalty forces exactly k real-real pairs.
    n_dummy_columns = n_source - k
    n_dummy_rows = n_target - k
    size = n_source + n_dummy_rows
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:n_source, :n_target] = -matrix
    if allowed_mask is not None:
        cost[:n_source, :n_target][~allowed_mask] = 1.0e6
    if n_dummy_rows and n_dummy_columns:
        cost[n_source:, n_target:] = 1.0e6

    rows, cols = linear_sum_assignment(cost)
    matches = [
        Match(int(row), int(col), float(matrix[row, col]))
        for row, col in zip(rows, cols, strict=True)
        if row < n_source and col < n_target
    ]
    if len(matches) != k:
        raise RuntimeError(
            f"Fixed-cardinality assignment returned {len(matches)} pairs; expected {k}"
        )
    if allowed_mask is not None and any(
        not allowed_mask[item.source_head, item.target_head] for item in matches
    ):
        raise ValueError("No feasible one-to-one assignment exists for this allowed_mask")
    return sorted(matches, key=lambda item: item.source_head)


def same_index_matching(scores: FloatArray, threshold: float = 0.15) -> list[Match]:
    matrix = _validated_scores(scores)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
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
    matrix = _validated_scores(scores)
    if source_heads_per_layer < 1 or target_heads_per_layer < 1:
        raise ValueError("heads_per_layer values must be positive")
    constrained = np.full_like(matrix, -1.0e6)
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
    matrix = _validated_scores(scores)
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
