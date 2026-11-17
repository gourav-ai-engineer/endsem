"""Similarity functions for cross-model head representations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _center_columns(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("At least two examples are required for CKA")
    return x - x.mean(axis=0, keepdims=True)


def linear_cka(x: FloatArray, y: FloatArray, epsilon: float = 1e-12) -> float:
    """Compute linear centered kernel alignment for paired examples.

    Feature dimensions may differ; the number and ordering of examples must match.
    """
    x_centered = _center_columns(x)
    y_centered = _center_columns(y)
    if x_centered.shape[0] != y_centered.shape[0]:
        raise ValueError("X and Y must contain the same paired examples")

    cross = x_centered.T @ y_centered
    x_gram = x_centered.T @ x_centered
    y_gram = y_centered.T @ y_centered
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.sqrt(np.square(x_gram).sum() * np.square(y_gram).sum())
    )
    if denominator <= epsilon:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def cka_similarity_matrix(
    source: FloatArray, target: FloatArray, epsilon: float = 1e-12
) -> NDArray[np.float64]:
    """Compute all pairwise CKA scores.

    Arrays have shape (n_heads, n_examples, feature_dim). Source and target
    feature dimensions may differ, but n_examples must be identical.
    """
    source = np.asarray(source)
    target = np.asarray(target)
    if source.ndim != 3 or target.ndim != 3:
        raise ValueError("Representations must have shape (heads, examples, features)")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target must use the same paired examples")

    scores = np.zeros((source.shape[0], target.shape[0]), dtype=np.float64)
    for i in range(source.shape[0]):
        for j in range(target.shape[0]):
            scores[i, j] = linear_cka(source[i], target[j], epsilon=epsilon)
    return scores
