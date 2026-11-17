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

    # Center each head's sample-by-feature matrix once.
    source_centered = source.astype(np.float64, copy=True)
    target_centered = target.astype(np.float64, copy=True)
    source_centered -= source_centered.mean(axis=1, keepdims=True)
    target_centered -= target_centered.mean(axis=1, keepdims=True)

    # Feature-space covariance matrices keep memory small (head x d x d).
    source_cov = np.einsum(
        "hni,hnj->hij", source_centered, source_centered, optimize=True
    )
    target_cov = np.einsum(
        "hni,hnj->hij", target_centered, target_centered, optimize=True
    )
    source_norm = np.sqrt(np.square(source_cov).sum(axis=(1, 2)))
    target_norm = np.sqrt(np.square(target_cov).sum(axis=(1, 2)))
    denominator = source_norm[:, None] * target_norm[None, :]

    scores = np.zeros((source.shape[0], target.shape[0]), dtype=np.float64)
    # Chunk source heads to bound the temporary cross-covariance tensor.
    chunk_size = max(1, min(8, source.shape[0]))
    for start in range(0, source.shape[0], chunk_size):
        stop = min(start + chunk_size, source.shape[0])
        cross = np.einsum(
            "hni,knj->hkij",
            source_centered[start:stop],
            target_centered,
            optimize=True,
        )
        numerator = np.square(cross).sum(axis=(2, 3))
        block_denominator = denominator[start:stop]
        valid = block_denominator > epsilon
        block = np.zeros_like(numerator)
        block[valid] = numerator[valid] / block_denominator[valid]
        scores[start:stop] = np.clip(block, 0.0, 1.0)
    return scores
