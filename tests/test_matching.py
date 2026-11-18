import numpy as np
import pytest

from cra_hp.matching import (
    fixed_cardinality_hungarian,
    partial_hungarian,
    same_index_matching,
)


def test_partial_hungarian_leaves_weak_heads_unmatched() -> None:
    scores = np.array([
        [0.9, 0.1, 0.0],
        [0.2, 0.8, 0.1],
        [0.0, 0.1, 0.05],
    ])
    matches = partial_hungarian(scores, threshold=0.5)
    pairs = {(m.source_head, m.target_head) for m in matches}
    assert pairs == {(0, 0), (1, 1)}


def test_partial_hungarian_is_one_to_one() -> None:
    scores = np.array([
        [0.95, 0.90],
        [0.94, 0.10],
    ])
    matches = partial_hungarian(scores, threshold=0.2)
    assert len({m.source_head for m in matches}) == len(matches)
    assert len({m.target_head for m in matches}) == len(matches)


def test_same_index_matching_uses_threshold() -> None:
    scores = np.array([[0.8, 0.2], [0.3, 0.1]])
    matches = same_index_matching(scores, threshold=0.15)
    assert [(m.source_head, m.target_head) for m in matches] == [(0, 0)]


def test_fixed_cardinality_hungarian_returns_exact_k_optimum() -> None:
    scores = np.array([
        [0.90, 0.80, 0.10],
        [0.85, 0.10, 0.10],
        [0.10, 0.95, 0.90],
    ])
    matches = fixed_cardinality_hungarian(scores, matched_count=2)
    pairs = {(m.source_head, m.target_head) for m in matches}
    assert len(matches) == 2
    assert len({i for i, _ in pairs}) == 2
    assert len({j for _, j in pairs}) == 2
    assert sum(m.score for m in matches) == pytest.approx(1.85)


def test_fixed_cardinality_supports_rectangular_matrices() -> None:
    scores = np.array([[0.2, 0.9, 0.3], [0.8, 0.1, 0.4]])
    matches = fixed_cardinality_hungarian(scores, matched_count=1)
    assert len(matches) == 1
    assert matches[0].score == pytest.approx(0.9)


def test_fixed_cardinality_zero_and_invalid_count() -> None:
    scores = np.eye(3)
    assert fixed_cardinality_hungarian(scores, 0) == []
    with pytest.raises(ValueError):
        fixed_cardinality_hungarian(scores, 4)


def test_fixed_cardinality_respects_allowed_mask() -> None:
    scores = np.array([
        [0.1, 0.99, 0.2],
        [0.98, 0.1, 0.3],
        [0.2, 0.4, 0.95],
    ])
    allowed = np.eye(3, dtype=bool)
    matches = fixed_cardinality_hungarian(scores, 2, allowed_mask=allowed)
    assert len(matches) == 2
    assert all(match.source_head == match.target_head for match in matches)
