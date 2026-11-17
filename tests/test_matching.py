import numpy as np

from cra_hp.matching import partial_hungarian, same_index_matching


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
