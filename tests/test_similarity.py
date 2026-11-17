import numpy as np
import pytest

from cra_hp.similarity import cka_similarity_matrix, linear_cka


def test_linear_cka_is_one_for_identical_representations() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 8))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-10)


def test_linear_cka_supports_different_feature_dimensions() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20, 8))
    y = rng.normal(size=(20, 5))
    score = linear_cka(x, y)
    assert 0.0 <= score <= 1.0


def test_cka_matrix_has_expected_shape() -> None:
    rng = np.random.default_rng(9)
    source = rng.normal(size=(3, 12, 8))
    target = rng.normal(size=(4, 12, 6))
    scores = cka_similarity_matrix(source, target)
    assert scores.shape == (3, 4)


def test_cka_rejects_mismatched_examples() -> None:
    with pytest.raises(ValueError):
        linear_cka(np.ones((10, 4)), np.ones((9, 4)))


def test_vectorized_matrix_matches_scalar_cka() -> None:
    rng = np.random.default_rng(10)
    source = rng.normal(size=(2, 15, 7))
    target = rng.normal(size=(3, 15, 5))
    matrix = cka_similarity_matrix(source, target)
    for i in range(source.shape[0]):
        for j in range(target.shape[0]):
            assert matrix[i, j] == pytest.approx(linear_cka(source[i], target[j]), abs=1e-10)
