import numpy as np

from cra_hp.bootstrap import bootstrap_matching


def test_bootstrap_matching_resamples_whole_clusters() -> None:
    rng = np.random.default_rng(12)
    source = rng.normal(size=(3, 12, 5))
    target = rng.normal(size=(4, 12, 6))
    cluster_ids = np.repeat(np.arange(6), 2)

    frame, summary = bootstrap_matching(
        source,
        target,
        replicates=6,
        threshold=0.0,
        seed=13,
        cluster_ids=cluster_ids,
    )

    assert summary["resampling_method"] == "cluster"
    assert summary["cluster_count"] == 6.0
    assert summary["replicates"] == 6.0
    assert set(frame.columns) == {
        "source_head",
        "target_head",
        "match_frequency",
        "bootstrap_count",
        "replicates",
    }


def test_bootstrap_matching_rejects_mismatched_cluster_ids() -> None:
    source = np.ones((2, 8, 4))
    target = np.ones((2, 8, 4))
    try:
        bootstrap_matching(source, target, replicates=2, cluster_ids=np.arange(7))
    except ValueError as exc:
        assert "one ID per example" in str(exc)
    else:
        raise AssertionError("Expected mismatched cluster IDs to be rejected")
