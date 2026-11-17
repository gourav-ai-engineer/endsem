from __future__ import annotations

import zipfile
from pathlib import Path

from cra_hp.data import load_wikitext_2_raw


def _make_archive(path: Path) -> Path:
    member = "wikitext-2-raw/wiki.train.raw"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            member,
            "short\n"
            "This is a sufficiently long first example.\n"
            "Another sufficiently long second example.\n"
            "A third sufficiently long example for testing.\n",
        )
    return path


def test_load_wikitext_raw_from_existing_archive_is_deterministic(tmp_path: Path) -> None:
    archive_path = _make_archive(tmp_path / "wikitext.zip")
    config = {
        "dataset": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "train",
        "archive_path": str(archive_path),
        "sample_count": 2,
        "sampling_seed": 17,
        "min_characters": 20,
    }

    texts_a, ids_a, metadata_a = load_wikitext_2_raw(config)
    texts_b, ids_b, metadata_b = load_wikitext_2_raw(config)

    assert texts_a == texts_b
    assert ids_a == ids_b
    assert len(texts_a) == 2
    assert all(len(text) >= 20 for text in texts_a)
    assert metadata_a["archive_sha256"] == metadata_b["archive_sha256"]
    assert metadata_a["archive_member"] == "wikitext-2-raw/wiki.train.raw"


def test_load_wikitext_rejects_unknown_split(tmp_path: Path) -> None:
    config = {
        "dataset": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "dev",
        "archive_path": str(tmp_path / "unused.zip"),
        "sample_count": 1,
        "sampling_seed": 1,
    }

    try:
        load_wikitext_2_raw(config)
    except ValueError as exc:
        assert "Unsupported WikiText-2 split" in str(exc)
    else:
        raise AssertionError("Expected unsupported split to raise ValueError")
