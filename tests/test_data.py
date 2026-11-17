from __future__ import annotations

import json
from typing import Any

import pytest
from typing_extensions import Self

from cra_hp import data


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _page(offset: int, total: int = 250) -> dict[str, Any]:
    rows = []
    for index in range(offset, min(offset + 100, total)):
        text = "" if index % 3 == 0 else f"WikiText example row number {index}."
        rows.append({"row": {"text": text}})
    return {"num_rows_total": total, "rows": rows}


def test_load_wikitext_api_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    requested_offsets: list[int] = []

    def fake_urlopen(request: Any, timeout: int = 30) -> _FakeResponse:
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(request.full_url).query)
        offset = int(query["offset"][0])
        requested_offsets.append(offset)
        return _FakeResponse(_page(offset))

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    config = {
        "dataset": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "train",
        "sample_count": 20,
        "sampling_seed": 17,
        "min_characters": 20,
    }
    texts_a, ids_a, metadata_a = data.load_wikitext_2_raw(config)
    texts_b, ids_b, _metadata_b = data.load_wikitext_2_raw(config)

    assert texts_a == texts_b
    assert ids_a == ids_b
    assert len(texts_a) == 20
    assert all(len(text) >= 20 for text in texts_a)
    assert metadata_a["total_row_count"] == 250
    assert metadata_a["sample_row_indices"] == ids_a
    assert metadata_a["dataset_revision"].startswith("main")
    assert metadata_a["fetched_row_count"] >= 20


def test_load_wikitext_rejects_unknown_split() -> None:
    config = {
        "dataset": "wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "dev",
        "sample_count": 1,
        "sampling_seed": 1,
    }
    with pytest.raises(ValueError, match="Unsupported WikiText-2 split"):
        data.load_wikitext_2_raw(config)


def test_load_wikitext_rejects_unknown_config() -> None:
    config = {
        "dataset": "wikitext",
        "config": "wikitext-103-v1",
        "split": "train",
        "sample_count": 1,
        "sampling_seed": 1,
    }
    with pytest.raises(ValueError, match="wikitext-2-raw-v1"):
        data.load_wikitext_2_raw(config)
