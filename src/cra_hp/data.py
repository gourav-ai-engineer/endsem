"""Load deterministic WikiText-2 samples through the Hugging Face Dataset Viewer API."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

WIKITEXT_2_DATASET = "Salesforce/wikitext"
WIKITEXT_2_API_URL = "https://datasets-server.huggingface.co/rows"
WIKITEXT_2_PAGE_SIZE = 100


def _fetch_page(
    split: str,
    offset: int,
    *,
    revision: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "dataset": WIKITEXT_2_DATASET,
            "config": "wikitext-2-raw-v1",
            "split": split,
            "offset": offset,
            "length": WIKITEXT_2_PAGE_SIZE,
            "revision": revision,
        }
    )
    request = urllib.request.Request(
        f"{WIKITEXT_2_API_URL}?{params}",
        headers={"User-Agent": "CRA-HP-research/0.2"},
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload.get("rows"), list):
                raise RuntimeError("Hugging Face rows API returned an invalid response")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(
        f"Could not fetch WikiText-2 split={split!r}, offset={offset} "
        f"after {retries + 1} attempts"
    ) from last_error


def load_wikitext_2_raw(
    data_cfg: dict[str, Any],
) -> tuple[list[str], list[int], dict[str, Any]]:
    """Fetch a deterministic sample of non-empty raw WikiText-2 rows.

    Pages are selected uniformly with a seeded RNG, then rows are sampled
    uniformly from the fetched pages. The API is paginated at 100 rows/page.
    """
    if data_cfg.get("dataset") != "wikitext":
        raise ValueError("The loader currently supports dataset='wikitext' only")

    dataset_config = data_cfg.get("config", "wikitext-2-raw-v1")
    if dataset_config != "wikitext-2-raw-v1":
        raise ValueError(
            "The loader supports config='wikitext-2-raw-v1' only"
        )

    split = str(data_cfg["split"])
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported WikiText-2 split {split!r}")

    count = int(data_cfg["sample_count"])
    if count < 1:
        raise ValueError("data.sample_count must be >= 1")

    seed = int(data_cfg["sampling_seed"])
    min_characters = int(data_cfg.get("min_characters", 1))
    revision = str(data_cfg.get("revision", "main"))
    timeout = int(data_cfg.get("request_timeout", 30))
    retries = int(data_cfg.get("request_retries", 3))

    # The first response supplies the split size and is reused if page zero
    # is selected, avoiding an additional metadata request.
    first_page = _fetch_page(
        split, 0, revision=revision, timeout=timeout, retries=retries
    )
    total_rows = int(first_page["num_rows_total"])
    if total_rows < 1:
        raise ValueError(f"WikiText-2 split {split!r} is empty")
    total_pages = (total_rows + WIKITEXT_2_PAGE_SIZE - 1) // WIKITEXT_2_PAGE_SIZE

    rng = np.random.default_rng(seed)
    # Fetch roughly twice the expected number of rows needed, allowing for
    # blank/short WikiText lines while keeping API calls modest.
    initial_page_count = min(
        total_pages, max(8, int(np.ceil(count / 50)) * 2)
    )
    page_order = rng.permutation(total_pages).tolist()
    selected_pages = page_order[:initial_page_count]
    remaining_pages = page_order[initial_page_count:]

    candidates: list[tuple[int, str]] = []
    fetched_rows = 0

    def add_page(offset: int, payload: dict[str, Any]) -> None:
        nonlocal fetched_rows
        rows = payload["rows"]
        fetched_rows += len(rows)
        for local_index, item in enumerate(rows):
            row = item.get("row") or {}
            value = row.get("text", "")
            if isinstance(value, str):
                cleaned = value.strip()
                if len(cleaned) >= min_characters:
                    candidates.append((offset + local_index, cleaned))

    for page_index in selected_pages:
        offset = page_index * WIKITEXT_2_PAGE_SIZE
        payload = first_page if page_index == 0 else _fetch_page(
            split, offset, revision=revision, timeout=timeout, retries=retries
        )
        add_page(offset, payload)

    while len(candidates) < count and remaining_pages:
        page_index = remaining_pages.pop(0)
        offset = page_index * WIKITEXT_2_PAGE_SIZE
        payload = _fetch_page(
            split, offset, revision=revision, timeout=timeout, retries=retries
        )
        add_page(offset, payload)

    if len(candidates) < count:
        raise ValueError(
            f"WikiText-2 split {split!r} has only {len(candidates)} usable "
            f"rows in the fetched pages; {count} requested. Lower "
            "data.min_characters or increase the page sampling budget."
        )

    chosen_positions = rng.choice(len(candidates), size=count, replace=False)
    selected = [candidates[int(position)] for position in chosen_positions]
    selected.sort(key=lambda item: item[0])

    metadata = {
        "dataset": WIKITEXT_2_DATASET,
        "config": dataset_config,
        "split": split,
        "dataset_revision": revision,
        "source_url": WIKITEXT_2_API_URL,
        "total_row_count": total_rows,
        "page_size": WIKITEXT_2_PAGE_SIZE,
        "fetched_page_count": len(selected_pages)
        + (len(remaining_pages) and 0),
        "fetched_row_count": fetched_rows,
        "usable_row_count": len(candidates),
        "min_characters": min_characters,
        "sampling_seed": seed,
        "sample_count": count,
        "sampling_note": (
            "Seeded page-level sampling followed by seeded row sampling; "
            "blank and short rows are excluded."
        ),
    }
    # Count pages actually fetched, including any extra pages fetched to meet
    # the requested sample size.
    metadata["fetched_page_count"] = fetched_rows // WIKITEXT_2_PAGE_SIZE
    if fetched_rows % WIKITEXT_2_PAGE_SIZE:
        metadata["fetched_page_count"] += 1
    metadata["sample_row_indices"] = [index for index, _ in selected]
    return [text for _, text in selected], [index for index, _ in selected], metadata
