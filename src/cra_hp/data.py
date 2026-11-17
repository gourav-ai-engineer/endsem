"""Download and load the official raw WikiText-2 archive without PyArrow."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

WIKITEXT_2_URL = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
)
WIKITEXT_2_MEMBERS = {
    "train": "wikitext-2-raw/wiki.train.raw",
    "validation": "wikitext-2-raw/wiki.valid.raw",
    "test": "wikitext-2-raw/wiki.test.raw",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_archive(data_cfg: dict[str, Any]) -> tuple[Path, str]:
    cache_dir = Path(data_cfg.get("cache_dir", "data/raw"))
    archive_path = Path(
        data_cfg.get("archive_path", cache_dir / "wikitext-2-raw-v1.zip")
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
        request = urllib.request.Request(
            WIKITEXT_2_URL, headers={"User-Agent": "CRA-HP-research/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary_path.open("wb") as destination:
                    while chunk := response.read(1024 * 1024):
                        destination.write(chunk)
            temporary_path.replace(archive_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    return archive_path, _sha256_file(archive_path)


def load_wikitext_2_raw(
    data_cfg: dict[str, Any],
) -> tuple[list[str], list[int], dict[str, Any]]:
    """Load deterministically sampled lines from the official WikiText-2 raw ZIP.

    The archive is cached locally. Set data.archive_path to an existing ZIP
    file to run fully offline after obtaining the official archive.
    """
    if data_cfg.get("dataset") != "wikitext":
        raise ValueError("The standard-library loader currently supports dataset='wikitext'")

    dataset_config = data_cfg.get("config", "wikitext-2-raw-v1")
    if dataset_config != "wikitext-2-raw-v1":
        raise ValueError(
            "The standard-library loader supports config='wikitext-2-raw-v1' only"
        )

    split = str(data_cfg["split"])
    if split not in WIKITEXT_2_MEMBERS:
        raise ValueError(f"Unsupported WikiText-2 split {split!r}")

    archive_path, archive_sha256 = _ensure_archive(data_cfg)
    member_name = WIKITEXT_2_MEMBERS[split]
    min_characters = int(data_cfg.get("min_characters", 1))

    try:
        with zipfile.ZipFile(archive_path) as archive:
            raw_text = archive.read(member_name).decode("utf-8")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Could not read {member_name!r} from {archive_path}. "
            "Check that this is the official WikiText-2 raw archive."
        ) from exc

    candidates = [
        (line_number, line.strip())
        for line_number, line in enumerate(io.StringIO(raw_text))
        if line.strip() and len(line.strip()) >= min_characters
    ]
    count = int(data_cfg["sample_count"])
    if count < 1:
        raise ValueError("data.sample_count must be >= 1")
    if len(candidates) < count:
        raise ValueError(
            f"Dataset has only {len(candidates)} usable rows; {count} requested"
        )

    rng = np.random.default_rng(int(data_cfg["sampling_seed"]))
    selected_positions = np.sort(
        rng.choice(len(candidates), size=count, replace=False)
    )
    selected = [candidates[int(position)] for position in selected_positions]
    metadata = {
        "dataset": "wikitext",
        "config": dataset_config,
        "split": split,
        "source_url": WIKITEXT_2_URL,
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_member": member_name,
        "usable_line_count": len(candidates),
        "min_characters": min_characters,
        "sampling_seed": int(data_cfg["sampling_seed"]),
        "sample_count": count,
    }
    return [text for _, text in selected], [index for index, _ in selected], metadata
