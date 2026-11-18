"""Pinned SST-2 loading and deterministic fit/selection/calibration splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TextExample:
    example_id: str
    text: str
    label: int


@dataclass(frozen=True)
class SST2Data:
    fit: list[TextExample]
    selection: list[TextExample]
    calibration: list[TextExample]
    heldout: list[TextExample]
    metadata: dict[str, Any]


def _digest(examples: list[TextExample]) -> str:
    payload = [
        {"id": row.example_id, "text": row.text, "label": row.label}
        for row in examples
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stratified_take(
    examples: list[TextExample], count: int, seed: int
) -> list[TextExample]:
    if count >= len(examples):
        return examples
    if count < 2:
        raise ValueError("A stratified subset must contain at least two examples")
    labels = np.asarray([row.label for row in examples])
    chosen, _ = train_test_split(
        np.arange(len(examples)),
        train_size=count,
        random_state=seed,
        stratify=labels,
    )
    return [examples[int(index)] for index in sorted(chosen)]


def load_sst2_pilot(data_cfg: dict[str, Any]) -> SST2Data:
    """Load a revision-pinned GLUE/SST-2 dataset and split train deterministically."""
    try:
        from datasets import load_dataset
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "SST-2 loading requires a working `datasets`/`pyarrow` installation. "
            "Repair the optional model environment, then rerun the pilot."
        ) from exc

    revision = str(data_cfg["revision"])
    dataset_id = str(data_cfg["id"])
    dataset_config = str(data_cfg["config"])
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        revision=revision,
    )
    train_raw = dataset[str(data_cfg["train_split"])]
    heldout_raw = dataset[str(data_cfg["heldout_split"])]
    text_column = str(data_cfg["text_column"])
    label_column = str(data_cfg["label_column"])
    id_column = str(data_cfg["row_id_column"])

    def convert(split: Any, split_name: str) -> list[TextExample]:
        rows: list[TextExample] = []
        for ordinal, row in enumerate(split):
            label = int(row[label_column])
            if label not in (0, 1):
                raise ValueError(f"Unexpected SST-2 label {label} in {split_name}")
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or invalid text in {split_name} row {ordinal}")
            stable_id = row.get(id_column, ordinal)
            rows.append(TextExample(f"{split_name}:{stable_id}", text, label))
        return rows

    train = convert(train_raw, str(data_cfg["train_split"]))
    heldout = convert(heldout_raw, str(data_cfg["heldout_split"]))
    partition = data_cfg["train_partition"]
    fit_fraction, selection_fraction, calibration_fraction = map(
        float, partition["fractions"]
    )
    if not np.isclose(fit_fraction + selection_fraction + calibration_fraction, 1.0):
        raise ValueError("Train partition fractions must sum to 1")
    split_seed = int(partition["seed"])

    all_indices = np.arange(len(train))
    labels = np.asarray([row.label for row in train])
    fit_indices, remainder_indices = train_test_split(
        all_indices,
        train_size=fit_fraction,
        random_state=split_seed,
        stratify=labels,
    )
    remainder_labels = labels[remainder_indices]
    calibration_share = calibration_fraction / (selection_fraction + calibration_fraction)
    selection_indices, calibration_indices = train_test_split(
        remainder_indices,
        test_size=calibration_share,
        random_state=split_seed + 1,
        stratify=remainder_labels,
    )
    fit = [train[int(index)] for index in sorted(fit_indices)]
    selection = [train[int(index)] for index in sorted(selection_indices)]
    calibration = [train[int(index)] for index in sorted(calibration_indices)]

    fit = _stratified_take(
        fit,
        int(partition.get("max_fit_examples", len(fit))),
        split_seed + 2,
    )
    selection = _stratified_take(
        selection,
        int(partition.get("max_selection_examples", len(selection))),
        split_seed + 3,
    )
    calibration = _stratified_take(
        calibration,
        int(partition.get("calibration_examples", len(calibration))),
        split_seed + 4,
    )

    metadata = {
        "dataset_id": dataset_id,
        "config": dataset_config,
        "revision": revision,
        "source_splits": {
            "fit_selection_calibration": str(data_cfg["train_split"]),
            "heldout": str(data_cfg["heldout_split"]),
        },
        "train_partition_seed": split_seed,
        "split_counts": {
            "fit": len(fit),
            "selection": len(selection),
            "calibration": len(calibration),
            "heldout": len(heldout),
        },
        "split_sha256": {
            "fit": _digest(fit),
            "selection": _digest(selection),
            "calibration": _digest(calibration),
            "heldout": _digest(heldout),
        },
        "split_example_ids": {
            "fit": [row.example_id for row in fit],
            "selection": [row.example_id for row in selection],
            "calibration": [row.example_id for row in calibration],
            "heldout": [row.example_id for row in heldout],
        },
        "label_names": ["negative", "positive"],
        "heldout_usage": "outer evaluation only; not used for training or selection",
    }
    return SST2Data(fit, selection, calibration, heldout, metadata)
