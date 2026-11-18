"""Fine-tune BERT/GPT-2 on SST-2 and measure head-level causal responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cra_hp.matching import fixed_cardinality_hungarian, partial_hungarian, random_matching
from cra_hp.similarity import cka_similarity_matrix
from cra_hp.task_data import SST2Data, TextExample, load_sst2_pilot


def _seed_everything(seed: int, deterministic: bool, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(
    examples: list[TextExample], batch_size: int, *, order: np.ndarray | None = None
):
    indices = np.arange(len(examples)) if order is None else order
    for start in range(0, len(indices), batch_size):
        yield [examples[int(index)] for index in indices[start : start + batch_size]]


def _encode_batch(
    tokenizer: Any,
    examples: list[TextExample],
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(
        [row.text for row in examples],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return (
        {key: value.to(device) for key, value in encoded.items()},
        torch.tensor([row.label for row in examples], dtype=torch.long, device=device),
    )


def _evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[TextExample],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for batch in _batches(examples, batch_size):
            inputs, labels = _encode_batch(tokenizer, batch, max_length, device)
            logits = model(**inputs).logits
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == labels).sum())
            count += len(batch)
    if not count:
        raise ValueError("Evaluation split is empty")
    return {"loss": total_loss / count, "accuracy": correct / count, "count": count}


def _train_model(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data: SST2Data,
    run_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    torch.manual_seed(seed)
    model_id = str(model_cfg["id"])
    revision = str(model_cfg["revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=str(model_cfg["tokenizer_revision"]),
    )
    if model_cfg["architecture"] == "gpt2":
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=int(train_cfg["num_labels"]),
    )
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer for {model_id} has no padding token")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    batch_size = int(train_cfg["batch_size"])
    max_length = int(train_cfg["max_length"])
    epochs = int(train_cfg["epochs"])
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    checkpoint_dir = run_dir / "checkpoints" / model_cfg["architecture"]
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(data.fit))
        epoch_loss = 0.0
        seen = 0
        for batch in _batches(data.fit, batch_size, order=order):
            inputs, labels = _encode_batch(tokenizer, batch, max_length, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs, labels=labels)
            output.loss.backward()
            clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
            optimizer.step()
            epoch_loss += float(output.loss.detach()) * len(batch)
            seen += len(batch)

        selection = _evaluate(
            model,
            tokenizer,
            data.selection,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        record = {
            "epoch": epoch + 1,
            "fit_loss": epoch_loss / max(seen, 1),
            "selection_loss": selection["loss"],
            "selection_accuracy": selection["accuracy"],
        }
        history.append(record)
        if selection["loss"] < best_loss:
            best_loss = selection["loss"]
            best_epoch = epoch + 1
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    outer_eval = _evaluate(
        model,
        tokenizer,
        data.heldout,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    metadata = {
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": str(model_cfg["tokenizer_revision"]),
        "architecture": str(model_cfg["architecture"]),
        "readout": str(model_cfg["readout"]),
        "num_layers": int(
            model.config.num_hidden_layers
            if hasattr(model.config, "num_hidden_layers")
            else model.config.n_layer
        ),
        "num_heads_per_layer": int(
            model.config.num_attention_heads
            if hasattr(model.config, "num_attention_heads")
            else model.config.n_head
        ),
        "training_examples": len(data.fit),
        "selection_examples": len(data.selection),
        "best_epoch": best_epoch,
        "training_history": history,
        "outer_validation": outer_eval,
        "checkpoint_path": str(checkpoint_dir),
    }
    return model, tokenizer, metadata


def _install_gates(
    model: torch.nn.Module,
    architecture: str,
    gates: torch.Tensor,
) -> list[Any]:
    if architecture == "bert":
        backbone = getattr(model, "bert", model)
        layers = list(backbone.encoder.layer)
        projections = [layer.attention.output.dense for layer in layers]
        heads = int(model.config.num_attention_heads)
        hidden = int(model.config.hidden_size)
    elif architecture == "gpt2":
        backbone = getattr(model, "transformer", model)
        layers = list(backbone.h)
        projections = [layer.attn.c_proj for layer in layers]
        heads = int(model.config.n_head)
        hidden = int(model.config.n_embd)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    head_dim = hidden // heads
    if len(layers) * heads != gates.numel():
        raise ValueError("Gate count does not match the model's attention heads")

    handles = []
    for layer_index, projection in enumerate(projections):
        start = layer_index * heads
        layer_gates = gates[start : start + heads]

        def apply_gate(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            current_gates: torch.Tensor = layer_gates,
            dim: int = head_dim,
        ):
            value = args[0] if args else kwargs.get("input")
            if value is None:
                raise RuntimeError("Attention output projection input was not provided")
            split = value.reshape(*value.shape[:-1], current_gates.numel(), dim)
            scaled = split * current_gates.to(value.device, value.dtype).view(1, 1, -1, 1)
            replacement = scaled.reshape_as(value)
            if args:
                return (replacement, *args[1:]), kwargs
            replaced_kwargs = dict(kwargs)
            replaced_kwargs["input"] = replacement
            return args, replaced_kwargs

        handles.append(
            projection.register_forward_pre_hook(apply_gate, with_kwargs=True)
        )
    return handles


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
    return true_logits - other_logits


def _exact_ablation(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[TextExample],
    architecture: str,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_heads = int(
        model.config.num_attention_heads if architecture == "bert" else model.config.n_head
    )
    n_layers = int(
        model.config.num_hidden_layers if architecture == "bert" else model.config.n_layer
    )
    total_heads = n_heads * n_layers
    model.eval()
    gates = torch.ones(total_heads, device=device)
    handles = _install_gates(model, architecture, gates)
    per_example: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for batch in _batches(examples, batch_size):
                inputs, labels = _encode_batch(tokenizer, batch, max_length, device)
                gates.fill_(1.0)
                base_logits = model(**inputs).logits
                base_losses = F.cross_entropy(base_logits, labels, reduction="none")
                base_margins = _margin(base_logits, labels)
                base_predictions = base_logits.argmax(dim=-1)
                for head_id in range(total_heads):
                    gates.fill_(1.0)
                    gates[head_id] = 0.0
                    ablated_logits = model(**inputs).logits
                    ablated_losses = F.cross_entropy(
                        ablated_logits, labels, reduction="none"
                    )
                    ablated_margins = _margin(ablated_logits, labels)
                    ablated_predictions = ablated_logits.argmax(dim=-1)
                    for row_index, example in enumerate(batch):
                        per_example.append({
                            "example_id": example.example_id,
                            "label": example.label,
                            "head_id": head_id,
                            "layer": head_id // n_heads,
                            "head_in_layer": head_id % n_heads,
                            "base_loss": float(base_losses[row_index]),
                            "ablated_loss": float(ablated_losses[row_index]),
                            "loss_delta": float(
                                ablated_losses[row_index] - base_losses[row_index]
                            ),
                            "base_true_class_margin": float(base_margins[row_index]),
                            "ablated_true_class_margin": float(ablated_margins[row_index]),
                            "true_class_margin_delta": float(
                                base_margins[row_index] - ablated_margins[row_index]
                            ),
                            "prediction_flip": bool(
                                ablated_predictions[row_index]
                                != base_predictions[row_index]
                            ),
                        })
    finally:
        for handle in handles:
            handle.remove()

    frame = pd.DataFrame(per_example)
    summary = frame.groupby(["head_id", "layer", "head_in_layer"], as_index=False).agg(
        mean_loss_delta=("loss_delta", "mean"),
        mean_true_class_margin_delta=("true_class_margin_delta", "mean"),
        prediction_flip_rate=("prediction_flip", "mean"),
        calibration_examples=("example_id", "nunique"),
    )
    return frame, summary


def _gradient_proxy(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[TextExample],
    architecture: str,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> pd.DataFrame:
    n_heads = int(
        model.config.num_attention_heads if architecture == "bert" else model.config.n_head
    )
    n_layers = int(
        model.config.num_hidden_layers if architecture == "bert" else model.config.n_layer
    )
    gates = torch.nn.Parameter(torch.ones(n_layers * n_heads, device=device))
    handles = _install_gates(model, architecture, gates)
    gradient_sum = torch.zeros_like(gates)
    count = 0
    model.eval()
    try:
        for batch in _batches(examples, batch_size):
            inputs, labels = _encode_batch(tokenizer, batch, max_length, device)
            logits = model(**inputs).logits
            per_example_losses = F.cross_entropy(logits, labels, reduction="none")
            gradient = torch.autograd.grad(per_example_losses.sum(), gates)[0]
            gradient_sum += gradient.detach()
            count += len(batch)
    finally:
        for handle in handles:
            handle.remove()
    mean_gradient = gradient_sum / max(count, 1)
    rows = []
    for head_id, value in enumerate(mean_gradient.cpu().tolist()):
        rows.append({
            "head_id": head_id,
            "layer": head_id // n_heads,
            "head_in_layer": head_id % n_heads,
            "mean_gate_loss_gradient": value,
            "first_order_loss_increase_if_removed": -value,
            "absolute_gradient_proxy": abs(value),
            "examples": count,
        })
    return pd.DataFrame(rows)


def _signature_tensor(
    effects: pd.DataFrame,
    calibration_ids: list[str],
    head_count: int,
) -> np.ndarray:
    """Build per-head, per-example [loss, margin] causal response signatures."""
    signatures = np.empty((head_count, len(calibration_ids), 2), dtype=np.float64)
    for head_id in range(head_count):
        head_rows = effects.loc[effects["head_id"] == head_id].set_index("example_id")
        ordered = head_rows.reindex(calibration_ids)
        if ordered[["loss_delta", "true_class_margin_delta"]].isna().any().any():
            raise ValueError(f"Incomplete calibration responses for head {head_id}")
        signatures[head_id] = ordered[
            ["loss_delta", "true_class_margin_delta"]
        ].to_numpy(dtype=np.float64)
    return signatures


def _match_rows(method: str, matches: list[Any], heads_per_layer: int) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "source_head": match.source_head,
            "source_layer": match.source_head // heads_per_layer,
            "source_head_in_layer": match.source_head % heads_per_layer,
            "target_head": match.target_head,
            "target_layer": match.target_head // heads_per_layer,
            "target_head_in_layer": match.target_head % heads_per_layer,
            "score": match.score,
        }
        for match in matches
    ]


def _causal_alignment(
    source_effects: pd.DataFrame,
    target_effects: pd.DataFrame,
    calibration: list[TextExample],
    *,
    heads_per_layer: int,
    seed: int,
    threshold: float,
    permutation_replicates: int,
    random_replicates: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    calibration_ids = [row.example_id for row in calibration]
    source_heads = int(source_effects["head_id"].max()) + 1
    target_heads = int(target_effects["head_id"].max()) + 1
    source = _signature_tensor(source_effects, calibration_ids, source_heads)
    target = _signature_tensor(target_effects, calibration_ids, target_heads)
    similarity = cka_similarity_matrix(source, target)
    global_matches = partial_hungarian(similarity, threshold=threshold)
    cardinality = len(global_matches)
    global_fixed = fixed_cardinality_hungarian(similarity, cardinality)

    same_index_mask = np.zeros_like(similarity, dtype=bool)
    np.fill_diagonal(same_index_mask, True)
    source_layers = np.arange(source_heads) // heads_per_layer
    target_layers = np.arange(target_heads) // heads_per_layer
    same_layer_mask = source_layers[:, None] == target_layers[None, :]
    same_index = fixed_cardinality_hungarian(
        similarity, cardinality, allowed_mask=same_index_mask
    )
    same_layer = fixed_cardinality_hungarian(
        similarity, cardinality, allowed_mask=same_layer_mask
    )

    random_rng = np.random.default_rng(seed + 11)
    random_scores: list[float] = []
    random_rows: list[dict[str, Any]] = []
    for replicate in range(random_replicates):
        matches = random_matching(
            similarity,
            cardinality,
            seed=int(random_rng.integers(0, 2**32 - 1)),
        )
        mean_score = float(np.mean([item.score for item in matches])) if matches else None
        random_rows.append({
            "replicate": replicate,
            "matched_count": len(matches),
            "mean_score": mean_score,
        })
        if mean_score is not None:
            random_scores.append(mean_score)

    permutation_rng = np.random.default_rng(seed + 12)
    calibration_labels = np.asarray([row.label for row in calibration])
    null_rows: list[dict[str, Any]] = []
    observed = float(np.mean([item.score for item in global_fixed])) if global_fixed else None
    for replicate in range(permutation_replicates):
        indices = np.arange(source.shape[1])
        for label in np.unique(calibration_labels):
            positions = np.flatnonzero(calibration_labels == label)
            indices[positions] = permutation_rng.permutation(positions)
        shuffled = cka_similarity_matrix(source, target[:, indices, :])
        null_matches = fixed_cardinality_hungarian(shuffled, cardinality)
        null_mean = (
            float(np.mean([item.score for item in null_matches]))
            if null_matches else None
        )
        null_rows.append({
            "replicate": replicate,
            "matched_count": cardinality,
            "mean_max_weight_score": null_mean,
        })
    null_values = np.asarray(
        [row["mean_max_weight_score"] for row in null_rows
         if row["mean_max_weight_score"] is not None], dtype=np.float64
    )
    empirical_p = (
        float((1 + np.sum(null_values >= observed)) / (1 + len(null_values)))
        if observed is not None and len(null_values)
        else None
    )
    random_values = np.asarray(random_scores, dtype=np.float64)
    summary = {
        "signature_features": ["loss_delta", "true_class_margin_delta"],
        "statistic": "mean score of maximum-weight fixed-cardinality assignment",
        "permutation_scheme": "shuffle paired example correspondence within sentiment-label strata",
        "matched_count": cardinality,
        "observed_global_mean_cka": observed,
        "global_partial_mean_cka": float(np.mean([item.score for item in global_matches]))
        if global_matches else None,
        "same_index_fixed_k_mean_cka": float(np.mean([item.score for item in same_index]))
        if same_index else None,
        "same_layer_fixed_k_mean_cka": float(np.mean([item.score for item in same_layer]))
        if same_layer else None,
        "random_replicates": random_replicates,
        "random_mean_cka": float(random_values.mean()) if len(random_values) else None,
        "random_std_cka": float(random_values.std(ddof=1)) if len(random_values) > 1 else None,
        "permutation_replicates": permutation_replicates,
        "permutation_null_mean": float(null_values.mean()) if len(null_values) else None,
        "permutation_null_std": float(null_values.std(ddof=1))
        if len(null_values) > 1 else None,
        "permutation_empirical_p_value": empirical_p,
    }
    match_frame = pd.DataFrame(
        _match_rows("global_partial", global_matches, heads_per_layer)
        + _match_rows("global_fixed_k", global_fixed, heads_per_layer)
        + _match_rows("same_index_fixed_k", same_index, heads_per_layer)
        + _match_rows("same_layer_fixed_k", same_layer, heads_per_layer)
    )
    return (
        similarity,
        match_frame,
        summary,
        pd.DataFrame(null_rows),
        pd.DataFrame(random_rows),
    )


def run(config_path: str) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repro = config["reproducibility"]
    seed = int(repro["seed"])
    _seed_everything(
        seed,
        bool(repro.get("deterministic_algorithms", False)),
        int(repro.get("num_threads", 4)),
    )
    data = load_sst2_pilot(config["dataset"])
    config_bytes = Path(config_path).read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_id = f"stage3_bert-gpt2_sst2_{seed}_{config_hash}_{timestamp}"
    run_dir = Path(config["outputs"]["root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_bytes(config_bytes)
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(data.metadata, indent=2), encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_summaries: dict[str, Any] = {}
    calibration_effects: dict[str, pd.DataFrame] = {}
    for name in ("encoder", "decoder"):
        model_cfg = config["models"][name]
        model, tokenizer, training = _train_model(
            model_cfg,
            config["models"],
            data,
            run_dir,
            device,
            seed + (0 if name == "encoder" else 1),
        )
        calibration_batch = min(int(config["models"]["batch_size"]), 4)
        per_example, effects = _exact_ablation(
            model,
            tokenizer,
            data.calibration,
            str(model_cfg["architecture"]),
            batch_size=calibration_batch,
            max_length=int(config["models"]["max_length"]),
            device=device,
        )
        gradients = _gradient_proxy(
            model,
            tokenizer,
            data.calibration,
            str(model_cfg["architecture"]),
            batch_size=calibration_batch,
            max_length=int(config["models"]["max_length"]),
            device=device,
        )
        per_example.to_csv(run_dir / f"{name}_calibration_effects.csv", index=False)
        effects.to_csv(run_dir / f"{name}_head_effects.csv", index=False)
        gradients.to_csv(run_dir / f"{name}_gradient_proxy.csv", index=False)
        model_summaries[name] = {
            **training,
            "exact_ablation_heads": len(effects),
            "calibration_examples": len(data.calibration),
            "gradient_proxy_heads": len(gradients),
        }
        calibration_effects[name] = per_example
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    source_heads_per_layer = int(model_summaries["encoder"]["num_heads_per_layer"])
    if source_heads_per_layer != int(model_summaries["decoder"]["num_heads_per_layer"]):
        raise ValueError("Fixed same-index/layer baselines require equal heads per layer")
    similarity, match_frame, alignment_summary, null_frame, random_frame = _causal_alignment(
        calibration_effects["encoder"],
        calibration_effects["decoder"],
        data.calibration,
        heads_per_layer=source_heads_per_layer,
        seed=seed,
        threshold=float(config["causal_intervention"]["unmatched_threshold"]),
        permutation_replicates=int(
            config["causal_intervention"]["permutation_replicates"]
        ),
        random_replicates=int(
            config["causal_intervention"]["random_assignment_replicates"]
        ),
    )
    pd.DataFrame(similarity).to_csv(
        run_dir / "causal_signature_cka.csv", index_label="source_head"
    )
    match_frame.to_csv(run_dir / "causal_matches.csv", index=False)
    null_frame.to_csv(run_dir / "causal_permutation_null.csv", index=False)
    random_frame.to_csv(run_dir / "causal_random_baseline.csv", index=False)

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "seed": seed,
        "dataset": data.metadata,
        "models": model_summaries,
        "causal_alignment": alignment_summary,
        "interpretation_warning": (
            "Pilot-sized SST-2 causal effects are exploratory. The labeled official "
            "validation split is an outer evaluation set and is not used to form "
            "causal signatures or select checkpoints. No test-split result is available."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage3_pilot.yaml")
    args = parser.parse_args()
    output = run(args.config)
    print(f"Stage 3 causal pilot artifacts written to: {output}")


if __name__ == "__main__":
    main()
