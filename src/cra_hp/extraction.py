"""Memory-conscious extraction of per-head attention context vectors."""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExtractionResult:
    representations: np.ndarray
    metadata: dict[str, Any]


def _backbone_layers(model: Any, model_type: str) -> tuple[Any, list[Any]]:
    if model_type == "bert":
        backbone = getattr(model, "bert", model)
        return backbone, list(backbone.encoder.layer)
    if model_type == "gpt2":
        backbone = getattr(model, "transformer", model)
        return backbone, list(backbone.h)
    raise ValueError(f"Unsupported model_type={model_type!r}; expected bert or gpt2")


def _projection_module(layer: Any, model_type: str) -> Any:
    if model_type == "bert":
        return layer.attention.output.dense
    if model_type == "gpt2":
        return layer.attn.c_proj
    raise ValueError(f"Unsupported model_type={model_type!r}")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_head_representations(
    model_id: str,
    texts: list[str],
    *,
    max_length: int = 64,
    batch_size: int = 1,
    device: str | None = None,
    revision: str = "main",
) -> ExtractionResult:
    """Extract masked-mean pooled head context vectors for each input text.

    Hook point: input to BERT attention output dense / GPT-2 attention c_proj.
    At this point, the concatenated per-head context vectors can be reshaped
    to (batch, sequence, heads, head_dim), before the output projection mixes
    them. This is a deliberately chosen comparable interface, not a claim that
    the two architectures have identical attention semantics.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model extraction requires the optional dependencies. Run: pip install -e '.[models]'"
        ) from exc

    if not texts:
        raise ValueError("At least one text example is required")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer for {model_id} has no usable padding token")

    model = AutoModel.from_pretrained(model_id, revision=revision)
    model_type = str(model.config.model_type)
    backbone, layers = _backbone_layers(model, model_type)
    num_heads = int(getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 0)))
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    if num_heads < 1 or hidden_size < 1 or hidden_size % num_heads:
        raise ValueError(
            f"Invalid head configuration for {model_id}: hidden_size={hidden_size}, heads={num_heads}"
        )
    head_dim = hidden_size // num_heads
    expected_layers = len(layers)

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device.startswith("cuda") and not torch.cuda.is_available():
        selected_device = "cpu"
    model.to(selected_device)
    model.eval()

    captured: dict[int, Any] = {}
    handles = []

    for layer_index, layer in enumerate(layers):
        projection = _projection_module(layer, model_type)

        def capture_input(
            _module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            index: int = layer_index,
        ) -> None:
            hidden_states = args[0] if args else kwargs.get("input")
            if hidden_states is None:
                raise RuntimeError(f"Could not capture projection input for layer {index}")
            captured[index] = hidden_states.detach()

        handles.append(projection.register_forward_pre_hook(capture_input, with_kwargs=True))

    representations = np.zeros(
        (expected_layers * num_heads, len(texts), head_dim), dtype=np.float32
    )
    try:
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            attention_mask = encoded["attention_mask"].to(selected_device)
            model_inputs = {key: value.to(selected_device) for key, value in encoded.items()}
            captured.clear()
            with torch.inference_mode():
                _ = model(**model_inputs)

            if len(captured) != expected_layers:
                raise RuntimeError(
                    f"Captured {len(captured)} layers, expected {expected_layers} for {model_id}"
                )

            mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            for layer_index in range(expected_layers):
                context = captured[layer_index]
                batch_tokens = context.shape[0]
                sequence_length = context.shape[1]
                if context.shape[-1] != hidden_size:
                    raise RuntimeError(
                        f"Unexpected projection input shape {tuple(context.shape)} at layer {layer_index}"
                    )
                context = context.reshape(
                    batch_tokens, sequence_length, num_heads, head_dim
                )
                denominator = mask.sum(dim=1).clamp_min(1.0)
                pooled = (context.float() * mask).sum(dim=1) / denominator
                pooled = pooled.cpu().numpy()
                for local_index in range(batch_tokens):
                    global_index = start + local_index
                    for head_index in range(num_heads):
                        flat_head = layer_index * num_heads + head_index
                        representations[flat_head, global_index] = pooled[
                            local_index, head_index
                        ]

            del encoded, model_inputs, attention_mask
            if selected_device.startswith("cuda"):
                torch.cuda.empty_cache()

    finally:
        for handle in handles:
            handle.remove()

    metadata = {
        "model_id": model_id,
        "model_type": model_type,
        "model_revision_requested": revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision_resolved": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "num_layers": expected_layers,
        "num_heads_per_layer": num_heads,
        "total_heads": expected_layers * num_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "max_length": max_length,
        "batch_size": batch_size,
        "device": selected_device,
        "pooling": "masked_mean",
        "hook_point": (
            "bert.encoder.layer[i].attention.output.dense input"
            if model_type == "bert"
            else "gpt2.h[i].attn.c_proj input"
        ),
        "sample_sha256": [_text_hash(text) for text in texts],
    }

    del model, tokenizer, backbone, layers, captured
    gc.collect()
    if selected_device.startswith("cuda"):
        torch.cuda.empty_cache()

    return ExtractionResult(representations=representations, metadata=metadata)
