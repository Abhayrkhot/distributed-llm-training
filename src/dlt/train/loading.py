"""Sharded loading of pretrained weights for FSDP2: no rank ever holds the full model.

The model is built with parameters on the meta device (buffers such as RoPE frequencies stay real),
wrapped with ``fully_shard``, and materialized empty on the GPU. Each rank then reads only the rows of
each tensor that it owns, straight from the safetensors files (``get_slice`` reads just those bytes),
and copies them into its local shard, upcasting BF16 checkpoints to the FP32 master parameters.

FSDP2 shards every parameter on dim 0 with ``torch.chunk`` semantics: rank r owns rows
[r*c, min((r+1)*c, n)) with c = ceil(n / world); trailing ranks may own fewer rows or none.

For a 14B model in FP32 this replaces ~59 GB of host RAM per rank (full ``from_pretrained``) with one
tensor slice at a time, and the GPU never holds an unsharded copy.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "LICENSE"]


def snapshot_dir(model: str, revision: str | None) -> Path:
    local = Path(model)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, revision=revision, allow_patterns=PATTERNS))


def weight_map(directory: Path) -> dict[str, Path]:
    """Tensor name -> safetensors file, from the index for sharded checkpoints."""
    index = directory / "model.safetensors.index.json"
    if index.exists():
        return {k: directory / v for k, v in json.loads(index.read_text())["weight_map"].items()}
    from safetensors import safe_open

    single = directory / "model.safetensors"
    with safe_open(single, framework="pt") as f:
        return dict.fromkeys(f.keys(), single)


def build_on_meta(model_id: str, revision: str | None, attn: str):
    """The model with parameters on meta and real buffers (accelerate keeps buffers off meta)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(hf_cfg, attn_implementation=attn, dtype=torch.float32)
    if getattr(hf_cfg, "tie_word_embeddings", False):
        # the meta init leaves lm_head with its own parameter; tie it before fully_shard so FSDP2
        # registers one shared parameter (as it does for a model loaded with from_pretrained)
        model.get_output_embeddings().weight = model.get_input_embeddings().weight
    return model


def owned_rows(n: int, rank: int, world: int) -> tuple[int, int]:
    c = math.ceil(n / world)
    start = min(rank * c, n)
    return start, min(start + c, n)


def materialize_and_load(model, directory: Path, device: torch.device, rank: int, world: int) -> dict:
    """``to_empty`` the sharded meta model on ``device`` and fill each rank's shard from safetensors."""
    from safetensors import safe_open

    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    model.to_empty(device=device)
    with torch.no_grad():
        for name, b in model.named_buffers():
            b.copy_(buffers[name])
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied:  # to_empty re-creates each module's parameter, which breaks the embedding / lm_head tie
        model.get_output_embeddings().weight = model.get_input_embeddings().weight

    files = weight_map(directory)
    params = dict(model.named_parameters())  # tied parameters appear once, under their first name
    missing = [n for n in params if n not in files]
    if missing:
        raise KeyError(f"parameters missing from the checkpoint: {missing[:5]}")
    by_file: dict[Path, list[str]] = {}
    for name in params:
        by_file.setdefault(files[name], []).append(name)

    t0, read = time.perf_counter(), 0
    with torch.no_grad():
        for path, names in sorted(by_file.items()):
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in names:
                    p = params[name]
                    local = p.to_local() if hasattr(p, "to_local") else p
                    if world > 1 and not hasattr(p, "to_local"):
                        raise TypeError(f"{name} is not sharded; call fully_shard before loading")
                    start, end = owned_rows(p.shape[0], rank, world) if world > 1 else (0, p.shape[0])
                    if local.shape[0] != end - start:
                        raise RuntimeError(f"{name}: local shard {tuple(local.shape)} != rows {start}:{end}")
                    if end > start:
                        rows = f.get_slice(name)[start:end]
                        local.copy_(rows)  # host to device and BF16 to FP32 in one copy
                        read += rows.numel() * rows.element_size()
    if tied and model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
        raise RuntimeError("lm_head is no longer tied to the input embedding")
    return {"load_read_bytes": read, "load_read_s": time.perf_counter() - t0}
