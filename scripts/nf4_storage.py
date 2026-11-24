"""Copy a saved NF4 checkpoint with its packed 4-bit weights stored as BF16 instead of uint8.

bitsandbytes packs two 4-bit values per byte; ``bnb_4bit_quant_storage`` only sets the dtype the packed
bytes are kept in. FSDP1 flattens each wrapped unit into one tensor, which needs one dtype, so the
FSDP-QLoRA setup keeps the packed weights in BF16 next to the BF16 norms. The bytes are the same, so a
checkpoint saved with uint8 storage becomes a BF16-storage checkpoint by viewing each packed weight as
BF16 (half as many elements) and changing the config: no 145 GB BF16 read and no re-quantization, and
the 4-bit values are exactly those the replicated runs use.

    python scripts/nf4_storage.py SRC_DIR DST_DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

QS = ".quant_state.bitsandbytes__nf4"
SHARD_BYTES = 4 << 30


def packed_keys(keys: list[str]) -> set[str]:
    """The packed 4-bit weights: every tensor that has a serialized NF4 quant state next to it."""
    ks = set(keys)
    return {k[: -len(QS)] for k in ks if k.endswith(QS) and k[: -len(QS)] in ks}


def convert(src: Path, dst: Path, storage: torch.dtype = torch.bfloat16) -> dict:
    cfg = json.loads((src / "config.json").read_text())
    q = cfg.get("quantization_config", {})
    if q.get("bnb_4bit_quant_type") != "nf4" or q.get("bnb_4bit_quant_storage", "uint8") != "uint8":
        raise ValueError(f"{src} is not an NF4 checkpoint with uint8 storage")
    dst.mkdir(parents=True, exist_ok=False)
    files = sorted(src.glob("*.safetensors"))
    shard: dict[str, torch.Tensor] = {}
    size, n_shard, weight_map, converted = 0, 0, {}, 0
    names: list[str] = []

    def flush() -> None:
        nonlocal shard, size, n_shard
        if not shard:
            return
        name = f"model-{n_shard:05d}.safetensors"
        save_file(shard, str(dst / name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, name))
        names.append(name)
        shard, size, n_shard = {}, 0, n_shard + 1

    for f in files:
        with safe_open(str(f), framework="pt") as h:
            keys = list(h.keys())
            packed = packed_keys(keys)
            for k in keys:
                t = h.get_tensor(k)
                if k in packed:
                    if t.dtype != torch.uint8 or t.numel() % 2:
                        raise ValueError(
                            f"{k}: expected an even number of uint8 bytes, got {t.dtype} {t.shape}"
                        )
                    t = t.contiguous().view(-1).view(storage).view(-1, 1)  # same bytes: [n, 1] -> [n / 2, 1]
                    converted += 1
                shard[k] = t
                size += t.numel() * t.element_size()
                if size >= SHARD_BYTES:
                    flush()
    flush()
    # final names in the usual pattern, with the index transformers reads for sharded checkpoints
    total = len(names)
    rename = {old: f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i, old in enumerate(names)}
    for old, new in rename.items():
        (dst / old).rename(dst / new)
    index = {"metadata": {}, "weight_map": {k: rename[v] for k, v in sorted(weight_map.items())}}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    q["bnb_4bit_quant_storage"] = str(storage).removeprefix("torch.")
    cfg["quantization_config"] = q
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for p in src.iterdir():
        if p.suffix != ".safetensors" and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
    return {"packed_weights": converted, "tensors": len(weight_map), "shards": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    print(json.dumps(convert(args.src, args.dst)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
