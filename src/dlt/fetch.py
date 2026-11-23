"""Download pinned model revisions into HF_HOME on a CPU node, so GPU jobs never wait on the network, and
write a checksum manifest for each (a ``model`` run record).

    python -m dlt.fetch [--verify] Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-72B

With ``--verify`` every safetensors file is hashed locally (in parallel) and compared with the sha256
the Hub publishes for it; the manifest records file names, sizes, published and local hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from dlt.results import new_record, write_record
from dlt.train.config import PINNED

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "LICENSE"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def fetch(model: str, verify: bool, workers: int) -> dict:
    rev = PINNED.get(model)
    if rev is None:
        raise SystemExit(f"{model} has no pinned revision in dlt.train.config.PINNED")
    t0 = time.time()
    path = Path(snapshot_download(model, revision=rev, allow_patterns=PATTERNS))
    download_s = time.time() - t0
    info = HfApi().model_info(model, revision=rev, files_metadata=True)
    published = {s.rfilename: (s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    files = []
    local = sorted(p for p in path.rglob("*") if p.is_file())
    hashes: dict[str, str] = {}
    t1 = time.time()
    if verify:
        big = [p for p in local if p.suffix == ".safetensors"]
        with ProcessPoolExecutor(workers) as ex:
            hashes = dict(
                zip((str(p.relative_to(path)) for p in big), ex.map(sha256, map(str, big)), strict=True)
            )
    verify_s = time.time() - t1
    for p in local:
        rel = str(p.relative_to(path))
        size, pub = published.get(rel, (None, None))
        files.append(
            {
                "file": rel,
                "bytes": p.stat().st_size,
                "published_bytes": size,
                "sha256": pub,
                "local_sha256": hashes.get(rel),
            }
        )
    mismatched = [f["file"] for f in files if f["local_sha256"] and f["local_sha256"] != f["sha256"]]
    summary = {
        "model": model,
        "revision": rev,
        "files": len(files),
        "total_gib": round(sum(f["bytes"] for f in files) / 2**30, 2),
        "download_s": round(download_s, 1),
        "verified_files": len(hashes),
        "verify_s": round(verify_s, 1),
        "mismatched": mismatched,
    }
    rec = new_record(
        "model",
        config={"model": model, "revision": rev, "patterns": PATTERNS},
        metrics={"files": files},
        summary=summary,
    )
    rec["run_id"] = f"model-{model.replace('/', '__')}-{rev[:10]}"
    write_record(rec)
    print(f"{model}@{rev[:10]} -> {path}: {summary}", flush=True)
    if mismatched:
        raise SystemExit(f"checksum mismatch: {mismatched}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    for m in args.models:
        fetch(m, args.verify, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
