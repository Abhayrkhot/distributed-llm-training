"""Download pinned model revisions into HF_HOME on a CPU node, so GPU jobs never wait on the network.

python -m dlt.fetch Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import sys
import time

from huggingface_hub import snapshot_download

from dlt.train.config import PINNED

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "LICENSE"]


def main(argv: list[str] | None = None) -> int:
    models = argv if argv is not None else sys.argv[1:]
    for m in models:
        rev = PINNED.get(m)
        if rev is None:
            raise SystemExit(f"{m} has no pinned revision in dlt.train.config.PINNED")
        t0 = time.time()
        path = snapshot_download(m, revision=rev, allow_patterns=PATTERNS)
        print(f"{m}@{rev[:10]} -> {path} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
