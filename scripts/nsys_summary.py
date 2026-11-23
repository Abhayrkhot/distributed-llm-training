"""Summarize one Nsight Systems report into a run record (kind ``profile``).

    python scripts/nsys_summary.py NSYS REPORT.nsys-rep --name NAME --rank R [--steps 2]

Exports the report to SQLite with ``nsys export``, reads every GPU kernel, and writes: time per step by
kernel category (GEMM, attention, NCCL, cross-entropy/softmax, optimizer, other), exposed communication
and GPU idle time (dlt.metrics.overlap, the same code as the in-process profiler), memcpy time, and the
top kernels. The .nsys-rep and .sqlite files stay on /scratch; only the summary is committed.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dlt.metrics.overlap import comm_overlap  # noqa: E402
from dlt.results import new_record, write_record  # noqa: E402

CATEGORIES = [  # first match wins, on the lower-cased kernel name
    ("nccl", ("nccl",)),
    ("attention", ("flash", "fmha", "attention", "attn", "cudnn")),
    ("gemm", ("gemm", "nvjet", "cutlass", "xmma", "cublas", "matmul")),
    ("cross_entropy", ("softmax", "nll_loss", "cross_entropy", "logsumexp")),
    ("optimizer", ("multi_tensor", "fused_adam", "adam")),
]


def category(name: str) -> str:
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def kernels(db: Path) -> list[tuple[str, float, float]]:
    con = sqlite3.connect(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        name_col = "shortName" if "shortName" in cols else "demangledName"
        rows = con.execute(
            f"SELECT s.value, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            f"JOIN StringIds s ON k.{name_col} = s.id"
        ).fetchall()
        try:
            memcpy = con.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0] or 0
        except sqlite3.OperationalError:
            memcpy = 0
    finally:
        con.close()
    # nanoseconds -> microseconds, the unit dlt.metrics.overlap takes
    return [(n, s / 1e3, e / 1e3) for n, s, e in rows], memcpy / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("nsys")
    ap.add_argument("report", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2, help="profiled steps in the capture range")
    args = ap.parse_args()
    if not args.report.exists():
        print(f"no report at {args.report}")
        return 1
    db = args.report.with_suffix(".sqlite")
    subprocess.run(
        [
            args.nsys,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            str(db),
            str(args.report),
        ],
        check=True,
    )
    ks, memcpy_s = kernels(db)
    per_cat: dict[str, float] = defaultdict(float)
    per_kernel: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for name, s, e in ks:
        per_cat[category(name)] += (e - s) / 1e6
        per_kernel[name][0] += 1
        per_kernel[name][1] += (e - s) / 1e6
    busy = sum(per_cat.values())
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:20]
    summary = {
        "name": args.name,
        "rank": args.rank,
        "steps": args.steps,
        "kernels": len(ks),
        "kernel_s_per_step": {k: round(v / args.steps, 5) for k, v in sorted(per_cat.items())},
        "kernel_share": {k: round(v / busy, 4) for k, v in sorted(per_cat.items())} if busy else {},
        "memcpy_s_per_step": round(memcpy_s / args.steps, 5),
        **{
            k: (round(v, 5) if v is not None else None)
            for k, v in comm_overlap(ks, n_steps=args.steps).items()
        },
    }
    metrics = {
        "top_kernels": [
            {"name": n[:160], "category": category(n), "calls": c, "s_per_step": round(t / args.steps, 5)}
            for n, (c, t) in top
        ]
    }
    rec = new_record(
        "profile",
        config={"name": args.name, "rank": args.rank, "report": args.report.name},
        metrics=metrics,
        summary=summary,
    )
    rec["run_id"] = f"profile-{args.name}-rank{args.rank}"
    print("wrote", write_record(rec))
    print(json_summary(summary))
    return 0


def json_summary(s: dict) -> str:
    import json

    return json.dumps(s, indent=1)


if __name__ == "__main__":
    raise SystemExit(main())
