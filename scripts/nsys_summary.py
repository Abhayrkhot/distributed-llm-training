"""Summarize one Nsight Systems report into a run record (kind ``profile``).

    python scripts/nsys_summary.py NSYS REPORT.nsys-rep --name NAME --rank R [--steps 2]

Exports the report to SQLite with ``nsys export``, reads every GPU kernel, and writes: time per step by
kernel category (GEMM, attention, NCCL, cross-entropy/softmax, optimizer, other), exposed communication
and GPU idle time (dlt.metrics.overlap, the same code as the in-process profiler), memcpy time, the
top kernels, and where the idle time comes from: each gap with no kernel running is split at the moment
the host called the API that launched the next kernel. The part before that call is time the GPU waited
for the host (host-bound); the part after is launch latency or a queued kernel waiting on something else.
The .nsys-rep and .sqlite files stay on /scratch; only the summary is committed.
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

from dlt.metrics.overlap import comm_overlap, idle_gaps  # noqa: E402
from dlt.results import new_record, write_record  # noqa: E402

# First match wins, on the lower-cased demangled name. The demangled name carries the template
# arguments: multi_tensor_apply_kernel is the launcher for fused Adam, gradient-norm and FSDP2 copy
# kernels alike, and only its functor argument tells them apart.
CATEGORIES = [
    ("nccl", ("nccl",)),
    ("attention", ("flash", "fmha", "attention", "attn", "cudnn", "sdpa")),
    ("gemm", ("gemm", "nvjet", "cutlass", "xmma", "cublas", "matmul", "gemv")),
    ("cross_entropy", ("softmax", "nll_loss", "cross_entropy", "logsumexp")),
    ("optimizer", ("fusedadam", "fused_adam", "adamw")),
    ("grad_clip", ("lpnorm", "norm_kernel", "foreach_norm")),
    ("fsdp_copy", ("chunk_cat", "split_with_sizes")),
    ("foreach", ("multi_tensor_apply",)),
    ("elementwise", ("elementwise_kernel",)),
    ("reduce", ("reduce_kernel",)),
]


def category(name: str) -> str:
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other"


def kernels(db: Path) -> tuple[list[tuple[str, str, float, float]], float, list[float | None]]:
    """(short name, demangled name, start_us, end_us) for every kernel, memcpy seconds, and the start of
    the CUDA runtime or driver call that launched each kernel (None where the API trace has no match)."""
    con = sqlite3.connect(db)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        short_col = "shortName" if "shortName" in cols else "demangledName"
        long_col = "demangledName" if "demangledName" in cols else short_col
        rows = con.execute(
            f"SELECT a.value, b.value, k.start, k.end, k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            f"JOIN StringIds a ON k.{short_col} = a.id JOIN StringIds b ON k.{long_col} = b.id"
        ).fetchall()
        try:
            memcpy = con.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0] or 0
        except sqlite3.OperationalError:
            memcpy = 0
        try:  # runtime and driver API calls share one table; correlationId ties a launch to its kernel
            launch = dict(con.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
        except sqlite3.OperationalError:
            launch = {}
    finally:
        con.close()
    # nanoseconds -> microseconds, the unit dlt.metrics.overlap takes
    ks = [(a, b, s / 1e3, e / 1e3) for a, b, s, e, _ in rows]
    launched = [launch[c] / 1e3 if c in launch else None for *_, c in rows]
    return ks, memcpy / 1e9, launched


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
    ks, memcpy_s, launched = kernels(db)
    per_cat: dict[str, float] = defaultdict(float)
    per_kernel: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0])
    for short, long, s, e in ks:
        cat = category(long)
        per_cat[cat] += (e - s) / 1e6
        per_kernel[(short, cat)][0] += 1
        per_kernel[(short, cat)][1] += (e - s) / 1e6
    busy = sum(per_cat.values())
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:25]
    timeline = [(long, s, e) for _, long, s, e in ks]
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
            for k, v in comm_overlap(timeline, n_steps=args.steps).items()
        },
        "kernels_per_step": round(len(ks) / args.steps, 1),
        "kernels_with_launch_call": sum(x is not None for x in launched),
        **idle_gaps([(s, e, x) for (_, _, s, e), x in zip(ks, launched, strict=True)], n_steps=args.steps),
    }
    metrics = {
        "top_kernels": [
            {"name": n[:160], "category": cat, "calls": c, "s_per_step": round(t / args.steps, 5)}
            for (n, cat), (c, t) in top
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
