"""Exposed communication from a GPU kernel timeline (PyTorch Profiler chrome trace).

Exposed communication is time when an NCCL kernel is running and no compute kernel is: the part of the
collective the step waits for. NCCL kernels spin while waiting for peers, so kernel time is an upper
bound on wire time; exposed time is still what the step pays.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

Interval = tuple[float, float]


def merge(intervals: list[Interval]) -> list[Interval]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def total(merged: list[Interval]) -> float:
    return sum(e - s for s, e in merged)


def minus(a: list[Interval], b: list[Interval]) -> float:
    """Length of merged(a) not covered by merged(b)."""
    a, b = merge(a), merge(b)
    covered, j = 0.0, 0
    for s, e in a:
        while j < len(b) and b[j][1] <= s:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            covered += min(e, b[k][1]) - max(s, b[k][0])
            k += 1
    return total(a) - covered


def is_comm(name: str) -> bool:
    return "nccl" in name.lower()


def kernels_from_trace(path: Path) -> list[tuple[str, float, float]]:
    """(name, start_us, end_us) for every GPU kernel in a chrome trace (optionally gzipped)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        events = json.load(f)["traceEvents"]
    return [
        (e["name"], float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in events
        if e.get("cat") == "kernel" and "dur" in e
    ]


def comm_overlap(kernels: list[tuple[str, float, float]], n_steps: int = 1) -> dict[str, float]:
    """Per-step seconds of compute, communication and exposed communication (inputs in microseconds)."""
    comm = merge([(s, e) for n, s, e in kernels if is_comm(n)])
    comp = merge([(s, e) for n, s, e in kernels if not is_comm(n)])
    busy = merge([(s, e) for _, s, e in kernels])
    exposed = minus(comm, comp)
    c = total(comm)
    span = (busy[-1][1] - busy[0][0]) if busy else 0.0
    return {
        "compute_s": total(comp) / 1e6 / n_steps,
        "comm_s": c / 1e6 / n_steps,
        "exposed_comm_s": exposed / 1e6 / n_steps,
        "overlapped_frac": (1 - exposed / c) if c else None,
        # no kernel running at all between the first and last kernel of the window: CPU-side gaps
        # (launch overhead, host-to-device copies from pageable memory, the per-step loss sync)
        "gpu_idle_s": (span - total(busy)) / 1e6 / n_steps,
    }
