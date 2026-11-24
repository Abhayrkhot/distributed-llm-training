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


def read_trace(path: Path) -> list[dict]:
    """The events of a chrome trace (optionally gzipped)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def kernels_from_events(events: list[dict]) -> list[tuple[str, float, float]]:
    """(name, start_us, end_us) for every GPU kernel."""
    return [
        (e["name"], float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in events
        if e.get("cat") == "kernel" and "dur" in e
    ]


def kernels_from_trace(path: Path) -> list[tuple[str, float, float]]:
    """(name, start_us, end_us) for every GPU kernel in a chrome trace (optionally gzipped)."""
    return kernels_from_events(read_trace(path))


LAUNCH_CATS = ("cuda_runtime", "cuda_driver")  # cudaLaunchKernel, and cuLaunchKernelEx (NCCL)


def kernel_launches(events: list[dict]) -> list[tuple[float, float, float | None]]:
    """(start_us, end_us, launch_us) for every GPU kernel: launch_us is the start of the runtime or
    driver API call that launched it (matched on the correlation id), None if the trace has no match."""
    launch = {
        e["args"]["correlation"]: float(e["ts"])
        for e in events
        if e.get("cat") in LAUNCH_CATS and "correlation" in e.get("args", {})
    }
    return [
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]), launch.get(e.get("args", {}).get("correlation")))
        for e in events
        if e.get("cat") == "kernel" and "dur" in e
    ]


GAP_BUCKETS_US = (10.0, 100.0, 1000.0, 10000.0)  # idle gaps <10 us, 10-100 us, 0.1-1 ms, 1-10 ms, >=10 ms
GAP_LABELS = ("lt_10us", "10_100us", "100us_1ms", "1_10ms", "ge_10ms")


def idle_gaps(timeline: list[tuple[float, float, float | None]], n_steps: int = 1) -> dict:
    """Split the GPU's idle time (no kernel on any stream) by what the GPU was waiting for.

    ``timeline`` holds (start_us, end_us, launch_us) per kernel, launch_us being the start of the API call
    that launched it. For each gap, the time from the gap's start to the next kernel's launch call is
    host-bound: nothing was queued, the host had not asked for the next kernel yet. The rest of the gap
    (the call itself, launch latency, a queued kernel waiting on an event or a copy) is not.
    Returns seconds per step, and the gaps by length (count and seconds per step).
    """
    idle = host = 0.0
    counts = [0] * len(GAP_LABELS)
    secs = [0.0] * len(GAP_LABELS)
    busy_end = None
    for s, e, launched in sorted(timeline, key=lambda k: k[0]):
        if busy_end is not None and s > busy_end:
            gap = s - busy_end
            idle += gap
            if launched is not None:
                host += max(0.0, min(launched, s) - busy_end)
            b = sum(gap >= edge for edge in GAP_BUCKETS_US)
            counts[b] += 1
            secs[b] += gap
        busy_end = e if busy_end is None else max(busy_end, e)
    per = 1e6 * n_steps
    return {
        "idle_s": round(idle / per, 5),
        "host_bound_idle_s": round(host / per, 5),
        "idle_gaps_per_step": {k: round(c / n_steps, 1) for k, c in zip(GAP_LABELS, counts, strict=True)},
        "idle_s_by_gap": {k: round(t / per, 5) for k, t in zip(GAP_LABELS, secs, strict=True)},
    }


def host_wait(events: list[dict], n_steps: int = 1) -> dict[str, float]:
    """Per-step floats for a run record: host-bound idle seconds, idle gaps, kernels, and the fraction
    of kernels whose launch call the trace holds (0 means the host-bound figure could not be measured)."""
    timeline = kernel_launches(events)
    g = idle_gaps(timeline, n_steps)
    return {
        "host_bound_idle_s": g["host_bound_idle_s"],
        "idle_gaps_per_step": float(sum(g["idle_gaps_per_step"].values())),
        "kernels_per_step": len(timeline) / n_steps,
        "launch_matched_frac": (sum(x is not None for *_, x in timeline) / len(timeline))
        if timeline
        else 0.0,
    }


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
