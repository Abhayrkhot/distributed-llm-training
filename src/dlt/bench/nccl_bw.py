"""NCCL collective bandwidth sweep through torch.distributed (run under torchrun, 2+ GPUs).

Bandwidth follows the nccl-tests conventions so numbers compare with published ones:
  all_reduce      size = buffer bytes;              busbw = algbw * 2(n-1)/n
  all_gather      size = total output bytes;        busbw = algbw * (n-1)/n
  reduce_scatter  size = total input bytes;         busbw = algbw * (n-1)/n
Each point: warmup, then several timed repeats of a batch of calls (CUDA events); the per-repeat time
is the max over ranks (a collective is as slow as its slowest rank) and the reported time is the median.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.distributed as dist

from dlt import dist as dlt_dist
from dlt.results import new_record, write_record

OPS = ("all_reduce", "all_gather", "reduce_scatter")


def bus_factor(op: str, n: int) -> float:
    if op == "all_reduce":
        return 2 * (n - 1) / n
    if op in ("all_gather", "reduce_scatter"):
        return (n - 1) / n
    raise ValueError(op)


def _make_call(op: str, size_bytes: int, n: int, device: torch.device):
    elems = size_bytes // 2  # bf16
    if op == "all_reduce":
        buf = torch.ones(elems, dtype=torch.bfloat16, device=device)
        return lambda: dist.all_reduce(buf)
    if op == "all_gather":
        out = torch.empty(elems, dtype=torch.bfloat16, device=device)
        inp = torch.ones(elems // n, dtype=torch.bfloat16, device=device)
        return lambda: dist.all_gather_into_tensor(out, inp)
    inp = torch.ones(elems, dtype=torch.bfloat16, device=device)
    out = torch.empty(elems // n, dtype=torch.bfloat16, device=device)
    return lambda: dist.reduce_scatter_tensor(out, inp)


def measure(ctx: dlt_dist.DistCtx, op: str, size_bytes: int, iters: int, repeats: int, warmup: int) -> dict:
    call = _make_call(op, size_bytes, ctx.world_size, ctx.device)
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        dist.barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call()
        end.record()
        end.synchronize()
        t = torch.tensor([start.elapsed_time(end) / iters / 1e3], dtype=torch.float64, device=ctx.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(t.item())
    med = statistics.median(times)
    algbw = size_bytes / med / 1e9
    return {
        "op": op,
        "bytes": size_bytes,
        "time_us_median": round(med * 1e6, 2),
        "time_us_min": round(min(times) * 1e6, 2),
        "time_us_max": round(max(times) * 1e6, 2),
        "algbw_gbps": round(algbw, 2),
        "busbw_gbps": round(algbw * bus_factor(op, ctx.world_size), 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-bytes", type=int, default=1 << 16)
    ap.add_argument("--max-bytes", type=int, default=1 << 30)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--target-ms", type=float, default=50.0, help="approximate time per timed batch")
    args = ap.parse_args(argv)

    ctx = dlt_dist.setup("cuda")
    if ctx.world_size < 2:
        raise SystemExit("nccl_bw needs at least 2 ranks")
    sizes = []
    s = args.min_bytes
    while s <= args.max_bytes:
        sizes.append(s)
        s *= args.factor
    rows = []
    for op in OPS:
        for size in sizes:
            probe = measure(ctx, op, size, iters=2, repeats=1, warmup=args.warmup)
            iters = max(2, min(1000, int(args.target_ms * 1e3 / max(probe["time_us_median"], 1.0))))
            row = measure(ctx, op, size, iters=iters, repeats=args.repeats, warmup=2)
            row["iters"] = iters
            rows.append(row)
            if ctx.is_main:
                us, bw = row["time_us_median"], row["busbw_gbps"]
                print(f"{op:15s} {size:>12d} B  {us:>10.1f} us  busbw {bw:>7.1f} GB/s")
    if ctx.is_main:
        peak = {op: max(r["busbw_gbps"] for r in rows if r["op"] == op) for op in OPS}
        rec = new_record(
            "nccl_bw",
            config={**vars(args), "world_size": ctx.world_size, "dtype": "bfloat16", "ops": list(OPS)},
            metrics={"rows": rows},
            summary={"world_size": ctx.world_size, "peak_busbw_gbps": peak},
        )
        print("wrote", write_record(rec))
    dlt_dist.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
