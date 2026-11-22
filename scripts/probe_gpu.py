"""Phase 0 GPU probe: device facts plus three quick microbenchmarks, printed as one JSON object.

Measures, per visible GPU:
  - BF16 GEMM throughput (8192^3, cuBLAS via torch.matmul), the achievable ceiling MFU is judged against
  - device-to-device copy bandwidth (HBM read + write)
and, for the first two GPUs, peer access and one-way copy bandwidth in both directions.

Timing uses CUDA events after warmup; every number is the median of several timed repeats.
"""

from __future__ import annotations

import json
import statistics
import sys

import torch


def _time_ms(fn, iters: int, repeats: int = 5, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def gemm_tflops(dev: int, n: int = 8192) -> float:
    with torch.cuda.device(dev):
        a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
        c = torch.empty(n, n, device=dev, dtype=torch.bfloat16)
        ms = _time_ms(lambda: torch.matmul(a, b, out=c), iters=20)
        return 2 * n**3 / (ms * 1e-3) / 1e12


def copy_gbps(dev: int, nbytes: int = 4 << 30) -> float:
    with torch.cuda.device(dev):
        src = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        dst = torch.empty_like(src)
        ms = _time_ms(lambda: dst.copy_(src), iters=5)
        return 2 * nbytes / (ms * 1e-3) / 1e9  # bytes read + bytes written


def p2p_gbps(src_dev: int, dst_dev: int, nbytes: int = 1 << 30) -> float:
    src = torch.empty(nbytes, dtype=torch.uint8, device=src_dev)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=dst_dev)
    with torch.cuda.device(src_dev):
        ms = _time_ms(lambda: dst.copy_(src, non_blocking=True), iters=5)
    return nbytes / (ms * 1e-3) / 1e9


def main() -> int:
    out: dict = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out["devices"].append(
            {
                "index": i,
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "sm_count": p.multi_processor_count,
                "total_bytes": total,
                "free_bytes_at_start": free,
                "gemm_bf16_8192_tflops": round(gemm_tflops(i), 1),
                "d2d_copy_gbps": round(copy_gbps(i), 1),
            }
        )
        torch.cuda.empty_cache()
    if torch.cuda.device_count() >= 2:
        out["p2p"] = {
            "can_access_0_1": torch.cuda.can_device_access_peer(0, 1),
            "can_access_1_0": torch.cuda.can_device_access_peer(1, 0),
            "copy_0_to_1_gbps": round(p2p_gbps(0, 1), 1),
            "copy_1_to_0_gbps": round(p2p_gbps(1, 0), 1),
        }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
