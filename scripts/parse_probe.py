"""Distill Phase 0 probe output (pulled from Explorer) into manifests/h200.json plus scrubbed raw files.

Usage:
    python scripts/parse_probe.py results/explorer/probe/h200-<job> [...] \
        [--cpu results/explorer/probe/cpu-<job>] [--scrub USERNAME]

Raw probe dirs stay out of git (results/explorer/ is ignored). This keeps only small, scrubbed text:
the Explorer username, GPU UUIDs and serial numbers are removed before anything is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_KEEP = [
    "meta.txt",
    "gpus.csv",
    "topo.txt",
    "nvlink.txt",
    "clocks.txt",
    "lscpu.txt",
    "numa.txt",
    "free.txt",
    "tools.txt",
    "torch_probe.json",
    "torch_probe.err",
    "nsys_test.log",
    "ncu_test.log",
]
CPU_KEEP = ["meta.txt", "quota_scratch.txt", "quota_home.txt", "hf_speed.txt"]


EXTRA_SCRUB: list[str] = []


def scrub(text: str) -> str:
    for word in EXTRA_SCRUB:
        text = text.replace(word, "$USER")
    text = re.sub(r"/(scratch|home|work|projects)/[A-Za-z0-9._-]+", r"/\1/$USER", text)
    text = re.sub(r"GPU-[0-9a-f]{8}-[0-9a-f-]{27}", "GPU-<uuid>", text)
    text = re.sub(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<internal-ip>", text)
    text = re.sub(r"(Serial Number\s*:\s*)\S+", r"\1<serial>", text)
    text = re.sub(r"(UserId|User|GroupId|Account)=\S+", r"\1=<redacted>", text)
    return text


def read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def kv(text: str) -> dict[str, str]:
    """Parse key=value lines; a line with one '=' keeps spaces in its value."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") == 1:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            for part in line.split():
                if "=" in part:
                    k, v = part.split("=", 1)
                    out[k] = v
    return out


def parse_gpus_csv(text: str) -> list[dict[str, str]]:
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    if len(rows) < 2:
        return []
    header = [h.strip() for h in rows[0]]
    return [dict(zip(header, (c.strip() for c in r), strict=False)) for r in rows[1:]]


def parse_topo(text: str) -> dict:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split()
    rows = {}
    for ln in lines[1:]:
        cells = ln.split()
        if cells and re.fullmatch(r"GPU\d+", cells[0]):
            rows[cells[0]] = dict(zip(header, cells[1:], strict=False))
    out: dict = {"matrix": rows}
    if "GPU0" in rows and "GPU1" in rows["GPU0"]:
        out["gpu0_gpu1_link"] = rows["GPU0"]["GPU1"]
    if "GPU0" in rows:
        out["gpu0_cpu_affinity"] = rows["GPU0"].get("CPU")
        out["gpu0_numa_affinity"] = rows["GPU0"].get("NUMA")
    return out


def parse_nvlink(text: str) -> dict:
    per_gpu: dict[str, list[float]] = {}
    current = None
    for ln in text.splitlines():
        m = re.match(r"GPU (\d+):", ln)
        if m:
            current = f"GPU{m.group(1)}"
            per_gpu[current] = []
            continue
        m = re.search(r"Link \d+:\s*([\d.]+)\s*GB/s", ln)
        if m and current:
            per_gpu[current].append(float(m.group(1)))
    return {
        g: {"active_links": len(v), "per_link_gbps": v[0] if v else None, "total_gbps": round(sum(v), 1)}
        for g, v in per_gpu.items()
    }


def parse_lscpu(text: str) -> dict:
    want = {
        "Model name": "model",
        "Socket(s)": "sockets",
        "Core(s) per socket": "cores_per_socket",
        "Thread(s) per core": "threads_per_core",
        "NUMA node(s)": "numa_nodes",
    }
    out = {}
    for ln in text.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            if k.strip() in want:
                out[want[k.strip()]] = v.strip()
    return out


def parse_tools(text: str) -> dict:
    out: dict[str, dict] = {}
    current = None
    for ln in text.splitlines():
        m = re.match(r"== module (\S+)", ln)
        if m:
            current = m.group(1)
            out[current] = {"nsys": None, "ncu": None, "lines": []}
            continue
        if current is None:
            continue
        out[current]["lines"].append(ln)
        if ln.endswith("/nsys"):
            out[current]["nsys"] = ln
        if ln.endswith("/ncu"):
            out[current]["ncu"] = ln
    for v in out.values():
        v["versions"] = [ln for ln in v.pop("lines") if "version" in ln.lower() or "Version" in ln]
    return out


def parse_h200(d: Path) -> dict:
    meta = kv(read(d / "meta.txt"))
    torch_probe = {}
    if (d / "torch_probe.json").exists():
        try:
            torch_probe = json.loads(read(d / "torch_probe.json"))
        except json.JSONDecodeError:
            torch_probe = {"error": "unparseable"}
    smi = read(d / "smi.txt")
    cuda_driver = re.search(r"CUDA Version\s*:\s*(\S+)", smi)
    return {
        "probe": d.name,
        "date": meta.get("date"),
        "node": meta.get("host"),
        "partition": meta.get("partition"),
        "git_sha": meta.get("git_sha"),
        "os": meta.get("os"),
        "kernel": meta.get("kernel"),
        "cuda_driver_api": cuda_driver.group(1) if cuda_driver else None,
        "gpus": parse_gpus_csv(read(d / "gpus.csv")),
        "topology": parse_topo(read(d / "topo.txt")),
        "nvlink": parse_nvlink(read(d / "nvlink.txt")),
        "cpu": parse_lscpu(read(d / "lscpu.txt")),
        "tools": parse_tools(read(d / "tools.txt")),
        "profilers": {
            "nsys_exit": meta.get("nsys_exit"),
            "nsys_report": meta.get("report"),
            "ncu_exit": meta.get("ncu_exit"),
            "ncu_perm_error_lines": meta.get("perm_error"),
        },
        "torch_probe": torch_probe,
    }


def parse_cpu(d: Path) -> dict:
    speeds = []
    for ln in read(d / "hf_speed.txt").splitlines():
        f = dict(p.split("=", 1) for p in ln.split() if "=" in p)
        if f.get("http") in ("200", "206") and float(f.get("bytes", 0)) > 0:
            speeds.append(float(f["bytes_per_s"]) / 1e6)
    return {
        "probe": d.name,
        "node": kv(read(d / "meta.txt")).get("host"),
        "hf_download_mb_per_s": {
            "samples": [round(s, 1) for s in speeds],
            "median": round(statistics.median(speeds), 1) if speeds else None,
        },
        "quota_scratch_raw": scrub(read(d / "quota_scratch.txt")).strip(),
        "quota_home_raw": scrub(read(d / "quota_home.txt")).strip(),
    }


def copy_scrubbed(src: Path, names: list[str], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = src / n
        if p.exists() and p.stat().st_size < 512_000:
            (dst / n).write_text(scrub(read(p)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("h200", nargs="+", type=Path)
    ap.add_argument("--cpu", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "manifests" / "h200.json")
    ap.add_argument("--scrub", action="append", default=[], help="literal string to replace with $USER")
    args = ap.parse_args()
    EXTRA_SCRUB.extend(args.scrub)

    probes = [parse_h200(d) for d in args.h200]
    names = {g.get("name") for p in probes for g in p["gpus"]}
    drivers = {g.get("driver_version") for p in probes for g in p["gpus"]}
    pair = next(
        (p["topology"].get("gpu0_gpu1_link") for p in probes if "gpu0_gpu1_link" in p["topology"]), None
    )
    gemm = [d["gemm_bf16_8192_tflops"] for p in probes for d in p["torch_probe"].get("devices", [])]
    hbm = [d["d2d_copy_gbps"] for p in probes for d in p["torch_probe"].get("devices", [])]
    p2p = [p["torch_probe"]["p2p"] for p in probes if "p2p" in p["torch_probe"]]
    manifest = {
        "summary": {
            "gpu_names": sorted(n for n in names if n),
            "driver_versions": sorted(d for d in drivers if d),
            "nodes_probed": sorted({p["node"] for p in probes if p["node"]}),
            "pair_link_gpu0_gpu1": pair,
            "gemm_bf16_8192_tflops": {"samples": gemm, "median": statistics.median(gemm) if gemm else None},
            "d2d_copy_gbps": {"samples": hbm, "median": statistics.median(hbm) if hbm else None},
            "p2p": p2p,
        },
        "probes": probes,
    }
    if args.cpu:
        manifest["cpu_probe"] = parse_cpu(args.cpu)
    text = scrub(json.dumps(manifest, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    raw = args.out.parent / "raw"
    for d in args.h200:
        copy_scrubbed(d, RAW_KEEP, raw / d.name)
    if args.cpu:
        copy_scrubbed(args.cpu, CPU_KEEP, raw / args.cpu.name)
    print(f"wrote {args.out.relative_to(ROOT)} and {raw.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
