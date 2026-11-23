"""Run records: one JSON file per run, the only source for numbers in the README.

A record holds what was run (resolved config, git SHA, lockfile hash, Slurm job, node, GPU) and what
was measured (per-step or per-size metrics plus a summary). Files live under
``<results root>/<kind>/<run_id>.json``.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KINDS = {"nccl_bw", "smoke_train", "bench", "train", "eval", "sweep"}
REQUIRED = {
    "schema_version": int,
    "kind": str,
    "run_id": str,
    "created_at": str,
    "git_sha": str,
    "lock_hash": str,
    "slurm": dict,
    "env": dict,
    "hardware": dict,
    "config": dict,
    "metrics": dict,
    "summary": dict,
}
PACKAGES = ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "nvidia-nccl-cu12"]


def now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def git_sha() -> str:
    if os.environ.get("DLT_GIT_SHA"):
        return os.environ["DLT_GIT_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            env[p] = _md.version(p)
        except _md.PackageNotFoundError:
            env[p] = None
    try:
        import torch

        env["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            env["nccl_runtime"] = ".".join(map(str, torch.cuda.nccl.version()))
    except ImportError:
        pass
    return env


def collect_slurm() -> dict[str, Any]:
    keys = {
        "job_id": "SLURM_JOB_ID",
        "partition": "SLURM_JOB_PARTITION",
        "nodes": "SLURM_JOB_NODELIST",
        "gpus_on_node": "SLURM_GPUS_ON_NODE",
        "cpus_on_node": "SLURM_CPUS_ON_NODE",
    }
    out: dict[str, Any] = {k: os.environ.get(v) for k, v in keys.items()}
    out["host"] = socket.gethostname().split(".")[0]
    return out


def collect_hardware() -> dict[str, Any]:
    hw: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            hw["gpu_name"] = torch.cuda.get_device_name(0)
            hw["gpu_count"] = torch.cuda.device_count()
            hw["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        pass
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.split()
        hw["driver"] = drv[0] if drv else None
    except (OSError, subprocess.SubprocessError):
        hw["driver"] = None
    return hw


def new_record(kind: str, config: dict, metrics: dict, summary: dict, notes: str = "") -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    slurm = collect_slurm()
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S")
    # the pid keeps records unique when runs that share a job (sweep pairs) finish in the same second
    run_id = f"{kind}-{slurm['job_id'] or 'local'}-{stamp}-p{os.getpid()}"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "run_id": run_id,
        "created_at": now_iso(),
        "git_sha": git_sha(),
        "lock_hash": os.environ.get("DLT_LOCK_HASH", "unknown"),
        "slurm": slurm,
        "env": collect_env(),
        "hardware": collect_hardware(),
        "config": config,
        "metrics": metrics,
        "summary": summary,
        "notes": notes,
    }


def validate(rec: dict[str, Any]) -> list[str]:
    errors = [f"missing or wrong type: {k}" for k, t in REQUIRED.items() if not isinstance(rec.get(k), t)]
    if rec.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version {rec.get('schema_version')} != {SCHEMA_VERSION}")
    if rec.get("kind") not in KINDS:
        errors.append(f"unknown kind {rec.get('kind')!r}")
    return errors


def results_root() -> Path:
    return Path(os.environ.get("DLT_RESULTS", "results/local"))


def write_record(rec: dict[str, Any], root: Path | None = None) -> Path:
    errors = validate(rec)
    if errors:
        raise ValueError("; ".join(errors))
    root = root or results_root()
    path = root / rec["kind"] / f"{rec['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)
    return path


def load_records(root: Path, kind: str | None = None) -> list[dict[str, Any]]:
    pattern = f"{kind}/*.json" if kind else "*/*.json"
    recs = [json.loads(p.read_text()) for p in sorted(root.glob(pattern))]
    return sorted(recs, key=lambda r: r.get("created_at", ""))
