"""Run a list of training configs inside one Slurm allocation (the Phase 3 sweeps).

    python -m dlt.sweep configs/sweeps/X.yaml [--dry-run] [--serial] [--retry-failed]

A sweep file has a ``name``, ``base`` TrainConfig fields shared by every run, a default ``repeats``,
and ``runs``: each with a ``name``, ``gpus`` (1 or 2) and ``set`` overrides. Run k of entry E is
named ``<sweep>.<E>.r<k>`` (the record's ``run_name``).

Scheduling inside a 2-GPU allocation: runs start in file order. A 2-GPU run waits for both GPUs; a
1-GPU run takes any free GPU, so consecutive 1-GPU runs execute in pairs, one per GPU (``--serial``
turns this off). Which runs overlapped is written to the sweep manifest, since a paired run shares the
node's host with its neighbour.

Resumable: a run whose record already exists with status ``ok`` (and not stopped early) or ``oom`` is
skipped, so the same sweep can be resubmitted or chained (slurm/chain.sh) until everything has run.
Failures are results: an OOM writes its own record (exit 75); any other non-zero exit without a record
gets a ``failed`` record with the log tail, and is not retried unless ``--retry-failed``.

Stopping: SIGUSR1 or SIGTERM stops new launches and forwards SIGUSR1 to the running workers, which
stop at the next step and write their (partial, so not counted as done) records.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from dlt.metrics.flops import QWEN25
from dlt.metrics.memory import GIB, predict_for_config
from dlt.results import load_records, new_record, write_record
from dlt.train.config import TrainConfig, coerce

OOM_EXIT = 75


@dataclass
class Run:
    name: str
    entry: str
    gpus: int
    sets: dict
    devices: list[str] = field(default_factory=list)
    proc: subprocess.Popen | None = None
    start: float = 0.0
    end: float = 0.0
    rc: int | None = None
    status: str = "pending"
    overlapped: set[str] = field(default_factory=set)


def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text())
    for key in ("name", "runs"):
        if key not in spec:
            raise ValueError(f"sweep file needs {key!r}")
    return spec


def plan(spec: dict) -> list[Run]:
    base = spec.get("base", {}) or {}
    runs = []
    for entry in spec["runs"]:
        if entry["gpus"] not in (1, 2):
            raise ValueError(f"{entry['name']}: gpus must be 1 or 2")
        sets = {**base, **(entry.get("set") or {})}
        sets = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in sets.items()}
        for k in sets:
            coerce(k, sets[k])  # unknown keys fail before any GPU time is spent
        for r in range(entry.get("repeats", spec.get("repeats", 1))):
            name = f"{spec['name']}.{entry['name']}.r{r}"
            runs.append(Run(name=name, entry=entry["name"], gpus=entry["gpus"], sets=sets))
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate run names")
    return runs


def predicted_gib(run: Run) -> float | None:
    cfg = TrainConfig(**{k: coerce(k, v) for k, v in run.sets.items()})
    shape = QWEN25.get(cfg.model)
    if shape is None or cfg.init == "tiny" or cfg.peft:  # the predictor covers full-parameter training
        return None
    accum = cfg.global_batch // (run.gpus * cfg.micro_batch)
    return predict_for_config(shape, cfg, run.gpus, accum)["total"] / GIB


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return yaml.safe_dump(v, default_flow_style=True).strip()
    return str(v)


def command(run: Run) -> list[str]:
    sets = {**run.sets, "run_name": run.name}
    args = [a for k, v in sets.items() for a in ("--set", f"{k}={_fmt(v)}")]
    if run.gpus == 1:
        return [sys.executable, "-m", "dlt.train", *args]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={run.gpus}",
        "-m",
        "dlt.train",
        *args,
    ]


def finished_runs(results: Path, retry_failed: bool) -> dict[str, str]:
    """run_name -> status for runs that count as done."""
    done: dict[str, str] = {}
    for rec in load_records(results, "train"):
        s = rec["summary"]
        name, status = s.get("run_name"), s.get("status", "ok")
        if not name:
            continue
        if status == "oom" or (status == "failed" and not retry_failed):
            done[name] = status
        elif status == "ok" and s.get("stopped_by") is None:
            done[name] = "ok"
    return done


def _children(pid: int) -> list[int]:
    kids: list[int] = []
    for task in Path(f"/proc/{pid}/task").glob("*"):
        with contextlib.suppress(OSError):
            kids += [int(k) for k in (task / "children").read_text().split()]
    return kids


def signal_workers(run: Run, sig: int = signal.SIGUSR1) -> None:
    """USR1 must reach the training processes, not torchrun (whose default action would kill it)."""
    if run.proc is None or run.proc.poll() is not None:
        return
    targets = [run.proc.pid] if run.gpus == 1 else _children(run.proc.pid)
    for pid in targets:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)


class Sweep:
    def __init__(self, spec: dict, runs: list[Run], args):
        self.spec, self.runs, self.args = spec, runs, args
        self.results = Path(os.environ.get("DLT_RESULTS", "results/local"))
        job = os.environ.get("SLURM_JOB_ID", "local")
        self.logs = Path(os.environ.get("DLT_HOME", ".")) / "logs" / f"sweep-{job}"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        self.devices = [d for d in visible.split(",") if d] if visible else ["0", "1"]
        self.stopping = False
        self.timeout = float(spec.get("run_timeout_s", 1800))

    def on_signal(self, signum, frame):  # noqa: ARG002
        if not self.stopping:
            print(f"sweep: signal {signum}, no new runs; stopping the running ones", flush=True)
        self.stopping = True
        for r in self.running():
            signal_workers(r)

    def running(self) -> list[Run]:
        return [r for r in self.runs if r.status == "running"]

    def free_devices(self) -> list[str]:
        used = {d for r in self.running() for d in r.devices}
        return [d for d in self.devices if d not in used]

    def launch(self, run: Run, devices: list[str]) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices), "DLT_SWEEP": self.spec["name"]}
        log = open(self.logs / f"{run.name}.log", "w")  # noqa: SIM115 (closed when the run ends)
        run.devices, run.start, run.status = devices, time.time(), "running"
        for other in self.running():
            if other is not run:
                other.overlapped.add(run.name)
                run.overlapped.add(other.name)
        run.proc = subprocess.Popen(
            command(run), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        run._log = log  # type: ignore[attr-defined]
        pred = predicted_gib(run)
        ptxt = f" predicted {pred:.1f} GiB" if pred else ""
        print(f"sweep: start {run.name} on GPU {','.join(devices)}{ptxt}", flush=True)

    def reap(self, run: Run) -> None:
        run.end, run.rc = time.time(), run.proc.returncode
        run._log.close()  # type: ignore[attr-defined]
        done = finished_runs(self.results, retry_failed=True)
        if run.rc == 0 and done.get(run.name) == "ok":
            run.status = "ok"
        elif run.rc == 0:
            run.status = "stopped"  # stopped by a signal before finishing; runs again next time
        elif run.rc == OOM_EXIT or done.get(run.name) == "oom":
            run.status = "oom"
        else:
            run.status = "failed"
            log = (self.logs / f"{run.name}.log").read_text(errors="replace").splitlines()
            rec = new_record(
                "train",
                config={**run.sets, "run_name": run.name},
                metrics={"log_tail": log[-60:]},
                summary={"run_name": run.name, "status": "failed", "returncode": run.rc, "gpus": run.gpus},
            )
            write_record(rec)
        print(f"sweep: {run.status} {run.name} rc={run.rc} in {run.end - run.start:.0f}s", flush=True)
        self.write_manifest()

    def write_manifest(self) -> None:
        rows = [
            {
                "run_name": r.name,
                "entry": r.entry,
                "gpus": r.gpus,
                "devices": r.devices,
                "status": r.status,
                "returncode": r.rc,
                "start": round(r.start, 1) if r.start else None,
                "seconds": round(r.end - r.start, 1) if r.end else None,
                "overlapped_with": sorted(r.overlapped),
                "predicted_gib": round(p, 2) if (p := predicted_gib(r)) else None,
            }
            for r in self.runs
        ]
        counts: dict[str, int] = {}
        for r in self.runs:
            counts[r.status] = counts.get(r.status, 0) + 1
        rec = new_record(
            "sweep",
            config={"spec": self.spec, "serial": self.args.serial},
            metrics={"runs": rows},
            summary={"sweep": self.spec["name"], "counts": counts},
        )
        rec["run_id"] = f"sweep-{self.spec['name']}-{os.environ.get('SLURM_JOB_ID', 'local')}"
        write_record(rec)

    def loop(self) -> int:
        signal.signal(signal.SIGUSR1, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        pending = [r for r in self.runs if r.status == "pending"]
        while pending or self.running():
            for r in self.running():
                if r.proc.poll() is not None:
                    self.reap(r)
                elif time.time() - r.start > self.timeout:
                    print(f"sweep: {r.name} exceeded {self.timeout:.0f}s, stopping it", flush=True)
                    signal_workers(r)
                    if time.time() - r.start > self.timeout + 120:
                        os.killpg(r.proc.pid, signal.SIGKILL)
            if not self.stopping and pending:
                nxt, free = pending[0], self.free_devices()
                serial_ok = not self.args.serial or not self.running()
                if nxt.gpus <= len(free) and serial_ok and (nxt.gpus == 1 or not self.running()):
                    self.launch(nxt, free[: nxt.gpus])
                    pending.pop(0)
                    continue
            if self.stopping and not self.running():
                break
            time.sleep(0.5)
        self.write_manifest()
        left = sum(r.status in ("pending", "stopped") for r in self.runs)
        print(f"sweep: done, {left} runs left for a later job", flush=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and predicted memory")
    ap.add_argument("--serial", action="store_true", help="never run two 1-GPU runs at once")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args(argv)
    spec = load_spec(args.spec)
    runs = plan(spec)
    results = Path(os.environ.get("DLT_RESULTS", "results/local"))
    done = finished_runs(results, args.retry_failed)
    for r in runs:
        if r.name in done:
            r.status = f"done:{done[r.name]}"
    for r in runs:
        pred = predicted_gib(r)
        ptxt = f"{pred:7.1f}" if pred else "      -"
        print(f"{r.status:10} {r.gpus} GPU  predicted {ptxt} GiB  {r.name}", flush=True)
    if args.dry_run:
        return 0
    return Sweep(spec, runs, args).loop()


if __name__ == "__main__":
    raise SystemExit(main())
