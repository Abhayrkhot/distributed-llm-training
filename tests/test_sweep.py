"""The sweep driver on CPU: pairing, OOM and failure records, resume by skipping finished runs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_sweep_pairs_records_failures_and_resumes(tmp_path):
    tokens = tmp_path / "train.bin"
    np.random.default_rng(0).integers(0, 512, size=20_000, dtype=np.uint32).tofile(tokens)
    spec = {
        "name": "t",
        "repeats": 1,
        "base": {
            "data": str(tokens),
            "init": "tiny",
            "seq_len": 32,
            "global_batch": 8,
            "micro_batch": 2,
            "steps": 6,
            "warmup_steps": 1,
            "backend": "ddp",
        },
        "runs": [
            {"name": "one", "gpus": 1, "repeats": 2},
            {"name": "two", "gpus": 2},
            {"name": "oom", "gpus": 1, "set": {"oom_at_step": 2}},
            {"name": "bad", "gpus": 2, "set": {"backend": "single"}},  # single needs one rank: fails
        ],
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec))
    env = {
        **os.environ,
        "DLT_RESULTS": str(tmp_path / "res"),
        "DLT_HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)

    def sweep():
        out = subprocess.run(
            [sys.executable, "-m", "dlt.sweep", str(path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
        return out.stdout

    sweep()
    (manifest,) = [json.loads(p.read_text()) for p in (tmp_path / "res" / "sweep").glob("*.json")]
    rows = {r["run_name"]: r for r in manifest["metrics"]["runs"]}
    assert {k: v["status"] for k, v in rows.items()} == {
        "t.one.r0": "ok",
        "t.one.r1": "ok",
        "t.two.r0": "ok",
        "t.oom.r0": "oom",
        "t.bad.r0": "failed",
    }
    assert rows["t.one.r0"]["overlapped_with"] == ["t.one.r1"]  # consecutive 1-GPU runs share the node
    assert rows["t.one.r0"]["devices"] != rows["t.one.r1"]["devices"]
    assert rows["t.two.r0"]["overlapped_with"] == []  # 2-GPU runs wait for both GPUs

    recs = [json.loads(p.read_text()) for p in (tmp_path / "res" / "train").glob("*.json")]
    by_status = sorted((r["summary"]["run_name"], r["summary"].get("status")) for r in recs)
    assert ("t.bad.r0", "failed") in by_status and ("t.oom.r0", "oom") in by_status
    assert any("needs exactly one rank" in line for r in recs for line in r["metrics"].get("log_tail", []))

    n_before = len(recs)
    second = sweep()  # everything is finished (failures are results): nothing runs again
    assert "sweep: start" not in second
    assert len(list((tmp_path / "res" / "train").glob("*.json"))) == n_before
