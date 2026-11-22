"""CPU runs of the benchmark entry points (single process, no torchrun)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dlt.bench import smoke_train
from dlt.bench.nccl_bw import bus_factor

ROOT = Path(__file__).resolve().parent.parent


def test_bus_factors_match_nccl_tests():
    assert bus_factor("all_reduce", 2) == 1.0
    assert bus_factor("all_reduce", 8) == pytest.approx(1.75)
    assert bus_factor("all_gather", 2) == 0.5
    assert bus_factor("reduce_scatter", 4) == pytest.approx(0.75)


def test_smoke_train_tiny_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("DLT_RESULTS", str(tmp_path))
    rc = smoke_train.main(
        [
            "--tiny",
            "--device",
            "cpu",
            "--steps",
            "12",
            "--warmup",
            "2",
            "--seq-len",
            "32",
            "--micro-batch",
            "2",
        ]
    )
    assert rc == 0
    (path,) = (tmp_path / "smoke_train").glob("*.json")
    rec = json.loads(path.read_text())
    s = rec["summary"]
    assert s["steps_run"] == 12
    assert s["stopped_by_signal_at_step"] is None
    assert s["loss_last"] < s["loss_first"]  # one fixed batch: the model must fit it
    assert s["mfu"] is None  # no GPU, no peak


def test_readme_is_generated():
    out = subprocess.run(
        [sys.executable, "scripts/gen_readme.py", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout
