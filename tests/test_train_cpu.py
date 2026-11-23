"""Engine correctness on CPU (gloo): resume parity, crash recovery, backend parity, accumulation.

Every run is a subprocess (crash tests SIGKILL themselves) on a 2-layer toy Qwen2 over random tokens,
in deterministic mode with eager attention, so "resumed equals uninterrupted" can be asserted bit for bit.
The same properties are re-checked on H200s with real models in the Phase 2 GPU job.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tokens(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "train.bin"
    np.random.default_rng(0).integers(0, 512, size=40_000, dtype=np.uint32).tofile(path)
    return path


def run(tmp: Path, tokens: Path, name: str, *sets: str, nproc: int = 1, expect_rc: int = 0) -> dict | None:
    base = {
        "data": str(tokens),
        "init": "tiny",
        "seq_len": "32",
        "global_batch": "8",
        "micro_batch": "2",
        "steps": "8",
        "warmup_steps": "2",
        "lr": "1e-3",
        "attn": "eager",
        "deterministic": "true",
        "backend": "single" if nproc == 1 else "ddp",
        "run_name": name,
    }
    for s in sets:
        k, _, v = s.partition("=")
        base[k] = v
    args = [a for k, v in base.items() for a in ("--set", f"{k}={v}")]
    if nproc == 1:
        cmd = [sys.executable, "-m", "dlt.train", *args]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={nproc}"]
        cmd += ["-m", "dlt.train", *args]
    env = {
        **os.environ,
        "DLT_RESULTS": str(tmp / "res"),
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
    }
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    rc = out.returncode if nproc == 1 or out.returncode >= 0 else out.returncode
    if expect_rc == 0:
        assert rc == 0, out.stdout[-3000:] + out.stderr[-3000:]
    else:
        assert rc != 0, "expected the run to die"
        return None
    recs = [json.loads(p.read_text()) for p in (tmp / "res" / "train").glob("*.json")]
    mine = sorted((r for r in recs if r["summary"]["run_name"] == name), key=lambda r: r["created_at"])
    return mine[-1] if mine else None


def test_stop_and_resume_is_bit_exact(tmp_path, tokens):
    full = run(tmp_path, tokens, "full")
    ck = tmp_path / "ck"
    first = run(tmp_path, tokens, "first", f"ckpt_dir={ck}", "stop_at_step=4")
    assert first["summary"]["stopped_by"] == "stop_at_step" and first["summary"]["end_step"] == 4
    second = run(tmp_path, tokens, "second", f"ckpt_dir={ck}")
    assert second["summary"]["resumed_from"] == "step_00000004"
    assert first["metrics"]["loss"] == full["metrics"]["loss"][:4]
    assert second["metrics"]["loss"] == full["metrics"]["loss"][4:]
    assert second["metrics"]["grad_norm"] == full["metrics"]["grad_norm"][4:]


def test_sigkill_then_resume_from_last_checkpoint(tmp_path, tokens):
    full = run(tmp_path, tokens, "full")
    ck = tmp_path / "ck"
    run(tmp_path, tokens, "crash", f"ckpt_dir={ck}", "ckpt_every=2", "crash_at_step=5", expect_rc=1)
    assert sorted(p.name for p in ck.iterdir()) == ["step_00000002", "step_00000004"]
    resumed = run(tmp_path, tokens, "resumed", f"ckpt_dir={ck}", "ckpt_every=2")
    assert resumed["summary"]["start_step"] == 4  # step 4 is lost work, recomputed
    assert resumed["metrics"]["loss"] == full["metrics"]["loss"][4:]


def test_crash_during_save_falls_back_to_previous_checkpoint(tmp_path, tokens):
    full = run(tmp_path, tokens, "full")
    ck = tmp_path / "ck"
    run(tmp_path, tokens, "a", f"ckpt_dir={ck}", "ckpt_every=2", "stop_at_step=4")
    run(tmp_path, tokens, "b", f"ckpt_dir={ck}", "ckpt_every=2", "crash_in_save=true", expect_rc=1)
    names = sorted(p.name for p in ck.iterdir())
    assert "step_00000006.tmp" in names and "step_00000006" not in names
    c = run(tmp_path, tokens, "c", f"ckpt_dir={ck}", "ckpt_every=2")
    assert c["summary"]["resumed_from"] == "step_00000004"
    assert c["metrics"]["loss"] == full["metrics"]["loss"][4:]
    assert not any(p.name.endswith(".tmp") for p in ck.iterdir())  # the next save cleaned it up


def test_gradient_accumulation_matches_one_big_micro_batch(tmp_path, tokens):
    big = run(tmp_path, tokens, "big", "micro_batch=8")
    small = run(tmp_path, tokens, "small", "micro_batch=2")
    assert small["summary"]["accum_steps"] == 4 and big["summary"]["accum_steps"] == 1
    np.testing.assert_allclose(small["metrics"]["loss"], big["metrics"]["loss"], rtol=1e-5)


def test_ddp_two_ranks_matches_single_process(tmp_path, tokens):
    single = run(tmp_path, tokens, "single")
    ddp = run(tmp_path, tokens, "ddp", nproc=2)
    assert ddp["summary"]["world_size"] == 2 and ddp["summary"]["accum_steps"] == 2
    np.testing.assert_allclose(ddp["metrics"]["loss"], single["metrics"]["loss"], rtol=1e-5)


def test_ddp_two_ranks_resume_is_bit_exact(tmp_path, tokens):
    full = run(tmp_path, tokens, "full", nproc=2)
    ck = tmp_path / "ck"
    run(tmp_path, tokens, "first", f"ckpt_dir={ck}", "stop_at_step=3", nproc=2)
    second = run(tmp_path, tokens, "second", f"ckpt_dir={ck}", nproc=2)
    assert second["metrics"]["loss"] == full["metrics"]["loss"][3:]


def test_fsdp2_two_ranks_tracks_single_process(tmp_path, tokens):
    single = run(tmp_path, tokens, "single")
    fsdp = run(tmp_path, tokens, "fsdp2", "backend=fsdp2", nproc=2)
    # FSDP2 computes with BF16 parameters (MixedPrecisionPolicy); the CPU single run is pure FP32
    np.testing.assert_allclose(fsdp["metrics"]["loss"], single["metrics"]["loss"], rtol=2e-3)


def test_fsdp2_resume_is_bit_exact(tmp_path, tokens):
    full = run(tmp_path, tokens, "full", "backend=fsdp2", nproc=2)
    ck = tmp_path / "ck"
    run(tmp_path, tokens, "first", "backend=fsdp2", f"ckpt_dir={ck}", "stop_at_step=5", nproc=2)
    second = run(tmp_path, tokens, "second", "backend=fsdp2", f"ckpt_dir={ck}", nproc=2)
    assert second["summary"]["resumed_from"] == "step_00000005"
    assert second["metrics"]["loss"] == full["metrics"]["loss"][5:]


def test_val_loss_across_world_sizes_resume_and_chain_end(tmp_path, tokens):
    v = (f"val_data={tokens}", "eval_samples=8")
    single = run(tmp_path, tokens, "single", *v)
    ddp = run(tmp_path, tokens, "ddp", *v, nproc=2)
    s, d = single["summary"], ddp["summary"]
    assert [e["step"] for e in single["metrics"]["val"]] == [0, 8]
    np.testing.assert_allclose(d["val_loss_start"], s["val_loss_start"], rtol=1e-6)
    np.testing.assert_allclose(d["val_loss_end"], s["val_loss_end"], rtol=1e-5)

    ck = tmp_path / "ck"
    first = run(tmp_path, tokens, "first", *v, f"ckpt_dir={ck}", "stop_at_step=4")
    assert first["summary"]["val_loss_start"] == s["val_loss_start"]
    second = run(tmp_path, tokens, "second", *v, f"ckpt_dir={ck}")
    assert second["summary"]["val_loss_start"] is None  # resumed runs skip the step-0 eval
    assert second["summary"]["val_loss_end"] == s["val_loss_end"]
    assert ck.joinpath("step_00000008").is_dir()  # completed runs checkpoint their last step

    # a further link of a chain finds the run complete and exits without loading a model or a record
    assert run(tmp_path, tokens, "third", *v, f"ckpt_dir={ck}") is None
