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


def save_tiny_checkpoint(path: Path, tied: bool) -> Path:
    """A BF16 safetensors checkpoint split over several files, with an odd vocab (uneven shards)."""
    import torch
    from transformers import AutoModelForCausalLM, Qwen2Config

    cfg = Qwen2Config(
        vocab_size=515,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tied,
    )
    torch.manual_seed(1)
    model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
    model.save_pretrained(path, max_shard_size="100KB")
    assert (path / "model.safetensors.index.json").exists()
    return path


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_fsdp2_sharded_load_matches_full_load(tmp_path, tokens, tied):
    ckpt_dir = save_tiny_checkpoint(tmp_path / "model", tied)
    common = ("backend=fsdp2", "init=pretrained", f"model={ckpt_dir}", "steps=4")
    full = run(tmp_path, tokens, "full", *common, "sharded_load=false", nproc=2)
    sharded = run(tmp_path, tokens, "sharded", *common, "sharded_load=true", nproc=2)
    assert sharded["summary"]["sharded_load"] and not full["summary"]["sharded_load"]
    assert sharded["metrics"]["loss"] == full["metrics"]["loss"]
    assert sharded["metrics"]["grad_norm"] == full["metrics"]["grad_norm"]


@pytest.mark.parametrize("backend", ["ddp", "fsdp2"])
def test_accum_sync_every_matches_last(tmp_path, tokens, backend):
    last = run(tmp_path, tokens, "last", f"backend={backend}", "accum_sync=last", nproc=2)
    every = run(tmp_path, tokens, "every", f"backend={backend}", "accum_sync=every", nproc=2)
    assert every["summary"]["accum_steps"] == 2
    np.testing.assert_allclose(every["metrics"]["loss"], last["metrics"]["loss"], rtol=1e-6)


def test_oom_writes_a_record_and_stops_every_rank(tmp_path, tokens):
    run(tmp_path, tokens, "oom", "oom_at_step=3", nproc=2, expect_rc=1)
    (rec,) = [json.loads(p.read_text()) for p in (tmp_path / "res" / "train").glob("*.json")]
    s = rec["summary"]
    assert s["status"] == "oom" and s["phase"] == "step:3" and s["rank"] == 0
    assert rec["run_id"].endswith("-rank0") and s["predicted_gib"]["total"] > 0


def test_step_split_and_profiler_window(tmp_path, tokens):
    r = run(tmp_path, tokens, "prof", "profile_steps=2", "profile_start=2")
    s = r["summary"]
    assert s["profiled_steps"] == [2, 3]
    assert s["timed_steps"] == 4  # 8 steps - 3 warmup - 2 profiled
    assert min(s["fwd_s_median"], s["bwd_s_median"], s["opt_s_median"]) > 0
    assert s["fwd_s_median"] + s["bwd_s_median"] + s["opt_s_median"] <= s["step_time_s_median"] * 1.05
    zeros = {"exposed_comm_s": 0.0, "comm_s": 0.0, "compute_s": 0.0, "gpu_idle_s": 0.0}
    assert s["comm_by_rank"] == [zeros]  # CPU: no kernels
    assert len(r["metrics"]["fwd_s"]) == 8


@pytest.mark.parametrize("ac,every,expected", [(False, 1, [1, 1]), (True, 1, [2, 2]), (True, 2, [2, 1])])
def test_activation_checkpointing_really_recomputes(tmp_path, ac, every, expected):
    """Phase 3 found checkpointing silently off for from_pretrained models (left in eval mode)."""
    import torch

    from dlt.train import engine
    from dlt.train.config import parse

    d = save_tiny_checkpoint(tmp_path / "m", tied=True)
    sets = [
        "data=/dev/null",
        "init=pretrained",
        f"model={d}",
        "backend=single",
        f"activation_checkpointing={str(ac).lower()}",
        f"ac_every={every}",
    ]
    model = engine.build_model(parse([a for s in sets for a in ("--set", s)]))
    assert model.training
    calls = [0] * len(model.model.layers)
    # a pre-hook counts entries: the recompute stops early inside the MLP once the last saved tensor
    # exists (checkpoint early stopping), so a post-forward hook would never see it
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_pre_hook(lambda m, a, i=i: calls.__setitem__(i, calls[i] + 1))
    model(input_ids=torch.randint(0, 512, (2, 16))).logits.float().mean().backward()
    assert calls == expected


@pytest.mark.parametrize("nproc,backend", [(1, "single"), (2, "fsdp2")])
def test_chunked_loss_matches_full_logits(tmp_path, tokens, nproc, backend):
    full = run(tmp_path, tokens, "full", f"backend={backend}", nproc=nproc)
    # 64 tokens per micro-batch in chunks of 24 + 24 + 16
    chunked = run(tmp_path, tokens, "chunked", f"backend={backend}", "loss_chunk=24", nproc=nproc)
    assert chunked["metrics"]["loss"][0] == pytest.approx(full["metrics"]["loss"][0], rel=1e-6)
    # FSDP2 computes in BF16: the lm_head weight gradient is summed over the chunks in BF16 rather than
    # inside one GEMM, so later steps differ by BF16 rounding (about 1e-5 relative)
    rtol = 1e-5 if backend == "single" else 2e-4
    np.testing.assert_allclose(chunked["metrics"]["loss"], full["metrics"]["loss"], rtol=rtol)
    np.testing.assert_allclose(chunked["metrics"]["grad_norm"], full["metrics"]["grad_norm"], rtol=10 * rtol)


@pytest.mark.parametrize(
    "backend,knobs",
    [
        ("ddp", ["ddp_bucket_view=true"]),
        ("ddp", ["ddp_bucket_mb=0.001"]),
        ("fsdp2", ["reshard_after_forward=false"]),
        ("fsdp2", ["fsdp_prefetch=2"]),
    ],
    ids=["bucket-view", "tiny-buckets", "no-reshard", "prefetch-2"],
)
def test_distributed_knobs_do_not_change_the_math(tmp_path, tokens, backend, knobs):
    base = run(tmp_path, tokens, "base", f"backend={backend}", nproc=2)
    knob = run(tmp_path, tokens, "knob", f"backend={backend}", *knobs, nproc=2)
    assert knob["metrics"]["loss"] == base["metrics"]["loss"]
    assert knob["metrics"]["grad_norm"] == base["metrics"]["grad_norm"]
