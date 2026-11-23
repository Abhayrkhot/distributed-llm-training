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
        ("fsdp2", ["fsdp_hold_params=true"]),
    ],
    ids=["bucket-view", "tiny-buckets", "no-reshard", "prefetch-2", "hold-params"],
)
def test_distributed_knobs_do_not_change_the_math(tmp_path, tokens, backend, knobs):
    base = run(tmp_path, tokens, "base", f"backend={backend}", nproc=2)
    knob = run(tmp_path, tokens, "knob", f"backend={backend}", *knobs, nproc=2)
    assert knob["metrics"]["loss"] == base["metrics"]["loss"]
    assert knob["metrics"]["grad_norm"] == base["metrics"]["grad_norm"]


@pytest.mark.parametrize("nproc,backend", [(1, "single"), (2, "ddp"), (2, "fsdp2")])
def test_compiled_layers_match_eager(tmp_path, tokens, nproc, backend):
    eager = run(tmp_path, tokens, "eager", f"backend={backend}", nproc=nproc)
    compiled = run(
        tmp_path,
        tokens,
        "compiled",
        f"backend={backend}",
        "compile=true",
        "compile_backend=aot_eager",
        nproc=nproc,
    )
    np.testing.assert_allclose(compiled["metrics"]["loss"], eager["metrics"]["loss"], rtol=1e-5)


def make_sft_fixture(root: Path) -> dict:
    """A tiny Qwen2 checkpoint with a word-level tokenizer, BIRD-style questions and schemas (no network)."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

    from dlt.bird.prompt import prompt, target

    rows = []
    for i in range(24):
        db = f"db{i % 2}"
        n = i % 6
        rows.append(
            {
                "question_id": i,
                "db_id": db,
                "question": f"how many rows where x equals {n} ?",
                "evidence": "x is a column" if i % 3 else "",
                "sql": f"SELECT count ( * ) FROM t WHERE x = {n}",
            }
        )
    schemas = {
        f"db{j}": {
            "ddl": f"CREATE TABLE t ( x INTEGER , y{j} TEXT ) ;",
            "ddl_rows": f"CREATE TABLE t ( x INTEGER , y{j} TEXT ) ;\n-- t row : ( 1 , 'a' )",
        }
        for j in range(2)
    }
    words = set()
    for r in rows:
        for s in schemas[r["db_id"]].values():
            words |= set((prompt(r, s) + target(r)).split())
    vocab = {w: i for i, w in enumerate(["[UNK]", "<|endoftext|>", *sorted(words)])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="[UNK]"
    )
    model_dir = root / "tiny-lm"
    cfg = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        eos_token_id=1,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    data = root / "bird"
    data.mkdir()
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:16]))
    (data / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[16:]))
    (data / "schemas.json").write_text(json.dumps(schemas))
    import sqlite3

    (data / "databases").mkdir()
    for j in range(2):
        con = sqlite3.connect(data / "databases" / f"db{j}.sqlite")
        con.execute(f"CREATE TABLE t (x INTEGER, y{j} TEXT)")
        con.executemany("INSERT INTO t VALUES (?, ?)", [(k % 6, f"v{k}") for k in range(20)])
        con.commit()
        con.close()
    return {"model": model_dir, "data": data, "tokenizer": tok}


def test_sft_examples_mask_prompt_and_padding(tmp_path):
    from dlt.data.packed import GlobalOrder
    from dlt.data.sft import SFTExamples

    fx = make_sft_fixture(tmp_path)
    ds = SFTExamples(fx["data"] / "train.jsonl", fx["data"] / "schemas.json", fx["tokenizer"], max_len=128)
    assert len(ds) == 16 and ds.dropped == 0
    import torch

    x, y = ds.batch(torch.tensor([0, 1]))
    for row, e in enumerate(ds.items[:2]):
        n = len(e.ids) - 1
        assert x[row, :n].tolist() == list(e.ids[:-1])
        assert (y[row, : e.n_prompt - 1] == -100).all() and (y[row, n:] == -100).all()
        assert y[row, e.n_prompt - 1 : n].tolist() == list(e.ids[e.n_prompt :])  # SQL + eos
    assert ds.target_tokens(torch.tensor([0, 1])) == int((y != -100).sum())
    order = GlobalOrder(len(ds), 8, 0)
    r0 = ds.micro_batches(order, 0, 0, 2, 2)
    r1 = ds.micro_batches(order, 0, 1, 2, 2)
    got = sorted(i for m in r0 + r1 for i in m.tolist())
    assert got == sorted(order.global_indices(0).tolist())  # the same examples, dealt round-robin by length
    tight = SFTExamples(fx["data"] / "train.jsonl", fx["data"] / "schemas.json", fx["tokenizer"], max_len=12)
    assert tight.dropped == 16


def sft_run(tmp, fx, name, *sets, nproc=1):
    base = (
        "task=sft",
        "init=pretrained",
        f"model={fx['model']}",
        f"schemas={fx['data'] / 'schemas.json'}",
        "peft=lora",
        "lora_r=4",
        "seq_len=128",
        "global_batch=4",
        "micro_batch=1",
        "steps=12",
        "lr=1e-2",
        "weight_decay=0.0",
    )
    return run(tmp, fx["data"] / "train.jsonl", name, *base, *sets, nproc=nproc)


def test_lora_sft_learns_audits_and_exports(tmp_path):
    fx = make_sft_fixture(tmp_path)
    r = sft_run(
        tmp_path,
        fx,
        "lora",
        "steps=40",
        "lora_r=16",
        "warmup_steps=5",
        f"adapter_out={tmp_path / 'adapter'}",
        f"val_data={fx['data'] / 'val.jsonl'}",
        "eval_samples=8",
    )
    s, a = r["summary"], r["summary"]["adapter_audit"]
    assert s["peft"] == "lora" and s["task"] == "sft"
    assert a["all_trainable_are_lora"] and a["optimizer_holds_only_trainable"]
    # 2 layers x r=16 x (q 64+64, k 64+32, v 64+32, o 64+64, gate/up 64+128, down 128+64)
    assert a["trainable_params"] == a["expected_lora_params"] == 2 * 16 * 1024
    assert a["frozen_unchanged_after_step"]
    loss = r["metrics"]["loss"]
    # only the adapters train (a random frozen base and output layer): the loss falls, it cannot vanish
    assert sum(loss[-4:]) < 0.85 * sum(loss[:4])
    assert s["val_loss_end"] is not None
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()


def test_lora_resume_is_bit_exact_and_saves_only_the_adapter(tmp_path):
    fx = make_sft_fixture(tmp_path)
    full = sft_run(tmp_path, fx, "full")
    ck = tmp_path / "ck"
    sft_run(tmp_path, fx, "first", f"ckpt_dir={ck}", "stop_at_step=5")
    second = sft_run(tmp_path, fx, "second", f"ckpt_dir={ck}")
    assert second["summary"]["resumed_from"] == "step_00000005"
    assert second["metrics"]["loss"] == full["metrics"]["loss"][5:]
    import torch.distributed.checkpoint as dcp

    keys = dcp.FileSystemReader(str(ck / "step_00000012")).read_metadata().state_dict_metadata
    model_keys = [k for k in keys if k.startswith("model.")]
    assert model_keys and all("lora_" in k for k in model_keys)  # the adapter, never the frozen base


def test_lora_sft_ddp_matches_single(tmp_path):
    fx = make_sft_fixture(tmp_path)
    single = sft_run(tmp_path, fx, "single", "backend=single", "global_batch=4", "micro_batch=2")
    ddp = sft_run(tmp_path, fx, "ddp", "backend=ddp", "global_batch=4", "micro_batch=2", nproc=2)
    # the first steps agree to 1e-6; DDP sums the two ranks' gradients in a different order than one process
    # accumulates them, and at this test's learning rate (1e-2) that grows to ~2e-5 by step 11 on some CPUs
    np.testing.assert_allclose(ddp["metrics"]["loss"][:6], single["metrics"]["loss"][:6], rtol=1e-5)
    np.testing.assert_allclose(ddp["metrics"]["loss"], single["metrics"]["loss"], rtol=1e-4)


def test_bird_evaluate_oracle_and_adapter(tmp_path):
    fx = make_sft_fixture(tmp_path)
    sft_run(tmp_path, fx, "lora", "steps=4", "lora_r=8", f"adapter_out={tmp_path / 'adapter'}")
    env = {
        **os.environ,
        "DLT_RESULTS": str(tmp_path / "res"),
        "DLT_HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "OMP_NUM_THREADS": "1",
    }

    def evaluate(*args):
        cmd = [
            sys.executable,
            "-m",
            "dlt.bird.evaluate",
            "--data",
            str(fx["data"]),
            "--split",
            "val",
            "--workers",
            "1",
            *args,
        ]
        out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
        recs = [json.loads(p.read_text()) for p in (tmp_path / "res" / "eval").glob("*.json")]
        return max(recs, key=lambda r: r["created_at"])["summary"]

    oracle = evaluate("--mode", "oracle", "--name", "oracle")
    assert oracle["ex"] == 1.0 and oracle["questions"] == 8 and oracle["gold_errors"] == 0
    tuned = evaluate(
        "--mode",
        "finetuned",
        "--model",
        str(fx["model"]),
        "--quant",
        "bf16",
        "--adapter",
        str(tmp_path / "adapter"),
        "--max-new",
        "12",
        "--batch",
        "4",
        "--name",
        "tuned",
    )
    assert tuned["questions"] == 8 and tuned["adapter"] == "adapter" and 0.0 <= tuned["ex"] <= 1.0
    lines = (tmp_path / "eval" / "tuned.jsonl").read_text().splitlines()
    assert len(lines) == 8 and {"pred", "correct", "pred_error"} <= set(json.loads(lines[0]))


def _no_gpu() -> bool:
    import torch

    return not torch.cuda.is_available()


@pytest.mark.skipif(
    _no_gpu(), reason="FSDP1 needs an accelerator; this parity check runs on H200s in Phase 7"
)
def test_lora_sft_fsdp1_matches_single_and_exports(tmp_path):
    fx = make_sft_fixture(tmp_path)
    single = sft_run(tmp_path, fx, "single", "backend=single", "micro_batch=2")
    fsdp1 = sft_run(
        tmp_path,
        fx,
        "fsdp1",
        "backend=fsdp1",
        "micro_batch=2",
        f"adapter_out={tmp_path / 'adapter'}",
        nproc=2,
    )
    np.testing.assert_allclose(fsdp1["metrics"]["loss"], single["metrics"]["loss"], rtol=1e-5)
    assert fsdp1["summary"]["adapter_audit"]["frozen_unchanged_after_step"]
    from safetensors.torch import load_file

    w = load_file(tmp_path / "adapter" / "adapter_model.safetensors")
    assert w and all("lora_" in k for k in w)


@pytest.mark.parametrize("nproc", [1, 2])
def test_async_checkpoint_resume_is_bit_exact(tmp_path, tokens, nproc):
    full = run(tmp_path, tokens, "full", nproc=nproc)
    ck = tmp_path / "ck"
    first = run(
        tmp_path,
        tokens,
        "first",
        f"ckpt_dir={ck}",
        "ckpt_every=2",
        "stop_at_step=5",
        "async_ckpt=true",
        nproc=nproc,
    )
    saves = first["summary"]["checkpoints"]
    assert [c["step"] for c in saves] == [2, 4, 5] and all(c["async"] for c in saves)
    assert all(c["blocking_s"] <= c["total_s"] for c in saves)
    second = run(tmp_path, tokens, "second", f"ckpt_dir={ck}", "async_ckpt=true", nproc=nproc)
    assert second["summary"]["resumed_from"] == "step_00000005"
    assert second["metrics"]["loss"] == full["metrics"]["loss"][5:]


def test_sigterm_is_a_graceful_stop_only_when_asked(tmp_path, tokens):
    ck = tmp_path / "ck"
    r = run(tmp_path, tokens, "term", f"ckpt_dir={ck}", "term_at_step=3", "stop_on_term=true")
    assert r["summary"]["stopped_by"] == "signal" and r["summary"]["end_step"] == 4
    assert (ck / "step_00000004" / "meta.json").exists()
    ck2 = tmp_path / "ck2"
    run(tmp_path, tokens, "killed", f"ckpt_dir={ck2}", "term_at_step=3", expect_rc=1)  # default: TERM kills
    assert not any(p.name.startswith("step_") and not p.name.endswith(".tmp") for p in ck2.glob("*"))


def test_crash_during_eval_keeps_the_final_checkpoint(tmp_path, tokens):
    ck = tmp_path / "ck"
    v = (f"val_data={tokens}", "eval_samples=8")
    run(tmp_path, tokens, "evalcrash", *v, f"ckpt_dir={ck}", "crash_in_eval=true", expect_rc=1)
    assert (ck / "step_00000008" / "meta.json").exists()  # the run had finished: only the eval is lost
    assert run(tmp_path, tokens, "again", *v, f"ckpt_dir={ck}") is None  # the next link sees it complete


def test_fsdp2_single_process_without_torchrun(tmp_path, tokens):
    """The sweep starts 1-GPU runs with plain python; FSDP2 then needs a group of its own (Phase 3 bug)."""
    single = run(tmp_path, tokens, "single")
    fsdp = run(tmp_path, tokens, "fsdp2-1", "backend=fsdp2")
    assert fsdp["summary"]["world_size"] == 1
    np.testing.assert_allclose(fsdp["metrics"]["loss"], single["metrics"]["loss"], rtol=2e-3)
