"""Memory predictor: structure, model-card sizes, and error against the committed H200 runs."""

import json
from pathlib import Path

import pytest

from dlt.metrics.flops import QWEN25
from dlt.metrics.memory import GIB, predict, predict_gib

ROOT = Path(__file__).resolve().parent.parent
H200_GIB = 139.8  # visible to PyTorch (docs/phase0.md)


@pytest.mark.parametrize(
    "model,params",
    [("0.5B", 0.494e9), ("1.5B", 1.54e9), ("3B", 3.09e9), ("7B", 7.61e9), ("14B", 14.7e9), ("72B", 72.7e9)],
)
def test_shapes_match_model_cards(model, params):
    assert QWEN25[f"Qwen/Qwen2.5-{model}"].total_params() == pytest.approx(params, rel=6e-3)


def kw(**over):
    base = dict(backend="fsdp2", world=2, seq_len=2048, micro_batch=1, accum=4)
    return {**base, **over}


def test_fsdp2_shards_states_and_no_sync_costs_a_full_fp32_gradient():
    s = QWEN25["Qwen/Qwen2.5-7B"]
    one, two = predict(s, **kw(world=1)), predict(s, **kw(world=2))
    assert two["adam_fp32"] == pytest.approx(one["adam_fp32"] / 2)
    assert two["fsdp_unsharded_grads"] == 4 * s.total_params()
    assert "grads_fp32" not in two  # sharded grads are freed until the last micro-batch's reduce-scatter
    every = predict(s, **kw(accum_sync="every"))
    assert "fsdp_unsharded_grads" not in every and every["grads_fp32"] == 2 * s.total_params()
    assert "fsdp_unsharded_grads" not in predict(s, **kw(accum=1))


def test_checkpointing_cuts_activations():
    s = QWEN25["Qwen/Qwen2.5-3B"]
    on = predict(s, **kw(activation_checkpointing=True))["activations"]
    off = predict(s, **kw())["activations"]
    assert on < off / 5


def test_predicted_feasibility_of_the_ladder():
    q = QWEN25
    # 14B on 2 GPUs needs sharding, checkpointing and per-micro-batch sync to fit
    assert predict_gib(q["Qwen/Qwen2.5-14B"], **kw()) > H200_GIB
    assert (
        predict_gib(q["Qwen/Qwen2.5-14B"], **kw(activation_checkpointing=True, accum_sync="every")) < H200_GIB
    )
    # 7B single-GPU full training does not fit without checkpointing
    assert predict_gib(q["Qwen/Qwen2.5-7B"], **kw(backend="single", world=1)) > H200_GIB


def committed_parity_runs():
    for path in sorted((ROOT / "results" / "train").glob("*.json")):
        rec = json.loads(path.read_text())
        s, c = rec["summary"], rec["config"]
        if s.get("run_name", "").startswith("parity-") and rec["hardware"].get("gpu_name"):
            yield s, c


def test_predictor_within_10_percent_of_measured_parity_runs():
    runs = list(committed_parity_runs())
    assert len(runs) == 3, "parity-single, parity-ddp and parity-fsdp2 from the Phase 2 GPU job"
    for s, c in runs:
        backend = "single" if s["world_size"] == 1 else s["backend"]
        pred = predict(
            QWEN25[c["model"]],
            backend=backend,
            world=s["world_size"],
            seq_len=c["seq_len"],
            micro_batch=c["micro_batch"],
            accum=s["accum_steps"],
            activation_checkpointing=c["activation_checkpointing"],
            accum_sync=c.get("accum_sync", "last"),
        )["total"]
        measured = max(s["peak_alloc_gib_by_rank"]) * GIB
        assert abs(pred - measured) / measured < 0.10, (s["run_name"], pred / GIB, measured / GIB)


def test_phase4_knob_terms():
    s = QWEN25["Qwen/Qwen2.5-1.5B"]
    ddp = dict(backend="ddp", world=2, seq_len=2048, micro_batch=4, accum=4)
    assert "ddp_buckets" not in predict(s, **ddp, ddp_bucket_view=True)
    full, chunk = predict(s, **ddp), predict(s, **ddp, loss_chunk=1024)
    assert chunk["logits"] == full["logits"] / 8
    ac1 = predict(s, **ddp, activation_checkpointing=True)["activations"]
    ac2 = predict(s, **ddp, activation_checkpointing=True, ac_every=2)["activations"]
    off = predict(s, **ddp)["activations"]
    assert ac1 < ac2 < off
    f = dict(backend="fsdp2", world=2, seq_len=2048, micro_batch=4, accum=4)
    assert (
        predict(s, **f, reshard_after_forward=False)["fsdp_gathered_bf16"]
        > predict(s, **f)["fsdp_gathered_bf16"]
    )


def test_current_predictor_within_5_percent_of_valid_phase3_runs():
    """Regression guard: every finished Phase 3 run (invalid records excluded) against today's model."""
    from types import SimpleNamespace

    from dlt.metrics.memory import predict_for_config
    from dlt.results import load_records

    runs = [
        r
        for r in load_records(ROOT / "results", "train")
        if r["summary"].get("run_name", "").startswith("p3.")
        and r["summary"].get("status") == "ok"
        and r["summary"].get("stopped_by") is None
    ]
    assert len(runs) >= 40
    for r in runs:
        s, c = r["summary"], SimpleNamespace(**r["config"])
        pred = predict_for_config(QWEN25[c.model], c, s["world_size"], s["accum_steps"])["total"] / GIB
        measured = max(s["peak_alloc_gib_by_rank"])
        assert abs(pred - measured) / measured < 0.05, (s["run_name"], pred, measured)
