"""Analytical peak-memory predictor for full-parameter training of Qwen2-style decoders (Phase 3).

Every term is bytes per rank, at the moment memory peaks: the start of backward on a micro-batch whose
gradients from earlier micro-batches and the Adam state already exist. That holds from the second
optimizer step on whenever accumulation is > 1; with accumulation 1 the gradient term is an upper bound
(gradients are freed by ``zero_grad(set_to_none=True)`` and regrow during backward).

Precision follows dlt.train.engine: FP32 master parameters, gradients and Adam moments.
- single / DDP: BF16 autocast. The residual stream stays FP32 (FP32 embedding output plus BF16 sublayer
  outputs), and every matmul input is cast to BF16 separately (autocast does not cache non-leaf casts).
  Autocast caches BF16 copies of the weights, which matmuls save for backward.
- FSDP2: parameters cast to BF16 for compute, so activations are BF16; gradients reduce in FP32.

Saved activations per token per decoder layer (bytes), from the ops in transformers' Qwen2 code:
- RMSNorm saves its FP32 input and normalized value; its BF16 output is saved by the next matmuls.
- SDPA (flash or memory-efficient, GQA without repeating K/V) saves q, k, v and the output.
- o_proj saves its (contiguous) input. SwiGLU saves the gate output, SiLU output, up output, and the
  product that feeds down_proj.
  FSDP2: 22h + 4 h_kv + 8i.   single / DDP: 32h + 4 h_kv + 8i (FP32 norms, one BF16 cast per matmul).
Logits: the engine keeps the BF16 logits, cross-entropy saves the FP32 log-softmax output, and its
backward holds the incoming and outgoing FP32 gradients: 14 V bytes per token.

Autocast with activation checkpointing has two candidate peaks. The autocast cache holds a BF16 copy of
every weight until the autocast region ends (the end of the forward); with checkpointing, matmuls inside
checkpointed layers save nothing, so those copies are gone by backward. The prediction is the larger of
the end of the forward (states + every weight copy + saved layer inputs + BF16 logits) and the start of
backward (states + the copies still saved + activations + logits gradients). Version 3; version 2 kept
every copy at the backward peak.

The model is checked against every measured run; the error is reported in the README, not tuned away.
"""

from __future__ import annotations

from dlt.metrics.flops import ModelShape

GIB = 2**30


def _layer_params(s: ModelShape) -> int:
    bias = (s.attn_width + 2 * s.n_kv_heads * s.head_dim) if s.qkv_bias else 0
    return s.layer_matmul_params() + bias + 2 * s.hidden


def activation_bytes_per_token_layer(s: ModelShape, backend: str) -> int:
    h, hkv, i = s.hidden, s.n_kv_heads * s.head_dim, s.intermediate
    if backend == "fsdp2":
        return 22 * h + 4 * hkv + 8 * i
    return 32 * h + 4 * hkv + 8 * i


def predict(
    s: ModelShape,
    *,
    backend: str,
    world: int,
    seq_len: int,
    micro_batch: int,
    accum: int,
    activation_checkpointing: bool = False,
    accum_sync: str = "last",
    ac_every: int = 1,
    loss_chunk: int = 0,
    ddp_bucket_view: bool = False,
    reshard_after_forward: bool = True,
    fsdp_prefetch: int = 0,
    fsdp_hold_params: bool = False,
) -> dict[str, float]:
    """Predicted peak bytes per rank, by term, plus ``total``. torch.compile is not modeled: its
    partitioner may recompute cheap ops in backward and save less, so compiled runs can come in under."""
    reshard_after_forward = reshard_after_forward and not fsdp_hold_params
    if backend not in ("single", "ddp", "fsdp2"):
        raise ValueError(backend)
    p = s.total_params()
    pl = _layer_params(s)
    tokens = micro_batch * seq_len
    shard = world if backend == "fsdp2" else 1
    t: dict[str, float] = {"params_fp32": 4 * p / shard, "adam_fp32": 8 * p / shard}
    fsdp_no_sync = backend == "fsdp2" and accum > 1 and accum_sync == "last"
    if fsdp_no_sync:
        # No-sync micro-batches accumulate unsharded FP32 gradients for the whole model. The sharded
        # gradients were freed by zero_grad(set_to_none=True) and reappear only during the last
        # micro-batch's reduce-scatter, as the unsharded ones are consumed, so the two never coexist
        # at the peak. (Version 1 of this model counted both; corrected after Phase 3.)
        t["fsdp_unsharded_grads"] = 4 * p
    else:
        t["grads_fp32"] = 4 * p / shard
    if backend == "ddp" and world > 1 and not ddp_bucket_view:
        t["ddp_buckets"] = 4 * p  # gradient_as_bucket_view=False: buckets copy the gradients
    n_ck = len(range(0, s.n_layers, ac_every)) if activation_checkpointing else 0  # layers 0, k, 2k, ...
    plm, vh = s.layer_matmul_params(), s.vocab * s.hidden
    if backend == "fsdp2":
        root = s.vocab * s.hidden * (1 if s.tied_embeddings else 2) + s.hidden
        gathered = s.n_layers if not reshard_after_forward else max(2, 1 + fsdp_prefetch)
        t["fsdp_gathered_bf16"] = 2 * root + 2 * gathered * pl  # root + layers held unsharded
        t["fsdp_layer_grad"] = 6 * pl  # one layer's unsharded BF16 grad and its FP32 reduce copy
    else:  # BF16 weight copies saved for backward: non-checkpointed layers and the lm_head
        t["autocast_weights_bf16"] = 2 * ((s.n_layers - n_ck) * plm + vh)

    per_layer = activation_bytes_per_token_layer(s, backend)
    stream = 2 if backend == "fsdp2" else 4  # bytes per element of the residual stream
    saved = n_ck * stream * s.hidden + (s.n_layers - n_ck) * per_layer
    if activation_checkpointing:
        t["activations"] = tokens * (saved + per_layer)  # + one layer recomputed in backward
    else:
        t["activations"] = tokens * s.n_layers * per_layer
    t["final_norm"] = tokens * (8 if backend == "fsdp2" else 10) * s.hidden
    t["logits"] = 14 * s.vocab * (min(loss_chunk, tokens) if loss_chunk else tokens)
    backward_peak = sum(t.values())

    if backend != "fsdp2" and activation_checkpointing:
        # end of the forward: the autocast cache still holds every weight copy, the checkpointed layers
        # hold only their inputs, and the BF16 logits exist (cross-entropy runs after autocast exits)
        states = sum(
            v for k, v in t.items() if k in ("params_fp32", "adam_fp32", "grads_fp32", "ddp_buckets")
        )
        forward_peak = (
            states
            + 2 * (s.n_layers * plm + vh)
            + tokens * saved
            + t["final_norm"]
            + 2 * s.vocab * (min(loss_chunk, tokens) if loss_chunk else tokens)
        )
        if forward_peak > backward_peak:
            t = {
                **{
                    k: v
                    for k, v in t.items()
                    if k in ("params_fp32", "adam_fp32", "grads_fp32", "ddp_buckets")
                },
                "autocast_cache_bf16": 2 * (s.n_layers * plm + vh),
                "activations": tokens * saved,
                "final_norm": t["final_norm"],
                "logits": 2 * s.vocab * (min(loss_chunk, tokens) if loss_chunk else tokens),
            }
    t["total"] = sum(t.values())
    return t


VERSION = 3


def predict_gib(s: ModelShape, **kw) -> float:
    return predict(s, **kw)["total"] / GIB


def predict_for_config(s: ModelShape, cfg, world: int, accum: int) -> dict[str, float]:
    """``predict`` for a TrainConfig-like object (attribute access) running on ``world`` ranks."""
    backend = "single" if world == 1 and cfg.backend == "ddp" else cfg.backend
    return predict(
        s,
        backend=backend,
        world=world,
        seq_len=cfg.seq_len,
        micro_batch=cfg.micro_batch,
        accum=accum,
        activation_checkpointing=cfg.activation_checkpointing,
        accum_sync=cfg.accum_sync,
        ac_every=getattr(cfg, "ac_every", 1),
        loss_chunk=getattr(cfg, "loss_chunk", 0),
        ddp_bucket_view=getattr(cfg, "ddp_bucket_view", False),
        reshard_after_forward=getattr(cfg, "reshard_after_forward", True),
        fsdp_prefetch=getattr(cfg, "fsdp_prefetch", 0),
        fsdp_hold_params=getattr(cfg, "fsdp_hold_params", False),
    )
