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
) -> dict[str, float]:
    """Predicted peak bytes per rank, by term, plus ``total``."""
    if backend not in ("single", "ddp", "fsdp2"):
        raise ValueError(backend)
    p = s.total_params()
    pl = _layer_params(s)
    tokens = micro_batch * seq_len
    shard = world if backend == "fsdp2" else 1
    t: dict[str, float] = {
        "params_fp32": 4 * p / shard,
        "grads_fp32": 4 * p / shard,
        "adam_fp32": 8 * p / shard,
    }
    if backend == "ddp" and world > 1:
        t["ddp_buckets"] = 4 * p  # gradient_as_bucket_view=False: buckets copy the gradients
    if backend == "fsdp2":
        root = s.vocab * s.hidden * (1 if s.tied_embeddings else 2) + s.hidden
        t["fsdp_gathered_bf16"] = 2 * root + 2 * 2 * pl  # root + current and prefetched layer
        t["fsdp_layer_grad"] = 6 * pl  # one layer's unsharded BF16 grad and its FP32 reduce copy
        if accum > 1 and accum_sync == "last":
            t["fsdp_unsharded_grads"] = 4 * p  # no-sync micro-batches accumulate full FP32 grads
    else:
        t["autocast_weights_bf16"] = 2 * (s.n_layers * s.layer_matmul_params() + s.vocab * s.hidden)

    per_layer = activation_bytes_per_token_layer(s, backend)
    stream = 2 if backend == "fsdp2" else 4  # bytes per element of the residual stream
    if activation_checkpointing:
        t["activations"] = tokens * (s.n_layers * stream * s.hidden + per_layer)
    else:
        t["activations"] = tokens * s.n_layers * per_layer
    t["final_norm"] = tokens * (8 if backend == "fsdp2" else 10) * s.hidden
    t["logits"] = tokens * 14 * s.vocab
    t["total"] = sum(t.values())
    return t


def predict_gib(s: ModelShape, **kw) -> float:
    return predict(s, **kw)["total"] / GIB
