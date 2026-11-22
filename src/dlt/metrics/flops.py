"""Parameter and FLOP counting for decoder-only transformers (the MFU convention in PLAN.md 4.3).

Per token, with N = matmul parameters (attention and MLP projections plus the lm_head, which costs a
matmul even when its weight is tied to the embedding), L layers, d = heads x head_dim, T = sequence:

    full training   6N + 12 L d T      (PaLM convention: full attention matrix, even when causal)
    QLoRA training  4N + 12 L d T      (frozen base: no weight-gradient GEMMs; attention backward kept)
    recompute       +2N + 4 L d T      (activation checkpointing replays the forward; HFU only)
"""

from __future__ import annotations

from dataclasses import dataclass

# Dense BF16 peak, TFLOPS, by the name torch reports. MFU is undefined for GPUs not listed here.
PEAK_BF16_TFLOPS = {
    "NVIDIA H200": 989.0,
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
}


@dataclass(frozen=True)
class ModelShape:
    n_layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool
    qkv_bias: bool = True

    @classmethod
    def from_hf_config(cls, cfg) -> ModelShape:
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(
            n_layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            intermediate=cfg.intermediate_size,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=head_dim,
            vocab=cfg.vocab_size,
            tied_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        )

    @property
    def attn_width(self) -> int:
        return self.n_heads * self.head_dim

    def layer_matmul_params(self) -> int:
        q = self.hidden * self.attn_width
        kv = 2 * self.hidden * self.n_kv_heads * self.head_dim
        o = self.attn_width * self.hidden
        mlp = 3 * self.hidden * self.intermediate
        return q + kv + o + mlp

    def matmul_params(self) -> int:
        """N in the FLOP formulas: every per-token weight matmul, lm_head included."""
        return self.n_layers * self.layer_matmul_params() + self.vocab * self.hidden

    def total_params(self) -> int:
        """Stored parameters (what a checkpoint holds), for checking against model cards."""
        bias = (self.attn_width + 2 * self.n_kv_heads * self.head_dim) if self.qkv_bias else 0
        norms = 2 * self.hidden
        per_layer = self.layer_matmul_params() + bias + norms
        embed = self.vocab * self.hidden
        lm_head = 0 if self.tied_embeddings else self.vocab * self.hidden
        return self.n_layers * per_layer + embed + lm_head + self.hidden


def train_flops_per_token(
    shape: ModelShape, seq_len: int, mode: str = "full", recompute: bool = False
) -> float:
    n = shape.matmul_params()
    attn = shape.n_layers * shape.attn_width * seq_len
    if mode == "full":
        flops = 6 * n + 12 * attn
    elif mode == "qlora":
        flops = 4 * n + 12 * attn
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if recompute:
        flops += 2 * n + 4 * attn
    return float(flops)


def mfu(flops_per_token: float, tokens_per_s: float, n_gpus: int, gpu_name: str | None) -> float | None:
    peak = PEAK_BF16_TFLOPS.get(gpu_name or "")
    if not peak:
        return None
    return flops_per_token * tokens_per_s / (peak * 1e12 * n_gpus)
