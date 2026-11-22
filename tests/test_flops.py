import pytest

from dlt.metrics.flops import ModelShape, mfu, train_flops_per_token

# Shapes from the pinned Hugging Face configs.
QWEN_72B = ModelShape(
    n_layers=80,
    hidden=8192,
    intermediate=29568,
    n_heads=64,
    n_kv_heads=8,
    head_dim=128,
    vocab=152064,
    tied_embeddings=False,
)
QWEN_05B = ModelShape(
    n_layers=24,
    hidden=896,
    intermediate=4864,
    n_heads=14,
    n_kv_heads=2,
    head_dim=64,
    vocab=151936,
    tied_embeddings=True,
)


def test_total_params_match_model_cards():
    assert QWEN_72B.total_params() == pytest.approx(72.7e9, rel=2e-3)
    assert QWEN_05B.total_params() == pytest.approx(0.494e9, rel=5e-3)


def test_matmul_params_count_lm_head_even_when_tied():
    assert QWEN_72B.matmul_params() == pytest.approx(71.46e9, rel=1e-3)
    tied = QWEN_05B.matmul_params()
    assert tied - QWEN_05B.n_layers * QWEN_05B.layer_matmul_params() == QWEN_05B.vocab * QWEN_05B.hidden


def test_flop_modes():
    full = train_flops_per_token(QWEN_72B, 4096, "full")
    qlora = train_flops_per_token(QWEN_72B, 4096, "qlora")
    n = QWEN_72B.matmul_params()
    attn = QWEN_72B.n_layers * QWEN_72B.attn_width * 4096
    assert full == 6 * n + 12 * attn
    assert qlora == 4 * n + 12 * attn
    assert qlora == pytest.approx(3.18e11, rel=5e-3)  # the PLAN.md 4.4 estimate
    assert train_flops_per_token(QWEN_72B, 4096, "full", recompute=True) == full + 2 * n + 4 * attn
    with pytest.raises(ValueError):
        train_flops_per_token(QWEN_72B, 4096, "lora")


def test_mfu():
    assert mfu(1e12, 989.0, 1, "NVIDIA H200") == pytest.approx(1.0)
    assert mfu(1e12, 989.0, 2, "NVIDIA H200") == pytest.approx(0.5)
    assert mfu(1e12, 1.0, 1, "unknown gpu") is None
