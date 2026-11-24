"""scripts/nf4_storage.py: a saved NF4 checkpoint re-stored as BF16 bytes loads and computes the same."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("bitsandbytes")


def test_bf16_storage_copy_matches_uint8_checkpoint(tmp_path):
    import nf4_storage
    from transformers import AutoModelForCausalLM, Qwen2Config

    from dlt.qlora.nf4 import bnb_config

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).save_pretrained(tmp_path / "bf16")
    cpu = {"": "cpu"}
    try:
        q = AutoModelForCausalLM.from_pretrained(
            tmp_path / "bf16", quantization_config=bnb_config(), dtype=torch.bfloat16, device_map=cpu
        )
    except Exception as e:  # noqa: BLE001 (bitsandbytes builds without a CPU backend)
        pytest.skip(f"NF4 on CPU unavailable: {e}")
    q.save_pretrained(tmp_path / "nf4")
    stats = nf4_storage.convert(tmp_path / "nf4", tmp_path / "nf4-bf16")
    assert stats["packed_weights"] == 2 * 7  # 7 projections per layer
    saved = json.loads((tmp_path / "nf4-bf16" / "config.json").read_text())["quantization_config"]
    assert saved["bnb_4bit_quant_storage"] == "bfloat16"

    a = AutoModelForCausalLM.from_pretrained(tmp_path / "nf4", dtype=torch.bfloat16, device_map=cpu)
    b = AutoModelForCausalLM.from_pretrained(tmp_path / "nf4-bf16", dtype=torch.bfloat16, device_map=cpu)
    wa, wb = a.model.layers[0].mlp.down_proj.weight, b.model.layers[0].mlp.down_proj.weight
    assert wb.dtype == torch.bfloat16 and wa.dtype == torch.uint8
    assert torch.equal(wa.data.view(-1), wb.data.view(-1).view(torch.uint8))  # the same packed bytes
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        assert torch.equal(a(x).logits, b(x).logits)
    with pytest.raises(ValueError):
        nf4_storage.convert(tmp_path / "nf4-bf16", tmp_path / "again")  # already BF16 storage
