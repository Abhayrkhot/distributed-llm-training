"""Kernel categories for the Nsight summary, on demangled names seen in H200 profiles."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from nsys_summary import category  # noqa: E402


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)", "nccl"),
        (
            "void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4>, false>",
            "attention",
        ),
        ("nvjet_tst_192x128_64x4_2x1_v_bz_coopB_TNT", "gemm"),
        ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>", "gemm"),
        (
            "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float>",
            "cross_entropy",
        ),
        (
            "void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous "
            "namespace)::TensorListMetadata<4>, "
            "at::native::(anonymous namespace)::FusedAdamMathFunctor<float>>",
            "optimizer",
        ),
        (
            "void at::native::multi_tensor_apply_kernel<at::native::TensorListMetadata<1>, "
            "at::native::(anonymous namespace)::LpNormFunctor<float, 2>, float*, int>",
            "grad_clip",
        ),
        ("void at::native::(anonymous namespace)::chunk_cat_cuda_kernel<c10::BFloat16>", "fsdp_copy"),
        (
            "void at::native::multi_tensor_apply_kernel<at::native::TensorListMetadata<2>, "
            "at::native::CopyFunctor<float, float>>",
            "foreach",
        ),
        (
            "void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float>>",
            "elementwise",
        ),
        (
            "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps>>",
            "reduce",
        ),
        ("void at::native::index_elementwise_kernel<128, 4>", "elementwise"),
        ("some_unknown_kernel", "other"),
    ],
)
def test_category(name, expected):
    assert category(name) == expected


def test_idle_gaps_split_host_bound_from_queued():
    from nsys_summary import idle_gaps

    # kernel A runs 0-10 us; B was launched at 5 us (queued) but starts at 30: a 20 us gap, none of it
    # host-bound. C is launched at 1,040 us, after the GPU went idle at 40: 1,000 us host-bound of a
    # 1,050 us gap. D overlaps C entirely and adds no gap.
    timeline = [(0.0, 10.0, 0.0), (30.0, 40.0, 5.0), (1090.0, 1100.0, 1040.0), (1091.0, 1095.0, None)]
    out = idle_gaps(timeline, n_steps=1)
    assert out["idle_s"] == round(1070 / 1e6, 5)
    assert out["host_bound_idle_s"] == round(1000 / 1e6, 5)
    assert out["idle_gaps_per_step"] == {
        "lt_10us": 0,
        "10_100us": 1,
        "100us_1ms": 0,
        "1_10ms": 1,
        "ge_10ms": 0,
    }
    assert out["idle_s_by_gap"]["1_10ms"] == round(1050 / 1e6, 5)


def test_idle_gaps_per_step_and_unordered_input():
    from nsys_summary import idle_gaps

    timeline = [(200.0, 210.0, 150.0), (0.0, 100.0, 0.0)]  # one 100 us gap, 50 us of it host-bound
    out = idle_gaps(timeline, n_steps=2)
    assert out["idle_s"] == round(100 / 2e6, 5)
    assert out["host_bound_idle_s"] == round(50 / 2e6, 5)
    assert out["idle_gaps_per_step"]["100us_1ms"] == 0.5
