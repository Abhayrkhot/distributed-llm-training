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
