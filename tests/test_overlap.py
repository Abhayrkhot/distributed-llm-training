from dlt.metrics.overlap import comm_overlap, merge, minus


def test_merge_and_minus():
    assert merge([(5, 7), (0, 2), (1, 3), (7, 8)]) == [(0, 3), (5, 8)]
    assert minus([(0, 10)], [(2, 3), (5, 7)]) == 7
    assert minus([(0, 2), (4, 6)], [(1, 5)]) == 2
    assert minus([(0, 1)], []) == 1


def test_comm_overlap_counts_only_uncovered_nccl_time():
    kernels = [
        ("sm90_xmma_gemm", 0, 100),
        ("ncclDevKernel_AllGather_RING_LL", 50, 150),  # 50 us hidden behind the GEMM, 50 exposed
        ("flash_fwd_kernel", 150, 200),
        ("ncclDevKernel_ReduceScatter", 210, 230),  # fully exposed
    ]
    r = comm_overlap(kernels, n_steps=2)
    assert r["comm_s"] == 120 / 1e6 / 2
    assert r["exposed_comm_s"] == 70 / 1e6 / 2
    assert r["compute_s"] == 150 / 1e6 / 2
    assert abs(r["overlapped_frac"] - 50 / 120) < 1e-12
