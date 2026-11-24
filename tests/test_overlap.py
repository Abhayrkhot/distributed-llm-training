from dlt.metrics.overlap import comm_overlap, host_wait, idle_gaps, kernel_launches, merge, minus


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


def test_idle_gaps_split_host_bound_from_queued():
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
    timeline = [(200.0, 210.0, 150.0), (0.0, 100.0, 0.0)]  # one 100 us gap, 50 us of it host-bound
    out = idle_gaps(timeline, n_steps=2)
    assert out["idle_s"] == round(100 / 2e6, 5)
    assert out["host_bound_idle_s"] == round(50 / 2e6, 5)
    assert out["idle_gaps_per_step"]["100us_1ms"] == 0.5


def _ev(cat, ts, dur, corr, name="k"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr}}


def test_kernel_launches_match_runtime_and_driver_calls():
    events = [
        _ev("cuda_runtime", 0.0, 3.0, 1, "cudaLaunchKernel"),
        _ev("kernel", 5.0, 10.0, 1),
        _ev("cuda_driver", 40.0, 2.0, 2, "cuLaunchKernelEx"),  # NCCL launches through the driver API
        _ev("kernel", 45.0, 5.0, 2, "ncclDevKernel_AllGather"),
        _ev("kernel", 60.0, 5.0, 3),  # no launch event in the trace
        {"ph": "i", "cat": "cpu_op", "name": "aten::mm", "ts": 1.0},
    ]
    assert kernel_launches(events) == [(5.0, 15.0, 0.0), (45.0, 50.0, 40.0), (60.0, 65.0, None)]
    out = host_wait(events, n_steps=1)
    # gap 15-45: the driver call starts at 40, so 25 us host-bound; gap 50-60: no launch event, 0
    assert out["host_bound_idle_s"] == round(25 / 1e6, 5)
    assert out["idle_gaps_per_step"] == 2.0
    assert out["kernels_per_step"] == 3
    assert abs(out["launch_matched_frac"] - 2 / 3) < 1e-12


def test_host_wait_without_launch_events_reports_zero_coverage():
    out = host_wait([_ev("kernel", 0.0, 1.0, 1), _ev("kernel", 5.0, 1.0, 2)], n_steps=1)
    assert out["launch_matched_frac"] == 0.0
    assert out["host_bound_idle_s"] == 0.0
