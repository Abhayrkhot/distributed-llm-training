"""Regenerate the README's measured sections from manifests/ and results/.

Sections sit between ``<!-- gen:NAME -->`` and ``<!-- /gen:NAME -->`` markers. ``--check`` exits 1 if
the README is out of date (CI runs this), so no number in the README can drift from its source file.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dlt.metrics.flops import QWEN25  # noqa: E402
from dlt.metrics.memory import VERSION as PREDICTOR_VERSION  # noqa: E402
from dlt.metrics.memory import predict_for_config  # noqa: E402
from dlt.results import load_records  # noqa: E402


def _fmt_bytes(n: int) -> str:
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= div:
            return f"{n / div:g} {unit}"
    return f"{n} B"


def hardware_section() -> str:
    path = ROOT / "manifests" / "h200.json"
    if not path.exists():
        return "_No hardware manifest yet._"
    m = json.loads(path.read_text())
    s = m["summary"]
    p2p = s["p2p"][0] if s.get("p2p") else {}
    rows = [
        ("GPU", ", ".join(s["gpu_names"])),
        ("Driver", ", ".join(s["driver_versions"])),
        ("GPU-GPU link", s.get("pair_link_gpu0_gpu1") or "n/a"),
        ("BF16 GEMM 8192^3 (median TFLOPS)", f"{s['gemm_bf16_8192_tflops']['median']:.0f}"),
        ("Device copy, read + write (median GB/s)", f"{s['d2d_copy_gbps']['median']:,.0f}"),
        ("Peer copy GPU0 to GPU1 (GB/s)", f"{p2p.get('copy_0_to_1_gbps', 'n/a')}"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    nodes = ", ".join(s["nodes_probed"])
    head = f"Probed on {nodes} ([manifests/h200.json](manifests/h200.json))."
    return f"{head}\n\n| Item | Value |\n|---|---|\n{body}"


def nccl_section() -> str:
    recs = load_records(ROOT / "results", "nccl_bw")
    if not recs:
        return "_No NCCL runs yet._"
    rec = recs[-1]
    rows = rec["metrics"]["rows"]
    sizes = sorted({r["bytes"] for r in rows})
    ops = list(dict.fromkeys(r["op"] for r in rows))
    by = {(r["op"], r["bytes"]): r for r in rows}
    head = "| Size | " + " | ".join(f"{op} busbw (GB/s)" for op in ops) + " |"
    sep = "|---|" + "---|" * len(ops)
    lines = [head, sep]
    for size in sizes:
        cells = [f"{by[(op, size)]['busbw_gbps']:.1f}" if (op, size) in by else "" for op in ops]
        lines.append(f"| {_fmt_bytes(size)} | " + " | ".join(cells) + " |")
    env = rec["env"]
    note = (
        f"{rec['summary']['world_size']} GPUs on {rec['slurm']['host']}, BF16, torch {env['torch']}, "
        f"NCCL {env.get('nccl_runtime')}; median of {rec['config']['repeats']} timed repeats "
        f"(`{rec['run_id']}`)."
    )
    return note + "\n\n" + "\n".join(lines)


def smoke_section() -> str:
    recs = load_records(ROOT / "results", "smoke_train")
    if not recs:
        return "_No smoke runs yet._"
    lines = [
        "| Run | Model | GPUs | Seq x micro-batch | Steps | Loss first -> last | Step (s) | Tokens/s | MFU | "
        "Peak alloc (GiB/rank) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        s, c = r["summary"], r["config"]
        stop = (
            f" (USR1 at {s['stopped_by_signal_at_step']})"
            if s.get("stopped_by_signal_at_step") is not None
            else ""
        )
        mfu = f"{100 * s['mfu']:.1f}%" if s.get("mfu") is not None else "n/a"
        mem = ", ".join(f"{m:.1f}" for m in s["peak_alloc_gib_by_rank"])
        cells = [
            f"`{r['run_id']}`",
            s["model"].split("/")[-1],
            str(s["world_size"]),
            f"{c['seq_len']} x {c['micro_batch']}",
            f"{s['steps_run']}{stop}",
            f"{s['loss_first']:.2f} -> {s['loss_last']:.2f}",
            str(s["step_time_s_median"]),
            f"{s['tokens_per_s']:,.0f}",
            mfu,
            mem,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _latest_by_name(kind: str = "train") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in load_records(ROOT / "results", kind):  # sorted by creation time: the latest wins
        name = r["summary"].get("run_name")
        if name:
            out[name] = r
    return out


def _gib(xs: list[float]) -> str:
    return f"{max(xs):.1f}"


def correctness_section() -> str:
    recs = _latest_by_name()
    names = ["parity-single", "parity-ddp", "parity-fsdp2"]
    if not all(n in recs for n in names):
        return "_No correctness runs yet._"
    ref = recs["parity-single"]["metrics"]["loss"]
    cfg = recs["parity-single"]["config"]
    lines = [
        f"Parity: {cfg['model'].split('/')[-1]}, {len(ref)} steps, global batch {cfg['global_batch']} x "
        f"{cfg['seq_len']}, micro-batch {cfg['micro_batch']}, same data order. Difference is per-step loss "
        "against the single-GPU run.",
        "",
        "| Run | GPUs | Accum | Loss step 1 -> last | Max abs diff | Max rel diff | Step (s) | Tokens/s | "
        "MFU | "
        "Peak alloc (GiB) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in names:
        r = recs[n]
        s, loss = r["summary"], r["metrics"]["loss"]
        d = [abs(a - b) for a, b in zip(loss, ref, strict=True)]
        rel = [abs(a - b) / abs(b) for a, b in zip(loss, ref, strict=True)]
        diffs = "reference | reference" if n == "parity-single" else f"{max(d):.1e} | {max(rel):.1e}"
        lines.append(
            f"| {s['backend']} | {s['world_size']} | {s['accum_steps']} | {loss[0]:.4f} -> {loss[-1]:.4f} | "
            f"{diffs} | {s['step_time_s_median']} | {s['tokens_per_s']:,.0f} | "
            f"{100 * s['mfu']:.1f}% | {_gib(s['peak_alloc_gib_by_rank'])} |"
        )
    lines += [
        "",
        "Resume and crash recovery: deterministic mode, eager attention, 20 steps of 1,024 tokens. "
        '"Stop + resume" stops at step 10 with a checkpoint and resumes in a new process; "SIGKILL" '
        "kills every rank mid-run with checkpoints every 5 steps and restarts from the latest. Bit-identical "
        "means every later per-step loss and gradient norm equals the uninterrupted run's.",
        "",
        "| Backend | Stop + resume | SIGKILL, resumed from | Recovery | Checkpoint | Save (s) | "
        "Restore (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in ("ddp", "fsdp2"):
        keys = [f"det-{b}-{k}" for k in ("full", "first", "second", "recovered")]
        if not all(k in recs for k in keys):
            continue
        full, first, second, rec = (recs[k] for k in keys)
        fm = full["metrics"]
        resumed = (first["metrics"]["loss"] + second["metrics"]["loss"] == fm["loss"]) and (
            first["metrics"]["grad_norm"] + second["metrics"]["grad_norm"] == fm["grad_norm"]
        )
        k = rec["summary"]["start_step"]
        recovered = (
            rec["metrics"]["loss"] == fm["loss"][k:] and rec["metrics"]["grad_norm"] == fm["grad_norm"][k:]
        )
        saves = [c for r in (first, second, rec) for c in r["summary"]["checkpoints"]]
        save_s = sorted(c["total_s"] for c in saves)
        restores = [r["summary"]["restore_s"] for r in (second, rec)]
        lines.append(
            f"| {b} | {'bit-identical' if resumed else 'DIFFERS'} | step {k} | "
            f"{'bit-identical' if recovered else 'DIFFERS'} | {saves[0]['bytes'] / 2**30:.2f} GiB | "
            f"{save_s[0]:.1f}-{save_s[-1]:.1f} | {min(restores):.1f}-{max(restores):.1f} |"
        )
    return "\n".join(lines)


def chain_section() -> str:
    links = [r for r in load_records(ROOT / "results", "train") if r["summary"].get("run_name") == "p2-chain"]
    if not links:
        return "_No chained run yet._"
    c = links[0]["config"]
    lines = [
        f"{c['model'].split('/')[-1]}, {c['backend']} on {links[0]['summary']['world_size']} GPUs, "
        f"{c['steps']:,} steps x {c['global_batch'] * c['seq_len']:,} tokens, validation loss on the first "
        f"{c['eval_samples']} held-out sequences.",
        "",
        "| Slurm job | Steps | Stopped by | Resumed (restore s) | Checkpoints (step: s) | Tokens/s | "
        "Val loss start -> end |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in links:
        s = r["summary"]
        ck = ", ".join(f"{x['step']}: {x['total_s']:.0f}" for x in s["checkpoints"])
        res = f"{s['resumed_from']} ({s['restore_s']:.0f})" if s["resumed_from"] else "fresh"
        v0 = f"{s['val_loss_start']:.4f}" if s.get("val_loss_start") is not None else "-"
        v1 = f"{s['val_loss_end']:.4f}" if s.get("val_loss_end") is not None else "-"
        lines.append(
            f"| {r['slurm']['job_id']} | {s['start_step']}-{s['end_step']} | "
            f"{s['stopped_by'] or 'finished'} | "
            f"{res} | {ck} | {s['tokens_per_s']:,.0f} | {v0} -> {v1} |"
        )
    return "\n".join(lines)


H200_GIB = 139.8  # visible to PyTorch per GPU (docs/phase0.md)


def _p3_entries(prefix: str = "p3.") -> dict[str, list[dict]]:
    """Sweep entry -> finished records (status ok, not stopped early) and OOM records, in file order."""
    out: dict[str, list[dict]] = {}
    for r in load_records(ROOT / "results", "train"):
        s = r["summary"]
        name = s.get("run_name") or ""
        if not name.startswith(prefix):
            continue
        status = s.get("status", "ok")
        if status == "failed" or (status == "ok" and s.get("stopped_by") is not None):
            continue  # failed runs are logged (docs/RESULTS_LOG.md) but carry no measurements
        entry = name.split(".")[1]
        if status == "oom" and any(x["summary"].get("status") == "oom" for x in out.get(entry, [])):
            continue  # one OOM record per entry (each rank may write one)
        out.setdefault(entry, []).append(r)
    return out


def _ok(recs: list[dict]) -> list[dict]:
    return [r for r in recs if r["summary"].get("status", "ok") == "ok"]


def _med(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _spread(xs: list[float], fmt: str) -> str:
    if len(xs) == 1:
        return format(xs[0], fmt)
    return f"{format(_med(xs), fmt)} ({format(min(xs), fmt)}-{format(max(xs), fmt)})"


def _model(r: dict) -> str:
    return r["config"]["model"].split("-")[-1]


def _exposed(r: dict) -> float | None:
    c = r["summary"].get("comm_by_rank")
    return max(x["exposed_comm_s"] for x in c) if c else None


def ladder_section() -> str:
    entries = _p3_entries()
    rows = [(e, _ok(v)) for e, v in entries.items() if _ok(v)]
    if not rows:
        return "_No Phase 3 runs yet._"
    lines = [
        "Median over repeats, (min-max) where there are several. Step split and exposed communication are "
        "per optimizer step; exposed communication is NCCL kernel time with no compute kernel running, "
        "from a 2-step profiler window (max over ranks).",
        "",
        "| Run | Model | GPUs | Backend | AC | Micro x accum | Repeats | Tokens/s | MFU | HFU | "
        "Fwd / bwd / opt (s) | Exposed comm (s) | Peak alloc (GiB) | Host RSS (GiB) | Load (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry, recs in rows:
        s0, c0 = recs[0]["summary"], recs[0]["config"]
        tps = [r["summary"]["tokens_per_s"] for r in recs]
        mfu = [100 * r["summary"]["mfu"] for r in recs]
        hfu = [100 * r["summary"]["hfu"] for r in recs]
        split = "/".join(
            f"{_med([r['summary'][k] for r in recs]):.2f}"
            for k in ("fwd_s_median", "bwd_s_median", "opt_s_median")
        )
        exp = [x for r in recs if (x := _exposed(r)) is not None]
        backend = s0["backend"] + (
            " (sync every)" if s0.get("accum_sync") == "every" and s0["world_size"] > 1 else ""
        )
        lines.append(
            f"| {entry} | {_model(recs[0])} | {s0['world_size']} | {backend} | "
            f"{'on' if s0['activation_checkpointing'] else 'off'} | "
            f"{c0['micro_batch']} x {s0['accum_steps']} | "
            f"{len(recs)} | {_spread(tps, ',.0f')} | {_spread(mfu, '.1f')}% | {_med(hfu):.1f}% | {split} | "
            f"{_spread(exp, '.3f') if exp else '-'} | "
            f"{_med([max(r['summary']['peak_alloc_gib_by_rank']) for r in recs]):.1f} | "
            f"{_med([max(r['summary']['host_rss_gib_by_rank']) for r in recs]):.1f} | "
            f"{_med([max(r['summary']['load_s_by_rank']) for r in recs]):.0f} |"
        )
    return "\n".join(lines)


def scaling_section() -> str:
    entries = {e: _ok(v) for e, v in _p3_entries().items() if _ok(v)}
    ones = {e: v for e, v in entries.items() if v[0]["summary"]["world_size"] == 1}
    twos = {e: v for e, v in entries.items() if v[0]["summary"]["world_size"] == 2}
    if not twos:
        return "_No Phase 3 runs yet._"
    lines = [
        "Scaling efficiency = tokens/s on 2 GPUs / (2 x tokens/s on 1 GPU), same model, data, sequence "
        "length, global batch, activation checkpointing and precision mechanism (medians over repeats). "
        "DDP is compared with the single-GPU run (both BF16 autocast over FP32 weights); FSDP2 with FSDP2 on "
        "1 GPU (BF16 parameter casts). A row marked * compares FSDP2 against the autocast single-GPU run "
        "because no 1-GPU FSDP2 run exists, which mixes two precision mechanisms.",
        "",
        "| Model | 1-GPU run | 2-GPU run | 1 GPU tokens/s | 2 GPU tokens/s | Efficiency |",
        "|---|---|---|---|---|---|",
    ]
    for e2, v2 in twos.items():
        c2 = v2[0]["config"]
        same = [
            e1
            for e1, v1 in ones.items()
            if v1[0]["config"]["model"] == c2["model"]
            and v1[0]["config"]["activation_checkpointing"] == c2["activation_checkpointing"]
            and v1[0]["config"].get("accum_sync", "last") == c2.get("accum_sync", "last")
        ]
        want = "fsdp2" if c2["backend"] == "fsdp2" else "single"
        exact = [e for e in same if ones[e][0]["config"]["backend"] == want]
        fallback = [e for e in same if ones[e][0]["config"]["backend"] == "single"]
        base, mark = (exact[0], "") if exact else ((fallback[0], " *") if fallback else (None, ""))
        t2 = _med([r["summary"]["tokens_per_s"] for r in v2])
        if base is None:
            mfu = _med([100 * r["summary"]["mfu"] for r in v2])
            lines.append(
                f"| {_model(v2[0])} | none (does not fit on 1 GPU) | {e2} | - | {t2:,.0f} | "
                f"n/a; MFU {mfu:.1f}% |"
            )
            continue
        t1 = _med([r["summary"]["tokens_per_s"] for r in ones[base]])
        lines.append(
            f"| {_model(v2[0])} | {base} | {e2} | {t1:,.0f} | {t2:,.0f} | {100 * t2 / (2 * t1):.1f}%{mark} |"
        )
    return "\n".join(lines)


def _predict_now(r: dict) -> float:
    """Today's predictor (dlt.metrics.memory.VERSION) for the config a record ran."""
    from types import SimpleNamespace

    s, c = r["summary"], SimpleNamespace(**r["config"])
    world = s["world_size"]
    accum = s.get("accum_steps") or c.global_batch // (world * c.micro_batch)
    return predict_for_config(QWEN25[c.model], c, world, accum)["total"] / 2**30


def _stats(errs: list[float]) -> str:
    a = sorted(abs(x) for x in errs)
    return f"median absolute error {100 * _med(a):.1f}%, worst {100 * a[-1]:.1f}%"


def memory_section() -> str:
    entries = _p3_entries()
    if not entries:
        return "_No Phase 3 runs yet._"
    lines = [
        "Predicted peak allocated memory per rank (dlt.metrics.memory, analytical, not fitted) against the "
        f"measured `max_memory_allocated` (max over ranks), for every run including OOMs. Limit: {H200_GIB} "
        f'GiB visible per H200. "Now" is predictor version {PREDICTOR_VERSION}; "at run time" is the '
        "version the run recorded (version 1 double-counted FSDP2 gradients under no-sync accumulation; "
        "version 2 kept autocast weight copies until backward under checkpointing).",
        "",
        "| Run | Predicted now (GiB) | At run time (GiB) | Measured (GiB) | Error now | Outcome |",
        "|---|---|---|---|---|---|",
    ]
    now_errs, then_errs, right, total = [], [], 0, 0
    for entry, recs in entries.items():
        s = recs[0]["summary"]
        now = _predict_now(recs[0])
        if s.get("status") == "oom":
            total += 1
            right += now > H200_GIB
            lines.append(
                f"| {entry} | {now:.1f} | {s['predicted_gib']['total']:.1f} | "
                f"OOM during {s.get('phase')} | - | "
                f"predicted {'OOM' if now > H200_GIB else 'fit'} |"
            )
            continue
        meas = [max(r["summary"]["peak_alloc_gib_by_rank"]) for r in recs]
        then = [r["summary"]["predicted_gib"] for r in recs]
        e = [(now - m) / m for m in meas]
        now_errs += e
        then_errs += [(p - m) / m for p, m in zip(then, meas, strict=True)]
        total += 1
        right += now <= H200_GIB
        lines.append(
            f"| {entry} | {now:.1f} | {_med(then):.1f} | {_med(meas):.1f} | {100 * _med(e):+.1f}% | "
            f"fit (predicted {'fit' if now <= H200_GIB else 'OOM'}) |"
        )
    p4 = [r for v in _p3_entries("p4.").values() for r in _ok(v)]
    if p4:
        e4 = [
            (_predict_now(r) - max(r["summary"]["peak_alloc_gib_by_rank"]))
            / max(r["summary"]["peak_alloc_gib_by_rank"])
            for r in p4
        ]
        worst = max(
            zip((abs(x) for x in e4), (r["summary"]["run_name"].split(".")[1] for r in p4), strict=True)
        )
        lines.append(
            f"\nOut of sample, on the {len(p4)} Phase 4 runs (configurations the corrections never saw: "
            f"larger micro-batches, chunked loss, longer sequences, other kernels): {_stats(e4)} "
            f"(worst: {worst[1]})."
        )
    if now_errs:
        lines.insert(
            1,
            f"\nOver {len(now_errs)} finished runs: {_stats(now_errs)} "
            f"(at run time: {_stats(then_errs)}); fit "
            f"vs OOM predicted correctly for {right} of {total} configurations.",
        )
    return "\n".join(lines)


def ac_section() -> str:
    entries = {e: _ok(v) for e, v in _p3_entries().items() if _ok(v)}
    pairs = []
    for v_on in entries.values():
        c_on, s_on = v_on[0]["config"], v_on[0]["summary"]
        if not c_on["activation_checkpointing"]:
            continue
        for v_off in entries.values():
            c_off, s_off = v_off[0]["config"], v_off[0]["summary"]
            same = all(c_off[k] == c_on[k] for k in ("model", "backend", "micro_batch", "accum_sync"))
            if same and s_off["world_size"] == s_on["world_size"] and not c_off["activation_checkpointing"]:
                pairs.append((v_off, v_on))
    if not pairs:
        return "_No activation checkpointing pairs yet._"
    lines = [
        "Same model, backend, GPUs and micro-batch, checkpointing every decoder layer on vs off (medians).",
        "",
        "| Model | GPUs | Backend | Peak off -> on (GiB) | Memory saved | Tokens/s off -> on | "
        "Throughput cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for v_off, v_on in pairs:
        m0 = _med([max(r["summary"]["peak_alloc_gib_by_rank"]) for r in v_off])
        m1 = _med([max(r["summary"]["peak_alloc_gib_by_rank"]) for r in v_on])
        t0 = _med([r["summary"]["tokens_per_s"] for r in v_off])
        t1 = _med([r["summary"]["tokens_per_s"] for r in v_on])
        s = v_off[0]["summary"]
        lines.append(
            f"| {_model(v_off[0])} | {s['world_size']} | {s['backend']} | {m0:.1f} -> {m1:.1f} | "
            f"{100 * (m0 - m1) / m0:.0f}% | {t0:,.0f} -> {t1:,.0f} | {100 * (t0 - t1) / t0:.1f}% |"
        )
    return "\n".join(lines)


def _idle(r: dict) -> float | None:
    c = r["summary"].get("comm_by_rank")
    vals = [x["gpu_idle_s"] for x in c if "gpu_idle_s" in x] if c else []
    return max(vals) if vals else None


_ROUNDS = {
    "p4": "one change at a time, configs/sweeps/p4_tuning.yaml",
    "p4b": "the two fixes the Nsight profiles pointed to, configs/sweeps/p4b_fusion.yaml",
    "p4c": "the changes that paid off, combined, configs/sweeps/p4c_combined.yaml",
}


def tuning_section() -> str:
    rounds = [(pre.rstrip("."), _p3_entries(pre)) for pre in ("p4.", "p4b.", "p4c.")]
    rounds = [(label, e) for label, e in rounds if e]
    if not rounds:
        return "_No Phase 4 runs yet._"
    head = (
        "Each change against its workload's baseline, both run 3 times in the same sweep. Change is the "
        'median tokens/s against the baseline median; "noise" marks a change whose min-max range overlaps '
        "the baseline's. GPU idle is time inside the profiled steps with no kernel running."
    )
    return "\n\n".join([head] + [_tuning_round(label, entries) for label, entries in rounds])


def _tuning_round(label: str, entries: dict) -> str:
    lines = [
        f"**Round {label}**: {_ROUNDS.get(label, '')}.",
        "",
        "| Run | Tokens/s | Change | MFU | Peak alloc (GiB) | Fwd / bwd / opt (s) | Exposed comm (s) | "
        "GPU idle (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for workload in dict.fromkeys(e.split("-")[0] for e in entries):
        base = _ok(entries.get(f"{workload}-base", []))
        if not base:
            continue
        bt = [r["summary"]["tokens_per_s"] for r in base]
        for entry, recs in entries.items():
            if entry.split("-")[0] != workload:
                continue
            ok = _ok(recs)
            if not ok:
                s = recs[0]["summary"]
                lines.append(f"| {entry} | OOM during {s.get('phase')} | - | - | - | - | - | - |")
                continue
            t = [r["summary"]["tokens_per_s"] for r in ok]
            if entry.endswith("-base"):
                change = "baseline"
            else:
                d = 100 * (_med(t) - _med(bt)) / _med(bt)
                noise = max(min(t), min(bt)) <= min(max(t), max(bt))
                change = f"{d:+.1f}%" + (" (noise)" if noise else "")
            split = "/".join(
                f"{_med([r['summary'][k] for r in ok]):.2f}"
                for k in ("fwd_s_median", "bwd_s_median", "opt_s_median")
            )
            exp = [x for r in ok if (x := _exposed(r)) is not None]
            idle = [x for r in ok if (x := _idle(r)) is not None]
            mfu = _med([100 * r["summary"]["mfu"] for r in ok])
            peak = _med([max(r["summary"]["peak_alloc_gib_by_rank"]) for r in ok])
            exp_txt = f"{_med(exp):.3f}" if exp else "-"
            idle_txt = f"{_med(idle):.3f}" if idle else "-"
            lines.append(
                f"| {entry} | {_spread(t, ',.0f')} | {change} | {mfu:.1f}% | {peak:.1f} | {split} | "
                f"{exp_txt} | {idle_txt} |"
            )
    return "\n".join(lines)


def profile_section() -> str:
    recs = [r for r in load_records(ROOT / "results", "profile") if r["summary"].get("rank") == 0]
    if not recs:
        return "_No Nsight Systems profiles yet._"
    groups = {  # README columns -> nsys_summary categories
        "GEMM": ("gemm",),
        "attention": ("attention",),
        "NCCL": ("nccl",),
        "elementwise + reduce": ("elementwise", "reduce"),
        "cross-entropy": ("cross_entropy",),
        "optimizer + clip": ("optimizer", "grad_clip", "foreach"),
        "FSDP2 copies": ("fsdp_copy",),
        "other": ("other",),
    }
    lines = [
        "Nsight Systems, rank 0, GPU kernel time per optimizer step by category (share of kernel time), "
        "over the captured steps ([scripts/nsys_summary.py](scripts/nsys_summary.py)). NCCL kernel time "
        "includes time spent waiting for the other rank.",
        "",
        "| Run | " + " | ".join(groups) + " | Exposed comm (s) | GPU idle (s) |",
        "|---|" + "---|" * (len(groups) + 2),
    ]
    for r in recs:
        s = r["summary"]
        cells = []
        for cats in groups.values():
            sec = sum(s["kernel_s_per_step"].get(c, 0) for c in cats)
            share = sum(s["kernel_share"].get(c, 0) for c in cats)
            cells.append(f"{sec:.3f} ({100 * share:.0f}%)")
        lines.append(
            f"| {s['name']} | "
            + " | ".join(cells)
            + f" | {s['exposed_comm_s']:.3f} | {s['gpu_idle_s']:.3f} |"
        )
    return "\n".join(lines)


def _by_name(kind: str, key: str = "name") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in load_records(ROOT / "results", kind):
        n = r["summary"].get(key)
        if n:
            out[n] = r
    return out


def nf4_section() -> str:
    bench, ev = _by_name("bench"), _by_name("eval")
    q = bench.get("nf4-quantize")
    if not q:
        return "_No 72B NF4 runs yet._"
    s = q["summary"]
    tk = s["tokenizer"]
    lines = [
        f"Qwen2.5-72B at `{s['revision'][:10]}`. Tokenizer: special-token ids "
        f"{'correct' if tk['special_ok'] else 'WRONG'}, {tk['roundtrip_failures']} round-trip failures over "
        f"{tk['roundtrip_texts']} example texts, vocabulary "
        f"{'identical to' if tk['vocab_matches_0p5b'] else 'DIFFERENT from'}"
        " Qwen2.5-0.5B's.",
        "",
        "| Load path | GPUs | Load (s) | Peak GPU memory (GiB) | Host RSS (GiB) |",
        "|---|---|---|---|---|",
        f"| BF16 checkpoint, quantized to NF4 while loading | 1 | {s['quantize_on_load_s']:.0f} | "
        f"{s['peak_gpu_gib_quantize'][0]:.1f} | {s['host_rss_gib']:.1f} |",
        f"| saved NF4 checkpoint ({s['saved_gib']:.1f} GiB on disk) | 1 | {s['reload_saved_s']:.0f} | "
        f"{s['peak_gpu_gib_reload'][0]:.1f} | - |",
    ]
    for name, label in (
        ("logits-bf16", "BF16, layers split over 2 GPUs"),
        ("logits-nf4-saved", "NF4 forward pass"),
    ):
        r = bench.get(name)
        if r:
            b = r["summary"]
            peaks = ", ".join(f"{x:.1f}" for x in b["peak_gpu_gib"])
            lines.append(f"| {label} | {b['gpus']} | {b['load_s']:.0f} | {peaks} | {b['host_rss_gib']:.1f} |")
    c = ev.get("nf4-vs-bf16")
    if c:
        cs = c["summary"]
        lines += [
            "",
            f"NF4 vs BF16, teacher-forced over {cs['examples']} validation questions ({cs['positions']:,} "
            f"positions, {cs['sql_positions']:,} in the SQL): the argmax next token agrees at "
            f"{100 * cs['argmax_agreement_all']:.2f}% of positions ({100 * cs['argmax_agreement_sql']:.2f}% "
            f"in the SQL); perplexity of the gold SQL {cs['ppl_sql']['a']:.4f} (NF4) vs "
            f"{cs['ppl_sql']['b']:.4f} (BF16), a {100 * cs['ppl_sql_gap']:+.2f}% gap "
            f"({100 * cs['ppl_all_gap']:+.2f}% over all tokens).",
        ]
    return "\n".join(lines)


def _correct_by_qid(r: dict) -> dict[int, int]:
    m = r["metrics"]
    return dict(zip(m["question_id"], m["correct"], strict=True))


def bird_section() -> str:
    evs = [
        r
        for r in load_records(ROOT / "results", "eval")
        if r["summary"].get("mode") in ("base", "finetuned", "instruct")
    ]
    if not evs:
        return "_No BIRD evaluations yet._"
    import sys as _sys

    _sys.path.insert(0, str(ROOT / "src"))
    from dlt.bird.execute import paired_bootstrap_gain

    latest = {}
    for r in evs:
        latest[(r["summary"]["mode"], r["summary"]["split"], r["summary"]["name"])] = r
    lines = [
        "Execution accuracy (EX): the predicted SQL returns the same set of rows as the gold SQL on the "
        "database (BIRD's official criterion), 95% bootstrap CI over questions. Same schema text, greedy "
        "decoding and scorer for every model; the base model gets 3 worked examples from other databases, "
        "the fine-tuned model none; the Instruct model (NF4, chat template) is a reference, not a baseline. "
        '"val" is the database-level holdout from BIRD train; "dev" is BIRD dev (1106 split).',
        "",
        "| Run | Model | Split | Questions | EX | 95% CI | Simple / moderate / challenging |",
        "|---|---|---|---|---|---|---|",
    ]
    for (mode, split, name), r in sorted(latest.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        s = r["summary"]
        diff = s.get("by_difficulty", {})
        d = (
            " / ".join(
                f"{100 * diff[k]['ex']:.1f}" for k in ("simple", "moderate", "challenging") if k in diff
            )
            or "-"
        )
        lines.append(
            f"| {name} | {mode} | {split} | {s['questions']} | {100 * s['ex']:.1f}% | "
            f"{100 * s['ex_ci95'][0]:.1f}-{100 * s['ex_ci95'][1]:.1f} | {d} |"
        )

    def first(mode: str, split: str) -> dict | None:
        rs = [
            r
            for (m, sp, n), r in sorted(latest.items())
            if m == mode and sp == split and not n.endswith("-repro")
        ]
        return rs[0] if rs else None

    for split in ("val", "dev"):
        tuned = first("finetuned", split)
        for mode, label in (("base", "base"), ("instruct", "Instruct (reference)")):
            other = first(mode, split)
            if not (tuned and other):
                continue
            a, b = _correct_by_qid(other), _correct_by_qid(tuned)
            q = sorted(set(a) & set(b))
            gain, lo, hi = paired_bootstrap_gain([bool(a[i]) for i in q], [bool(b[i]) for i in q])
            lines.append(
                f"\nFine-tuned minus {label} on {split} ({len(q)} questions, paired bootstrap): "
                f"{100 * gain:+.1f} points, 95% CI {100 * lo:+.1f} to {100 * hi:+.1f}."
            )
    for (_, split, name), r in sorted(latest.items()):
        orig = next((v for (_, sp, n), v in latest.items() if sp == split and n + "-repro" == name), None)
        if orig:
            a, b = _correct_by_qid(orig), _correct_by_qid(r)
            q = sorted(set(a) & set(b))
            same = sum(a[i] == b[i] for i in q)
            lines.append(
                f"\nRepeat of {orig['summary']['name']} in a fresh process ({name}): "
                f"{same} of {len(q)} questions scored the same."
            )
    return "\n".join(lines)


def qlora_section() -> str:
    recs = [r for r in load_records(ROOT / "results", "train") if r["summary"].get("peft")]
    if not recs:
        return "_No 72B QLoRA training runs yet._"
    lines = [
        "72B QLoRA training runs (one record per Slurm job). Tokens are real tokens (prompt + SQL, no "
        "padding); MFU uses the QLoRA convention 4N + 12LdT (PLAN.md 4.3).",
        "",
        "| Run | Backend x GPUs | r | Steps | Stopped by | Loss first -> last | Val loss | Tokens/s | MFU | "
        "Peak GPU (GiB) | Audit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        s, c = r["summary"], r["config"]
        a = s.get("adapter_audit") or {}
        audit = (
            f"{a.get('trainable_params', 0) / 1e6:.1f}M LoRA params, base unchanged"
            if a.get("frozen_unchanged_after_step")
            else "-"
        )
        vl = s.get("val_loss_end")
        lines.append(
            f"| {s['run_name']} | {s['backend']} x {s['world_size']} | {c['lora_r']} | "
            f"{s['start_step']}-{s['end_step']} | "
            f"{s['stopped_by'] or 'finished'} | {s['loss_first']:.3f} -> {s['loss_last']:.3f} | "
            f"{f'{vl:.4f}' if vl is not None else '-'} | {s['tokens_per_s']:,.0f} | "
            f"{100 * s['mfu']:.1f}% | {', '.join(f'{x:.1f}' for x in s['peak_alloc_gib_by_rank'])} | "
            f"{audit} |"
        )
    return "\n".join(lines)


def p7_section() -> str:
    """Phase 7: the 72B QLoRA step on 1 GPU, replicated on 2 (DDP) and sharded on 2 (FSDP1)."""
    recs = {k: v for k, v in _latest_by_name().items() if k.startswith("p7.")}
    one = recs.get("p7.72b-single.r0")
    if not one:
        return "_No Phase 7 runs yet._"
    base_tps = one["summary"]["tokens_per_s"]
    base_loss = one["metrics"]["loss"]
    lines = [
        "Qwen2.5-72B NF4 + LoRA r=16, 16 BIRD questions per step, activation checkpointing, 20 steps, the "
        "same data order on every run. Scaling = 2-GPU tokens/s over twice the 1-GPU tokens/s. Exposed "
        "communication: NCCL kernel time with no compute kernel running, per step, from the profiler "
        "(steps 12-13). Loss difference: largest relative gap to the 1-GPU run's per-step loss.",
        "",
        "| Run | GPUs | Tokens/s | Scaling | MFU | Median step | "
        "Peak GPU allocated / reserved per rank (GiB) | Exposed comm per step, by rank | Loss vs 1 GPU |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("p7.72b-single.r0", "p7.72b-ddp.r0", "p7.72b-fsdp1.r0"):
        r = recs.get(name)
        if not r:
            continue
        s = r["summary"]
        label = name.split(".")[1]
        if s.get("status") != "ok":
            lines.append(
                f"| {label} | {s.get('world_size', '-')} | {s.get('status')} | - | - | - | - | - | - |"
            )
            continue
        n = s["world_size"]
        scaling = "-" if n == 1 else f"{100 * s['tokens_per_s'] / (n * base_tps):.1f}%"
        mem = ", ".join(
            f"{a:.1f} / {b:.1f}"
            for a, b in zip(s["peak_alloc_gib_by_rank"], s["peak_reserved_gib_by_rank"], strict=True)
        )
        comm = (
            ", ".join(f"{c['exposed_comm_s']:.2f} s" for c in s["comm_by_rank"])
            if s.get("comm_by_rank")
            else "-"
        )
        loss = r["metrics"]["loss"]
        gap = (
            "-"
            if n == 1
            else f"{100 * max(abs(a - b) / a for a, b in zip(base_loss, loss, strict=True)):.2f}%"
        )
        lines.append(
            f"| {label} | {n} | {s['tokens_per_s']:,.0f} | {scaling} | {100 * s['mfu']:.1f}% | "
            f"{s['step_time_s_median']:.2f} s | {mem} | {comm} | {gap} |"
        )
    return "\n".join(lines)


def faults_section() -> str:
    recs = _latest_by_name()
    ref = recs.get("p8-reference")
    if not ref:
        return "_No fault-injection runs yet._"
    full = ref["metrics"]["loss"]
    # (label, the step the failure hit, the resumed run). A killed run writes no record, so the failure
    # point comes from the scenario's configuration in slurm/phase8_faults.sbatch.
    scen = [
        ("SIGKILL mid-step (before step 25)", 25, "p8-kill-step-resume"),
        ("SIGKILL after writing, before publishing, the step-30 checkpoint", 30, "p8-kill-save-resume"),
        ("SIGTERM with stop_on_term (before step 25)", None, "p8-term-graceful-resume"),
        ("SIGTERM, default action (before step 25)", 25, "p8-term-default-resume"),
    ]
    step_s = ref["summary"]["step_time_s_median"]
    lines = [
        'Qwen2.5-1.5B, FSDP2 on 2 H200s, deterministic mode, checkpoints every 10 steps. "Matches" means '
        "every per-step loss after the resume equals the uninterrupted run's. Lost steps are the steps "
        "recomputed after the resume; their time is lost steps x the median step, plus the restart.",
        "",
        "| Failure | Resumed from | Lost steps (s) | Restart: load + restore (s) | Matches uninterrupted |",
        "|---|---|---|---|---|",
    ]
    for label, hit, resumed in scen:
        r = recs.get(resumed)
        if not r:
            lines.append(f"| {label} | - | - | - | - |")
            continue
        s = r["summary"]
        k = s["start_step"]
        lost = max(0, hit - k) if hit is not None else 0
        load = max(s.get("load_s_by_rank") or [0])
        ok = r["metrics"]["loss"] == full[k : k + len(r["metrics"]["loss"])]
        lines.append(
            f"| {label} | step {k} | {lost} ({lost * step_s:.1f}) | {load:.0f} + {s['restore_s'] or 0:.0f} | "
            f"{'yes' if ok else 'NO'} |"
        )
    if recs.get("p8-kill-eval") is None and "p8-kill-eval-resume" not in recs:
        lines.append(
            "| SIGKILL during the end-of-run evaluation | the final checkpoint; the next link finds the run "
            "complete and exits | 0 (only the evaluation) | - | - |"
        )
    sync, asy = recs.get("p8-sync"), recs.get("p8-async")
    if sync and asy:

        def stats(r):
            cs = r["summary"]["checkpoints"]
            return (
                _med([c["blocking_s"] if "blocking_s" in c else c["total_s"] for c in cs]),
                _med([c["total_s"] for c in cs]),
                cs[0]["bytes"] / 2**30,
                r["summary"]["step_time_s_median"],
            )

        b0, t0, g0, st0 = stats(sync)
        b1, t1, g1, st1 = stats(asy)
        lines += [
            "",
            "| Checkpoint | Size (GiB) | Training blocked per save (s) | Save complete (s) | "
            "Median step (s) |",
            "|---|---|---|---|---|",
            f"| synchronous DCP | {g0:.1f} | {b0:.1f} | {t0:.1f} | {st0:.3f} |",
            f"| DCP async_save | {g1:.1f} | {b1:.1f} | {t1:.1f} | {st1:.3f} |",
        ]
    return "\n".join(lines)


def summary_section() -> str:
    """Headline numbers, each computed from the same records as the detailed tables below."""
    rows = []
    entries = {e: _ok(v) for e, v in _p3_entries().items() if _ok(v)}
    if entries:
        best = max(entries.items(), key=lambda kv: _med([r["summary"]["mfu"] for r in kv[1]]))
        effs = []
        for e, v in entries.items():
            if v[0]["summary"]["world_size"] == 2 and v[0]["config"]["backend"] == "ddp":
                one = entries.get(e.replace("-ddp", "-single"))
                if one:
                    effs.append(
                        _med([r["summary"]["tokens_per_s"] for r in v])
                        / (2 * _med([r["summary"]["tokens_per_s"] for r in one]))
                    )
        rows.append(
            (
                "Full-parameter training, Qwen2.5 0.5B-14B, 1-2 H200s",
                f"up to {100 * _med([r['summary']['mfu'] for r in best[1]]):.1f}% MFU "
                f"untuned ({best[0]}); DDP 1->2 GPU scaling {100 * min(effs):.1f}-{100 * max(effs):.1f}%"
                if effs
                else "",
                "docs/phase3.md",
            )
        )
    p4c = _p3_entries("p4c.")
    if p4c and _ok(p4c.get("7b-best", [])) and _ok(p4c.get("7b-base", [])):
        b = _med([r["summary"]["tokens_per_s"] for r in _ok(p4c["7b-base"])])
        tuned = _ok(p4c["7b-best"])
        t_ = _med([r["summary"]["tokens_per_s"] for r in tuned])
        rows.append(
            (
                "Profiler-guided tuning, 7B FSDP2 on 2 H200s",
                f"{100 * (t_ - b) / b:+.0f}% tokens/s, "
                f"{100 * _med([r['summary']['mfu'] for r in tuned]):.1f}% MFU",
                "docs/phase4.md",
            )
        )
    phase3 = [r for v in _p3_entries().values() for r in _ok(v)]
    if phase3:
        e = [
            (_predict_now(r) - max(r["summary"]["peak_alloc_gib_by_rank"]))
            / max(r["summary"]["peak_alloc_gib_by_rank"])
            for r in phase3
        ]
        p4 = [r for v in _p3_entries("p4.").values() for r in _ok(v)]
        e4 = [
            (_predict_now(r) - max(r["summary"]["peak_alloc_gib_by_rank"]))
            / max(r["summary"]["peak_alloc_gib_by_rank"])
            for r in p4
        ]
        txt = f"median error {100 * _med(sorted(abs(x) for x in e)):.1f}% on {len(e)} runs"
        if e4:
            txt += f", {100 * _med(sorted(abs(x) for x in e4)):.1f}% out of sample ({len(e4)} runs)"
        rows.append(("Analytical peak-memory predictor", txt, "docs/phase3.md"))
    c = _by_name("eval").get("nf4-vs-bf16")
    if c:
        cs = c["summary"]
        rows.append(
            (
                "Qwen2.5-72B in NF4 vs BF16",
                f"next-token agreement {100 * cs['argmax_agreement_sql']:.1f}% on SQL tokens, "
                f"perplexity gap {100 * cs['ppl_sql_gap']:+.2f}%",
                "docs/phase5.md",
            )
        )
    evs = {}
    for r in load_records(ROOT / "results", "eval"):
        s = r["summary"]
        if (
            s.get("split") == "dev"
            and s.get("mode") in ("base", "finetuned")
            and not s["name"].endswith("-repro")
        ):
            evs[s["mode"]] = r
    if "base" in evs:
        b = evs["base"]["summary"]
        if "finetuned" in evs:
            import sys as _sys

            _sys.path.insert(0, str(ROOT / "src"))
            from dlt.bird.execute import paired_bootstrap_gain

            a, f = _correct_by_qid(evs["base"]), _correct_by_qid(evs["finetuned"])
            q = sorted(set(a) & set(f))
            gain, lo, hi = paired_bootstrap_gain([bool(a[i]) for i in q], [bool(f[i]) for i in q])
            fs = evs["finetuned"]["summary"]
            txt = (
                f"{100 * b['ex']:.1f}% -> {100 * fs['ex']:.1f}% execution accuracy on BIRD dev, "
                f"{100 * gain:+.1f} points (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})"
            )
        else:
            txt = f"base 3-shot {100 * b['ex']:.1f}% on BIRD dev; fine-tune in progress"
        rows.append(("72B QLoRA on one H200, BIRD text-to-SQL", txt, "docs/phase6.md"))
    recs = _latest_by_name()
    one, rep, shd = (recs.get(f"p7.72b-{k}.r0") for k in ("single", "ddp", "fsdp1"))
    if one and rep and shd and all(r["summary"].get("status") == "ok" for r in (one, rep, shd)):
        tps = one["summary"]["tokens_per_s"]
        mem = lambda r: max(r["summary"]["peak_alloc_gib_by_rank"])  # noqa: E731
        rows.append(
            (
                "72B QLoRA on 2 H200s, replicated vs sharded",
                f"DDP {100 * rep['summary']['tokens_per_s'] / (2 * tps):.1f}% scaling at "
                f"{mem(rep):.1f} GiB per GPU; FSDP1 {mem(shd):.1f} GiB per GPU at "
                f"{100 * shd['summary']['tokens_per_s'] / (2 * tps):.1f}% scaling",
                "docs/phase7.md",
            )
        )
    if recs.get("p8-sync") and recs.get("p8-async"):
        blk = lambda r: _med([c.get("blocking_s", c["total_s"]) for c in r["summary"]["checkpoints"]])  # noqa: E731
        rows.append(
            (
                "Fault tolerance",
                "every injected failure resumes bit-exact; async checkpoints stall "
                f"{blk(recs['p8-async']):.1f} s "
                f"instead of {blk(recs['p8-sync']):.1f} s per 18 GiB save",
                "docs/phase8.md",
            )
        )
    if not rows:
        return "_No results yet._"
    lines = ["| Result | Number | Details |", "|---|---|---|"]
    lines += [f"| {a} | {b} | [{c.split('/')[-1]}]({c}) |" for a, b, c in rows]
    return "\n".join(lines)


SECTIONS = {
    "hardware": hardware_section,
    "nccl": nccl_section,
    "smoke": smoke_section,
    "correctness": correctness_section,
    "chain": chain_section,
    "ladder": ladder_section,
    "scaling": scaling_section,
    "memory": memory_section,
    "ac": ac_section,
    "tuning": tuning_section,
    "profile": profile_section,
    "nf4": nf4_section,
    "bird": bird_section,
    "qlora": qlora_section,
    "p7": p7_section,
    "faults": faults_section,
    "summary": summary_section,
}


def render(text: str) -> str:
    for name, fn in SECTIONS.items():
        # The body between the markers may be empty (adjacent marker lines), so it must not need its own
        # newline before the closing marker.
        pat = re.compile(rf"(<!-- gen:{name} -->\n).*?(<!-- /gen:{name} -->)", re.S)
        text = pat.sub(lambda m, fn=fn: m.group(1) + fn() + "\n" + m.group(2), text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    readme = ROOT / "README.md"
    old = readme.read_text()
    new = render(old)
    if args.check:
        if old != new:
            sys.stdout.writelines(
                difflib.unified_diff(old.splitlines(True), new.splitlines(True), "README.md", "generated")
            )
            print("\nREADME.md is out of date: run python scripts/gen_readme.py")
            return 1
        return 0
    readme.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
