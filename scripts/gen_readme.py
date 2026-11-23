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
        if status == "ok" and s.get("stopped_by") is not None:
            continue
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
        "from a 3-step profiler window (max over ranks).",
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
        "Scaling efficiency = tokens/s on 2 GPUs / (2 x tokens/s on 1 GPU), same model, precision, data, "
        "sequence length, global batch and activation checkpointing (medians over repeats).",
        "",
        "| Model | 1-GPU run | 2-GPU run | 1 GPU tokens/s | 2 GPU tokens/s | Efficiency |",
        "|---|---|---|---|---|---|",
    ]
    for e2, v2 in twos.items():
        c2 = v2[0]["config"]
        base = [
            e1
            for e1, v1 in ones.items()
            if v1[0]["config"]["model"] == c2["model"]
            and v1[0]["config"]["activation_checkpointing"] == c2["activation_checkpointing"]
        ]
        t2 = _med([r["summary"]["tokens_per_s"] for r in v2])
        if not base:
            mfu = _med([100 * r["summary"]["mfu"] for r in v2])
            lines.append(
                f"| {_model(v2[0])} | none (does not fit on 1 GPU) | {e2} | - | {t2:,.0f} | "
                f"n/a; MFU {mfu:.1f}% |"
            )
            continue
        t1 = _med([r["summary"]["tokens_per_s"] for r in ones[base[0]]])
        lines.append(
            f"| {_model(v2[0])} | {base[0]} | {e2} | {t1:,.0f} | {t2:,.0f} | {100 * t2 / (2 * t1):.1f}% |"
        )
    return "\n".join(lines)


def memory_section() -> str:
    entries = _p3_entries()
    if not entries:
        return "_No Phase 3 runs yet._"
    lines = [
        "Predicted peak allocated memory per rank (dlt.metrics.memory, analytical, not fitted) against the "
        f"measured `max_memory_allocated` (max over ranks), for every run including OOMs. Limit: {H200_GIB} "
        "GiB visible per H200.",
        "",
        "| Run | Predicted (GiB) | Measured (GiB) | Error | Outcome |",
        "|---|---|---|---|---|",
    ]
    errs, right, total = [], 0, 0
    for entry, recs in entries.items():
        s = recs[0]["summary"]
        if s.get("status") == "oom":
            pred = s["predicted_gib"]["total"]
            phase = s.get("phase")
            total += 1
            right += pred > H200_GIB
            lines.append(
                f"| {entry} | {pred:.1f} | OOM during {phase} | - | "
                f"predicted {'OOM' if pred > H200_GIB else 'fit'} |"
            )
            continue
        preds = [r["summary"]["predicted_gib"] for r in recs]
        meas = [max(r["summary"]["peak_alloc_gib_by_rank"]) for r in recs]
        e = [(p - m) / m for p, m in zip(preds, meas, strict=True)]
        errs += e
        total += 1
        right += preds[0] <= H200_GIB
        lines.append(
            f"| {entry} | {_med(preds):.1f} | {_med(meas):.1f} | {100 * _med(e):+.1f}% | "
            f"fit (predicted {'fit' if preds[0] <= H200_GIB else 'OOM'}) |"
        )
    if errs:
        abs_e = sorted(abs(x) for x in errs)
        lines.insert(
            1,
            f"\nOver {len(errs)} finished runs: median absolute error {100 * _med(abs_e):.1f}%, worst "
            f"{100 * abs_e[-1]:.1f}%; fit vs OOM predicted correctly for {right} of {total} configurations.",
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
