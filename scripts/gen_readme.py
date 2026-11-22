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


SECTIONS = {"hardware": hardware_section, "nccl": nccl_section, "smoke": smoke_section}


def render(text: str) -> str:
    for name, fn in SECTIONS.items():
        pat = re.compile(rf"(<!-- gen:{name} -->\n).*?(\n<!-- /gen:{name} -->)", re.S)
        text = pat.sub(lambda m, fn=fn: m.group(1) + fn() + m.group(2), text)
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
