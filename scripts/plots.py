"""Figures for the README and docs, from the committed run records only.

    uv run --with matplotlib==3.10.7 python scripts/plots.py        (writes docs/img/*.png)

Matplotlib is not a project dependency (changing uv.lock would force a new venv on the cluster), so
the figures are made in an ephemeral environment. Colors: categorical slots 1-3 of the reference
palette (blue, orange, aqua), which pass the colorblind checks for all pairs; one color per backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import gen_readme as g  # noqa: E402

from dlt.results import load_records  # noqa: E402

OUT = ROOT / "docs" / "img"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BACKEND = {"single": "#2a78d6", "ddp": "#eb6834", "fsdp2": "#1baf7a"}
SPLIT = {"forward": "#2a78d6", "backward": "#eb6834", "optimizer": "#1baf7a", "other": "#eda100"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def chain_plot() -> None:
    links = [r for r in load_records(ROOT / "results", "train") if r["summary"].get("run_name") == "p2-chain"]
    if not links:
        return
    fig, ax = plt.subplots(figsize=(8, 3.6))
    steps, losses = [], []
    for r in links:
        first = r["metrics"]["first_step"]
        steps += [first + i + 1 for i in range(len(r["metrics"]["loss"]))]
        losses += r["metrics"]["loss"]
    w = 50
    smooth = [
        sum(losses[max(0, i - w + 1) : i + 1]) / len(losses[max(0, i - w + 1) : i + 1])
        for i in range(len(losses))
    ]
    ax.plot(steps, losses, color=BACKEND["single"], alpha=0.18, linewidth=1)
    ax.plot(steps, smooth, color=BACKEND["single"], linewidth=2, label=f"train loss ({w}-step mean)")
    vs = [(v["step"], v["loss"]) for r in links for v in r["metrics"]["val"]]
    ax.plot(
        [s for s, _ in vs],
        [v for _, v in vs],
        "o",
        color=BACKEND["ddp"],
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="validation loss",
        zorder=3,
    )
    for r in links[1:]:
        b = r["summary"]["start_step"]
        ax.axvline(b, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"job {r['slurm']['job_id']} resumes at step {b}",
            (b, ax.get_ylim()[0]),
            xytext=(6, 8),
            textcoords="offset points",
            color=MUTED,
            fontsize=9,
        )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (nats/token)")
    ax.set_title("One run across two Slurm jobs (Qwen2.5-0.5B, FSDP2, 2 H200s)")
    ax.legend(loc="upper right", ncol=2)
    save(fig, "p2_chain.png")


def mfu_plot() -> None:
    entries = {e: g._ok(v) for e, v in g._p3_entries().items() if g._ok(v)}
    if not entries:
        return
    names = list(entries)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.2))
    for i, e in enumerate(names):
        recs = entries[e]
        m = [100 * r["summary"]["mfu"] for r in recs]
        b = recs[0]["summary"]["backend"]
        ax.barh(i, g._med(m), color=BACKEND[b], height=0.7, edgecolor=SURFACE, linewidth=2)
        if len(m) > 1:
            ax.plot([min(m), max(m)], [i, i], color=INK, linewidth=1)
        ax.text(g._med(m) + 0.6, i, f"{g._med(m):.1f}%", va="center", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("MFU (% of 989 TFLOPS dense BF16 per GPU), median and range over repeats")
    ax.set_title("Model FLOPs utilization across the Qwen2.5 ladder")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BACKEND.values()]
    ax.legend(handles, list(BACKEND), loc="lower right")
    save(fig, "p3_mfu.png")


def memory_plot() -> None:
    entries = g._p3_entries()
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(6, 5.4))
    top = 190
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--", label="predicted = measured")
    ax.axhline(g.H200_GIB, color=GRID, linewidth=2)
    ax.text(4, g.H200_GIB + 2, f"{g.H200_GIB} GiB per H200", color=MUTED, fontsize=9)
    seen, oom_label = set(), "ran out of memory (drawn on the limit)"
    for recs in entries.values():
        s = recs[0]["summary"]
        b = s["backend"]
        label = b if b not in seen else None
        seen.add(b)
        pred = g._predict_now(recs[0])
        if s.get("status") == "oom":
            ax.plot(
                pred, g.H200_GIB, "x", color=BACKEND[b], markersize=10, markeredgewidth=2.5, label=oom_label
            )
            oom_label = None
            continue
        for r in recs:
            ax.plot(
                pred,
                max(r["summary"]["peak_alloc_gib_by_rank"]),
                "o",
                color=BACKEND[b],
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                label=label,
            )
            label = None
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"predicted peak per rank (GiB), predictor version {g.PREDICTOR_VERSION}")
    ax.set_ylabel("measured max_memory_allocated (GiB)")
    ax.set_title("Memory predictor vs measurement")
    ax.legend(loc="lower right")
    save(fig, "p3_memory.png")


def split_plot() -> None:
    entries = {e: g._ok(v) for e, v in g._p3_entries().items() if g._ok(v)}
    if not entries:
        return
    names = list(entries)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.2))
    for i, e in enumerate(names):
        recs = entries[e]
        f = g._med([r["summary"]["fwd_s_median"] for r in recs])
        b = g._med([r["summary"]["bwd_s_median"] for r in recs])
        o = g._med([r["summary"]["opt_s_median"] for r in recs])
        t = g._med([r["summary"]["step_time_s_median"] for r in recs])
        parts = {"forward": f, "backward": b, "optimizer": o, "other": max(0.0, t - f - b - o)}
        left = 0.0
        for k, v in parts.items():
            ax.barh(
                i,
                v / t * 100,
                left=left,
                color=SPLIT[k],
                height=0.7,
                edgecolor=SURFACE,
                linewidth=2,
                label=k if i == 0 else None,
            )
            left += v / t * 100
        ax.text(101, i, f"{t:.2f} s", va="center", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of step time (%); total step time at right")
    ax.set_title("Where a training step goes")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
    save(fig, "p3_step_split.png")


if __name__ == "__main__":
    chain_plot()
    mfu_plot()
    memory_plot()
    split_plot()
