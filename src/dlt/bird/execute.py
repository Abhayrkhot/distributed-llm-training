"""Execution accuracy (EX) on SQLite databases, following BIRD's official evaluator.

A prediction is correct when the set of rows it returns equals the set of rows the gold SQL returns on the
same database (``set(pred) == set(gold)``, as in DAMO-ConvAI/bird ``llm/src/evaluation.py``): row order
and duplicate rows do not matter, column order does. A query that errors, returns more than
``MAX_ROWS`` rows or runs past its timeout is wrong.

The timeout is enforced inside SQLite (a progress handler aborts the statement once its own deadline
passes, measured from when that query starts), so a runaway query frees its worker instead of blocking
it, and a query waiting in the pool's queue is never charged for time it did not run. Connections are
read-only.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 30.0
MAX_ROWS = 1_000_000


@dataclass(frozen=True)
class Result:
    ok: bool  # the query executed
    rows: frozenset | None
    error: str | None = None
    seconds: float = 0.0


def local_copies(dbs: dict[str, Path], root: str | Path | None = None) -> dict[str, Path]:
    """Copy databases to node-local disk ($TMPDIR, else /tmp) and return the new paths. Queries against
    the shared /scratch (VAST NFS) copies stalled on file locking during preparation; local copies
    opened with ``immutable=1`` take no locks at all."""
    import shutil

    base = Path(root or os.environ.get("TMPDIR") or "/tmp") / f"dlt-bird-{os.getuid()}"
    base.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, src in dbs.items():
        dst = base / Path(src).name
        if not dst.exists() or dst.stat().st_size != Path(src).stat().st_size:
            tmp = dst.with_suffix(f".tmp{os.getpid()}")
            shutil.copy2(src, tmp)
            tmp.replace(dst)
        out[key] = dst
    return out


def run_sql(db: str | Path, sql: str, timeout: float = DEFAULT_TIMEOUT_S) -> Result:
    t0 = time.monotonic()
    deadline = t0 + timeout
    try:
        # immutable: the databases never change, so SQLite takes no file locks (read-only still enforced)
        con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(MAX_ROWS + 1)
        finally:
            con.close()
        if len(rows) > MAX_ROWS:
            return Result(False, None, f"more than {MAX_ROWS} rows", time.monotonic() - t0)
        return Result(True, frozenset(rows), None, time.monotonic() - t0)
    except sqlite3.OperationalError as e:
        msg = f"timeout after {timeout:.0f}s" if time.monotonic() > deadline else f"OperationalError: {e}"
        return Result(False, None, msg[:300], time.monotonic() - t0)
    except Exception as e:  # noqa: BLE001 (any failure makes the query wrong)
        return Result(False, None, f"{type(e).__name__}: {e}"[:300], time.monotonic() - t0)


def _worker(args) -> Result:
    return run_sql(*args)


def run_many(
    jobs: list[tuple[Path, str]], workers: int = 8, timeout: float = DEFAULT_TIMEOUT_S
) -> list[Result]:
    """Run (db, sql) pairs in parallel, in order; each query has its own timeout."""
    if workers <= 1:
        return [run_sql(db, sql, timeout) for db, sql in jobs]
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=200) as pool:
        return list(pool.imap(_worker, [(str(db), sql, timeout) for db, sql in jobs], chunksize=1))


def ex_correct(pred: Result, gold: Result) -> bool:
    return pred.ok and gold.ok and pred.rows == gold.rows


def bootstrap_ci(
    correct: list[bool], n: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap over questions for the accuracy."""
    rng = random.Random(seed)
    k = len(correct)
    if not k:
        return (0.0, 0.0)
    vals = [int(c) for c in correct]
    stats = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return stats[int(lo * n)], stats[min(n - 1, int(hi * n))]


def paired_bootstrap_gain(
    a: list[bool], b: list[bool], n: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Accuracy of b minus a on the same questions: (gain, 2.5th, 97.5th percentile), resampling questions."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs the same questions")
    rng = random.Random(seed)
    k = len(a)
    d = [int(y) - int(x) for x, y in zip(a, b, strict=True)]
    stats = sorted(sum(d[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(d) / k, stats[int(0.025 * n)], stats[min(n - 1, int(0.975 * n))]


def by_group(correct: Iterable[bool], groups: Iterable[str]) -> dict[str, dict]:
    acc: dict[str, list[int]] = {}
    for c, g in zip(correct, groups, strict=True):
        acc.setdefault(g, []).append(int(c))
    return {g: {"n": len(v), "ex": sum(v) / len(v)} for g, v in sorted(acc.items())}
