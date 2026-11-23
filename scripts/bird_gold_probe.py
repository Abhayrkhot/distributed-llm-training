"""Why did gold SQL time out during BIRD preparation? Re-run a sample of the timed-out golds one at a time,
against node-local copies of the databases, with and without SQLite file locking, and report times.

    python scripts/bird_gold_probe.py $DLT_HOME/data/bird --train 40 --timeout 120     (in a CPU job)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path


def _child(db: str, sql: str, uri: str, timeout: float, q) -> None:
    t0 = time.monotonic()
    con = sqlite3.connect(f"file:{db}?{uri}", uri=True)
    con.set_progress_handler(lambda: 1 if time.monotonic() - t0 > timeout else 0, 10_000)
    try:
        n = len(con.execute(sql).fetchall())
        q.put(f"ok {n} rows")
    except sqlite3.Error as e:
        q.put(("timeout" if time.monotonic() - t0 > timeout else f"error {e}")[:80])
    finally:
        con.close()


def timed(db: Path, sql: str, uri: str, timeout: float) -> tuple[str, float]:
    """In a child process killed after timeout + 30 s: a call blocked on a file lock never reaches
    SQLite's progress handler, so only a hard kill bounds it."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.monotonic()
    p = ctx.Process(target=_child, args=(str(db), sql, uri, timeout, q))
    p.start()
    p.join(timeout + 30)
    if p.is_alive():
        p.kill()
        p.join()
        return "hung (killed)", time.monotonic() - t0
    return (q.get() if not q.empty() else f"died rc={p.exitcode}"), time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()
    m = json.loads((args.data / "manifest.json").read_text())
    train_q = json.loads(
        Path(glob.glob(str(args.data / "raw/train/**/train.json"), recursive=True)[0]).read_text()
    )
    dev_rows = [json.loads(line) for line in (args.data / "dev.jsonl").read_text().splitlines()]
    dev_q = {r["question_id"]: r for r in dev_rows}
    slow = [r for r in m["dropped_train"] if (r["error"] or "").startswith("timeout")]
    random.Random(0).shuffle(slow)
    cases = [("train", r["db_id"], train_q[r["question_id"]]["SQL"]) for r in slow[: args.train]]
    cases += [("dev", r["db_id"], dev_q[r["question_id"]]["sql"]) for r in m["dev_gold_failed"]]
    local = Path(tempfile.mkdtemp(prefix="bird-", dir=os.environ.get("TMPDIR", "/tmp")))
    for db in sorted({c[1] for c in cases}):
        shutil.copy2(args.data / "databases" / f"{db}.sqlite", local / f"{db}.sqlite")
    out = []
    for split, db, sql in cases:
        row = {"split": split, "db": db}
        for name, path, uri in (
            ("nfs_locking", args.data / "databases" / f"{db}.sqlite", "mode=ro"),
            ("local_immutable", local / f"{db}.sqlite", "mode=ro&immutable=1"),
        ):
            res, sec = timed(path, sql, uri, args.timeout)
            row[name] = {"result": res, "s": round(sec, 2)}
        out.append(row)
        print(json.dumps(row), flush=True)
    for split in ("train", "dev"):
        rows = [r for r in out if r["split"] == split]
        for k in ("nfs_locking", "local_immutable"):
            ok = [r[k]["s"] for r in rows if r[k]["result"].startswith("ok")]
            under30 = sum(s <= 30 for s in ok)
            print(
                f"{split} {k}: {len(ok)}/{len(rows)} finish within {args.timeout:.0f}s, "
                f"{under30} within 30s, "
                f"median {sorted(ok)[len(ok) // 2] if ok else None}s",
                flush=True,
            )
    shutil.rmtree(local, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
