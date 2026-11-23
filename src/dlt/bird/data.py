"""Download and prepare BIRD for training and evaluation (run once in a CPU job).

    python -m dlt.bird.data --out $DLT_HOME/data/bird [--workers 16]

Sources (docs/phase0.md): the official train.zip and dev.zip (databases, train questions), and the
cleaned dev questions released 2025-11-06 on Hugging Face (birdsql/bird_sql_dev_20251106, pinned).
Output in ``--out``:
    train.jsonl, val.jsonl, dev.jsonl   one question per line: question_id, db_id, question, evidence,
                                        sql, difficulty (dev), split
    schemas.json                        db_id -> CREATE statements, and with sample rows
    databases/<db_id>.sqlite            the SQLite files (hard links or copies of the extracted ones)
    manifest.json                       sources with sha256, counts, dropped questions, the split,
                                        gold-execution results and prompt-length statistics

The validation split holds out whole databases from BIRD train (dev databases never occur in train, so
a database-level split mimics that shift); it is used for every model-selection decision, and dev is
scored once per reported model. Train and val questions whose gold SQL does not execute are dropped and
counted; dev questions are all kept, and gold failures on dev are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sqlite3
import statistics
import time
import urllib.request
import zipfile
from pathlib import Path

from dlt.bird import execute

SOURCES = {
    "train.zip": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
        "bytes": 8_919_543_554,
        "etag": "1FBA2891BD215A42CE1004736AB7F47F-400",
    },
    "dev.zip": {
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        "bytes": 346_207_293,
        "etag": "04B4AF221C9186361F09B16ABFD917EC",
    },
}
DEV_1106 = {
    "repo": "birdsql/bird_sql_dev_20251106",
    "revision": "3c11fb193e5439b338e23677fa0aae11e8b85db9",
    "file": "data/dev_20251106-00000-of-00001.json",
}
VAL_FRACTION = 0.08  # of BIRD train questions, in whole databases
VAL_MAX_SCHEMA_TOKENS = 2500  # dev schemas are at most ~1,900 tokens; a few train schemas reach 5,000
SAMPLE_ROWS = 3
TOKENIZER = ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987")  # same vocabulary as 72B


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(url: str, dst: Path, expected: int) -> Path:
    """Stream to ``dst`` with resume (HTTP Range); the environment's proxy settings apply."""
    if dst.exists() and dst.stat().st_size == expected:
        return dst
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 24)
    size = part.stat().st_size
    if size != expected:
        raise RuntimeError(f"{url}: got {size} bytes, expected {expected}")
    part.replace(dst)
    print(f"downloaded {dst.name}: {size / 2**30:.2f} GiB in {time.time() - t0:.0f}s", flush=True)
    return dst


def extract_all(zip_path: Path, out: Path) -> None:
    """Extract a zip and any zips inside it (BIRD nests the databases)."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    for inner in sorted(out.rglob("*.zip")):
        if "__MACOSX" in inner.parts or inner.name.startswith("._"):  # macOS metadata, not archives
            continue
        with zipfile.ZipFile(inner) as z:
            z.extractall(inner.parent)
        inner.unlink()


def find_databases(root: Path) -> dict[str, Path]:
    dbs = {}
    for p in sorted(root.rglob("*.sqlite")):
        if "__MACOSX" in p.parts:
            continue
        dbs[p.stem] = p
    return dbs


def ddl(db: Path) -> list[tuple[str, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
    finally:
        con.close()
    return [(name, " ".join(sql.split())) for name, sql in rows if sql]


def sample_rows(db: Path, table: str, k: int) -> list[tuple]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(f'SELECT * FROM "{table}" LIMIT {k}').fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _cell(v) -> str:
    s = repr(v) if not isinstance(v, str) else repr(v[:40] + ("..." if len(v) > 40 else ""))
    return s


def schema_texts(db: Path, k: int = SAMPLE_ROWS) -> dict[str, str]:
    """CREATE statements alone, and with ``k`` sample rows per table as SQL comments."""
    tables = ddl(db)
    plain = "\n".join(sql + ";" for _, sql in tables)
    lines = []
    for name, sql in tables:
        lines.append(sql + ";")
        for row in sample_rows(db, name, k):
            lines.append(f"-- {name} row: (" + ", ".join(_cell(v) for v in row) + ")")
    return {"ddl": plain, "ddl_rows": "\n".join(lines)}


def load_train_questions(root: Path) -> list[dict]:
    (path,) = [p for p in root.rglob("train.json") if "__MACOSX" not in p.parts]
    rows = json.loads(path.read_text())
    return [
        {
            "question_id": i,
            "db_id": r["db_id"],
            "question": r["question"],
            "evidence": r.get("evidence") or "",
            "sql": r["SQL"],
        }
        for i, r in enumerate(rows)
    ]


def load_dev_1106(cache: Path) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DEV_1106["repo"],
        DEV_1106["file"],
        revision=DEV_1106["revision"],
        repo_type="dataset",
        cache_dir=str(cache),
    )
    text = Path(path).read_text()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [
        {
            "question_id": r["question_id"],
            "db_id": r["db_id"],
            "question": r["question"],
            "evidence": r.get("evidence") or "",
            "sql": r["SQL"],
            "difficulty": r.get("difficulty"),
        }
        for r in rows
    ]


def split_by_database(
    questions: list[dict],
    fraction: float,
    seed: int = 0,
    eligible: set[str] | None = None,
    min_databases: int = 1,
) -> tuple[set[str], set[str]]:
    """Whole databases to val (drawn from ``eligible``, in a seeded order) until it holds ``fraction``
    of the questions and at least ``min_databases`` databases; the rest train."""
    counts: dict[str, int] = {}
    for q in questions:
        counts[q["db_id"]] = counts.get(q["db_id"], 0) + 1
    dbs = sorted(db for db in counts if eligible is None or db in eligible)
    random.Random(seed).shuffle(dbs)
    val, n = set(), 0
    for db in dbs:
        if n >= fraction * len(questions) and len(val) >= min_databases:
            break
        val.add(db)
        n += counts[db]
    return set(counts) - val, val


def check_golds(questions: list[dict], dbs: dict[str, Path], workers: int) -> list:
    jobs = [(dbs[q["db_id"]], q["sql"]) for q in questions]
    return execute.run_many(jobs, workers=workers)


def token_stats(questions: list[dict], schemas: dict[str, dict], tokenizer) -> dict:
    from dlt.bird.prompt import prompt, target

    out = {}
    for key in ("ddl", "ddl_rows"):
        lens = [len(tokenizer(prompt(q, schemas[q["db_id"]][key]) + target(q)).input_ids) for q in questions]
        lens.sort()
        out[key] = {
            "median": statistics.median(lens),
            "p95": lens[int(0.95 * (len(lens) - 1))],
            "max": lens[-1],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args(argv)
    out: Path = args.out
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    sources = {}
    for name, src in SOURCES.items():
        path = download(src["url"], raw / name, src["bytes"])
        sources[name] = {**src, "sha256": sha256(path)}
        target_dir = raw / name.removesuffix(".zip")
        if not target_dir.exists():
            tmp = target_dir.with_suffix(".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            extract_all(path, tmp)
            tmp.replace(target_dir)

    train_dbs, dev_dbs = find_databases(raw / "train"), find_databases(raw / "dev")
    dbs = {**train_dbs, **dev_dbs}
    db_dir = out / "databases"
    db_dir.mkdir(exist_ok=True)
    for db_id, p in dbs.items():
        dst = db_dir / f"{db_id}.sqlite"
        if not dst.exists():
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)
    dbs = {db_id: db_dir / f"{db_id}.sqlite" for db_id in dbs}

    # queries run against node-local copies: on the shared /scratch copies they stalled on NFS locks
    local = execute.local_copies(dbs)
    train_all = load_train_questions(raw / "train")
    dev = load_dev_1106(out / "hf")
    missing = sorted({q["db_id"] for q in train_all + dev} - set(dbs))
    if missing:
        raise RuntimeError(f"databases missing for {missing}")

    t0 = time.time()
    train_res = check_golds(train_all, local, args.workers)
    dev_res = check_golds(dev, local, args.workers)
    gold_s = time.time() - t0
    # a gold that errors is broken SQL and is dropped; one that is merely slow is kept (training needs
    # valid SQL, not fast SQL, and dropping slow queries would skew training away from hard ones)
    broken = [not r.ok and not (r.error or "").startswith("timeout") for r in train_res]
    keep = [q for q, b in zip(train_all, broken, strict=True) if not b]
    dropped = [
        {"question_id": q["question_id"], "db_id": q["db_id"], "error": r.error}
        for q, r, b in zip(train_all, train_res, broken, strict=True)
        if b
    ]
    slow_kept = sum(not r.ok and not b for r, b in zip(train_res, broken, strict=True))
    dev_failed = [
        {"question_id": q["question_id"], "db_id": q["db_id"], "error": r.error}
        for q, r in zip(dev, dev_res, strict=True)
        if not r.ok
    ]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    train_db_list = sorted({q["db_id"] for q in keep})
    schemas = {db_id: schema_texts(dbs[db_id]) for db_id in sorted(set(train_db_list) | set(dev_dbs))}
    schema_tokens = {db_id: len(tok(s["ddl"]).input_ids) for db_id, s in schemas.items()}
    # val stands in for dev during model selection, so its databases have dev-sized schemas
    eligible = {db for db in train_db_list if schema_tokens[db] <= VAL_MAX_SCHEMA_TOKENS}
    train_db_ids, val_db_ids = split_by_database(keep, VAL_FRACTION, eligible=eligible, min_databases=6)
    splits = {
        "train": [{**q, "split": "train"} for q in keep if q["db_id"] in train_db_ids],
        "val": [{**q, "split": "val"} for q in keep if q["db_id"] in val_db_ids],
        "dev": [{**q, "split": "dev"} for q in dev],
    }
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    needed = sorted({q["db_id"] for rows in splits.values() for q in rows})
    schemas = {db_id: schemas[db_id] for db_id in needed}
    (out / "schemas.json").write_text(json.dumps(schemas, indent=1))
    manifest = {
        "sources": sources,
        "dev_1106": DEV_1106,
        "counts": {
            "train_questions_in_zip": len(train_all),
            "train_gold_failed_dropped": len(dropped),
            "train_gold_slow_kept": slow_kept,
            **{name: len(rows) for name, rows in splits.items()},
            "dev_gold_failed": len(dev_failed),
            "databases": {"train": len(train_db_ids), "val": len(val_db_ids), "dev": len(dev_dbs)},
        },
        "val_databases": sorted(val_db_ids),
        "val_max_schema_tokens": VAL_MAX_SCHEMA_TOKENS,
        "schema_tokens": schema_tokens,
        "gold_check_seconds": round(gold_s, 1),
        "dropped_train": dropped,
        "dev_gold_failed": dev_failed,
        "prompt_tokens": {name: token_stats(rows, schemas, tok) for name, rows in splits.items()},
        "database_sha256": {db_id: sha256(dbs[db_id]) for db_id in needed},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(manifest["counts"], indent=1), json.dumps(manifest["prompt_tokens"], indent=1), flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
