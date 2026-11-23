"""BIRD preparation, prompts and execution scoring on a small generated SQLite database."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dlt.bird import data, execute, prompt

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def db(tmp_path) -> Path:
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE "order" (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
        INSERT INTO customer VALUES (1, 'Ana', 'Boston'), (2, 'Bo', 'Austin'), (3, 'Cy', 'Boston');
        INSERT INTO "order" VALUES (1, 1, 10.5), (2, 1, 3.0), (3, 2, 7.25);
        """
    )
    con.commit()
    con.close()
    return path


def test_execution_match_is_set_equality(db):
    gold = execute.run_sql(db, "SELECT name FROM customer WHERE city = 'Boston' ORDER BY name")
    same = execute.run_sql(db, "SELECT DISTINCT name FROM customer WHERE city = 'Boston' ORDER BY id DESC")
    dup = execute.run_sql(db, 'SELECT c.name FROM customer c JOIN "order" o ON o.customer_id = c.id')
    other_cols = execute.run_sql(db, "SELECT name, city FROM customer WHERE city = 'Boston'")
    assert execute.ex_correct(same, gold)  # order and duplicates do not matter
    assert execute.run_sql(db, "SELECT name FROM customer WHERE id = 1").rows == frozenset({("Ana",)})
    assert dup.rows == frozenset({("Ana",), ("Bo",)})
    assert not execute.ex_correct(other_cols, gold)  # columns do


def test_errors_and_timeouts_are_wrong(db):
    bad = execute.run_sql(db, "SELECT nope FROM customer")
    assert not bad.ok and "no such column" in bad.error
    runaway = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
    slow = execute.run_sql(db, runaway, timeout=0.3)
    assert not slow.ok and slow.error.startswith("timeout") and slow.seconds < 5
    gold = execute.run_sql(db, "SELECT 1")
    assert not execute.ex_correct(bad, gold)


def test_run_many_keeps_order_and_queued_queries_get_their_own_timeout(db):
    runaway = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
    jobs = [(db, runaway)] * 3 + [(db, "SELECT count(*) FROM customer")] * 3
    res = execute.run_many(jobs, workers=2, timeout=0.5)
    assert [r.ok for r in res] == [False] * 3 + [True] * 3
    assert res[-1].rows == frozenset({(3,)})


def test_read_only(db):
    r = execute.run_sql(db, "DELETE FROM customer")
    assert not r.ok
    assert execute.run_sql(db, "SELECT count(*) FROM customer").rows == frozenset({(3,)})


def test_bootstrap():
    correct = [True] * 60 + [False] * 40
    lo, hi = execute.bootstrap_ci(correct, n=2000)
    assert lo < 0.6 < hi and hi - lo < 0.25
    assert execute.bootstrap_ci(correct, n=2000) == (lo, hi)  # seeded
    gain, glo, ghi = execute.paired_bootstrap_gain(
        [False] * 50 + [True] * 50, [True] * 70 + [False] * 30, n=2000
    )
    assert gain == pytest.approx(0.2) and glo <= gain <= ghi
    assert execute.by_group([True, False, True], ["a", "a", "b"]) == {
        "a": {"n": 2, "ex": 0.5},
        "b": {"n": 1, "ex": 1.0},
    }


def test_schema_texts(db):
    s = data.schema_texts(db, k=2)
    assert s["ddl"].startswith("CREATE TABLE customer") and s["ddl"].count("CREATE TABLE") == 2
    assert "-- customer row: (1, 'Ana', 'Boston')" in s["ddl_rows"]
    assert "-- order row: (1, 1, 10.5)" in s["ddl_rows"]


def test_split_by_database_is_whole_databases_and_deterministic():
    qs = [{"db_id": f"db{i % 10}"} for i in range(1000)]
    train, val = data.split_by_database(qs, 0.08)
    assert not train & val and train | val == {f"db{i}" for i in range(10)}
    assert sum(q["db_id"] in val for q in qs) >= 80
    assert data.split_by_database(qs, 0.08) == (train, val)
    t2, v2 = data.split_by_database(qs, 0.08, eligible={"db1", "db2", "db3"}, min_databases=2)
    assert v2 <= {"db1", "db2", "db3"} and len(v2) >= 2


def test_prompt_and_extract():
    ex = {"question": "How many customers?", "evidence": "", "sql": "SELECT count(*) FROM customer;"}
    p = prompt.prompt(ex, "CREATE TABLE customer (id INTEGER);")
    assert p.endswith("-- Question: How many customers?\n-- SQL:\n") and "External knowledge" not in p
    assert prompt.target(ex) == "SELECT count(*) FROM customer;\n"
    assert prompt.extract_sql("SELECT 1\nFROM t\n\n-- Question: next") == "SELECT 1\nFROM t"
    assert prompt.extract_sql("Here:\n```sql\nSELECT 2;\n```\nDone") == "SELECT 2;"
    shots = prompt.few_shot_prefix([ex])
    assert "-- SQL:\nSELECT count(*) FROM customer;" in shots
    msgs = prompt.instruct_messages({**ex, "evidence": "k"}, "CREATE TABLE t (a);")
    assert msgs[0]["role"] == "system" and "External knowledge: k" in msgs[1]["content"]


def test_nf4_compare(tmp_path):
    def save(path, argmax, lp):
        np.savez(
            path,
            argmax=np.array(argmax, dtype=np.int32),
            gold_logprob=np.array(lp, dtype=np.float32),
            offsets=np.array([0, 3, 5]),
            sql_start=np.array([1, 0]),
            question_id=np.array([7, 9]),
        )

    save(tmp_path / "a.npz", [1, 2, 3, 4, 5], [-1.0, -1.0, -1.0, -2.0, -2.0])
    save(tmp_path / "b.npz", [1, 2, 9, 4, 5], [-1.0, -0.5, -0.5, -2.0, -2.0])
    env = {**os.environ, "DLT_RESULTS": str(tmp_path / "res"), "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-m", "dlt.qlora.nf4", "compare", str(tmp_path / "a.npz"), str(tmp_path / "b.npz")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    (rec,) = [json.loads(p.read_text()) for p in (tmp_path / "res" / "eval").glob("*.json")]
    s = rec["summary"]
    assert s["argmax_agreement_all"] == pytest.approx(0.8)
    assert s["argmax_agreement_sql"] == pytest.approx(
        3 / 4
    )  # positions 1, 2 of the first, both of the second
    assert s["ppl_sql"]["a"] == pytest.approx(np.exp(1.5))
    assert s["ppl_sql"]["b"] == pytest.approx(np.exp(1.25))
