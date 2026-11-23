"""One prompt format for every model: the base few-shot baseline, the fine-tuned model, and (as chat) the
Instruct reference. The fine-tuned model is trained to continue a prompt with the SQL and end-of-text.
"""

from __future__ import annotations

import re

HEADER = "-- SQLite database schema"
INSTRUCT_SYSTEM = (
    "You are an expert SQLite developer. Given a database schema, optional external knowledge and a "
    "question, write one SQLite query that answers the question. Reply with the SQL query only."
)


def prompt(ex: dict, schema: str) -> str:
    parts = [HEADER, schema, ""]
    if ex.get("evidence"):
        parts.append(f"-- External knowledge: {ex['evidence']}")
    parts += [f"-- Question: {ex['question']}", "-- SQL:"]
    return "\n".join(parts) + "\n"


def target(ex: dict) -> str:
    return ex["sql"].strip() + "\n"


def few_shot_prefix(shots: list[dict]) -> str:
    """Worked examples for the base model: question, knowledge and SQL, without their schemas."""
    blocks = []
    for ex in shots:
        lines = []
        if ex.get("evidence"):
            lines.append(f"-- External knowledge: {ex['evidence']}")
        lines += [f"-- Question: {ex['question']}", "-- SQL:", target(ex).rstrip(), ""]
        blocks.append("\n".join(lines))
    return "-- Examples of questions and their SQL on other databases\n\n" + "\n".join(blocks) + "\n"


def instruct_messages(ex: dict, schema: str) -> list[dict]:
    user = [HEADER, schema, ""]
    if ex.get("evidence"):
        user.append(f"External knowledge: {ex['evidence']}")
    user.append(f"Question: {ex['question']}")
    return [{"role": "system", "content": INSTRUCT_SYSTEM}, {"role": "user", "content": "\n".join(user)}]


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """The SQL from a completion: a fenced block if present, else text up to a blank line or a new
    ``--`` block (how the base model starts the next example)."""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    out = []
    for line in text.strip().splitlines():
        if not line.strip() or line.lstrip().startswith("-- "):
            break
        out.append(line)
    return "\n".join(out).strip()
