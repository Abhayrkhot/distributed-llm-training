import json
from pathlib import Path

import pytest

from dlt.results import KINDS, load_records, new_record, validate, write_record

ROOT = Path(__file__).resolve().parent.parent


def test_roundtrip(tmp_path):
    rec = new_record("smoke_train", config={"a": 1}, metrics={"loss": [1.0]}, summary={"x": 2})
    assert validate(rec) == []
    path = write_record(rec, tmp_path)
    assert path.parent.name == "smoke_train"
    loaded = load_records(tmp_path, "smoke_train")
    assert loaded == [json.loads(path.read_text())]


def test_rejects_bad_records(tmp_path):
    with pytest.raises(ValueError):
        new_record("not_a_kind", {}, {}, {})
    rec = new_record("bench", {}, {}, {})
    rec["metrics"] = []
    rec["schema_version"] = 99
    errors = validate(rec)
    assert any("metrics" in e for e in errors)
    assert any("schema_version" in e for e in errors)
    with pytest.raises(ValueError):
        write_record(rec, tmp_path)


def test_committed_results_are_valid_and_scrubbed():
    paths = [p for k in KINDS for p in (ROOT / "results" / k).glob("*.json")]
    for p in paths:
        text = p.read_text()
        assert validate(json.loads(text)) == [], p
        assert "/scratch/$USER" in text or "/scratch/" not in text, f"unscrubbed path in {p}"
