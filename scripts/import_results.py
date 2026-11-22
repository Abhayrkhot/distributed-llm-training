"""Copy run records pulled from Explorer (results/explorer/<kind>/*.json) into results/<kind>/.

Each record is validated against the schema and scrubbed (cluster paths and the given usernames) before
it is written. Existing files are left alone, so re-running is safe.

    python scripts/import_results.py [--scrub NAME ...] [--only RUN_ID ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dlt.results import KINDS, validate  # noqa: E402


def scrub(text: str, words: list[str]) -> str:
    for w in words:
        text = text.replace(w, "$USER")
    return re.sub(r"/(scratch|home|work|projects)/(?!\$USER)[A-Za-z0-9._-]+", r"/\1/$USER", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "results" / "explorer")
    ap.add_argument("--scrub", action="append", default=[])
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args()
    imported = skipped = 0
    for kind in sorted(KINDS):
        for path in sorted((args.src / kind).glob("*.json")):
            rec = json.loads(path.read_text())
            if args.only and rec.get("run_id") not in args.only:
                continue
            errors = validate(rec)
            if errors:
                print(f"invalid {path}: {errors}")
                return 1
            dst = ROOT / "results" / kind / path.name
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(scrub(path.read_text(), args.scrub))
            print("imported", dst.relative_to(ROOT))
            imported += 1
    print(f"{imported} imported, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
