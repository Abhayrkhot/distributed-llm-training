#!/usr/bin/env bash
# Run a command in the CPU dev container on the box, e.g.  scripts/dev.sh 'pytest -q'
# The image is built on first use; the venv (synced from uv.lock) and uv cache are bind mounts.
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=dltdev:py3.12
docker image inspect "$IMG" >/dev/null 2>&1 || docker build -q -t "$IMG" -f containers/dev.Dockerfile containers >/dev/null
mkdir -p "$HOME/.cache/dlt/venv" "$HOME/.cache/dlt/uv" "$HOME/.cache/dlt/hf"
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
  -v "$PWD:/work" -v "$HOME/.cache/dlt/venv:/venv" -v "$HOME/.cache/dlt/uv:/uv-cache" -v "$HOME/.cache/dlt/hf:/hf" \
  -w /work "$IMG" bash -lc "if [ -f uv.lock ]; then uv sync --frozen -q; fi; export PATH=/venv/bin:\$PATH; $*"
