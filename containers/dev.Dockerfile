# CPU dev container for the box: unit tests, lint, README generation. The venv lives in a bind mount
# (see scripts/dev.sh) so it survives container restarts and matches uv.lock exactly.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv==0.13.9
ENV UV_PROJECT_ENVIRONMENT=/venv UV_CACHE_DIR=/uv-cache UV_LINK_MODE=copy PYTHONPATH=/work/src
WORKDIR /work
