# Shared environment for Explorer jobs (sourced). Expects SRC = the code directory of one commit.
# The venv is keyed by the uv.lock hash, so commits with the same lockfile share one venv.
export DLT_HOME=${DLT_HOME:-/scratch/$USER/dlt}
export UV=$DLT_HOME/tools/uv
export UV_CACHE_DIR=$DLT_HOME/uv-cache
export UV_PYTHON_INSTALL_DIR=$DLT_HOME/python
export UV_LINK_MODE=copy
export HF_HOME=$DLT_HOME/hf
export DLT_RESULTS=$DLT_HOME/results
export PYTHONNOUSERSITE=1
export PYTHONPATH=$SRC/src
export no_proxy=localhost,127.0.0.1${no_proxy:+,$no_proxy}
DLT_LOCK_HASH=$(sha256sum "$SRC/uv.lock" | cut -c1-12)
export DLT_LOCK_HASH
export VENV=$DLT_HOME/venv/$DLT_LOCK_HASH
DLT_GIT_SHA=$(cat "$SRC/GIT_SHA" 2>/dev/null || echo unknown)
export DLT_GIT_SHA
