#!/bin/bash
# Build the job venv from uv.lock (idempotent). Runs inside slurm/setup_env.sbatch on a CPU node.
set -euo pipefail
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
source "$SRC/scripts/env.sh"
UV_VERSION=0.13.9

if ! "$UV" --version 2>/dev/null | grep -q " $UV_VERSION"; then
  mkdir -p "$DLT_HOME/tools"
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$DLT_HOME/tools" UV_NO_MODIFY_PATH=1 sh
fi
"$UV" --version

cd "$SRC"
UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync --frozen --no-dev --managed-python --python 3.12
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import sys
print("python", sys.version.split()[0])
for p in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "nvidia-nccl-cu12"]:
    print(p, md.version(p))
EOF
echo "venv ready: $VENV"
