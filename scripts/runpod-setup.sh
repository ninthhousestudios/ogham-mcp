#!/bin/bash
# RunPod bootstrap for ColBERT compression benchmark.
#
# Docker image: runpod/base:0.7.0-ubuntu2204 (or any Ubuntu 22.04+ RunPod image)
#
# No data upload needed — ingests BEAM dataset and embeds from scratch on the pod.
#
# Usage:
#   1. Start a CPU pod with 48GB+ RAM on RunPod
#   2. SSH in and run:
#      git clone https://github.com/ninthhousestudios/ogham-mcp.git /workspace/ogham-mcp
#      cd /workspace/ogham-mcp && git checkout worktree-colbert-reembed
#      bash scripts/runpod-setup.sh
#
# After setup completes, run:
#   cd /workspace/ogham-mcp
#   uv run python scripts/run-benchmark-matrix.py --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40
#   uv run python scripts/generate-results-table.py

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/ogham-mcp"

echo "=== Step 1: System packages ==="
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib postgresql-server-dev-all git build-essential curl

echo "=== Step 2: pgvector ==="
if ! find /usr/lib/postgresql -name "vector.so" 2>/dev/null | grep -q .; then
    cd /tmp
    git clone --depth 1 https://github.com/pgvector/pgvector.git
    cd pgvector
    make && make install
fi

echo "=== Step 3: Start postgres + create DB ==="
PG_VER=$(pg_lsclusters -h | head -1 | awk '{print $1}')
echo "Found PostgreSQL ${PG_VER}"
pg_ctlcluster "${PG_VER}" main start || true
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='ogham'\" | grep -q 1 || createdb ogham"
su - postgres -c "psql ogham -c 'CREATE EXTENSION IF NOT EXISTS vector'"
echo "PostgreSQL ${PG_VER} with pgvector ready"

echo "=== Step 4: Clone BEAM dataset ==="
if [ ! -d /tmp/BEAM ]; then
    git clone --depth 1 https://github.com/mohammadtavakoli78/BEAM /tmp/BEAM
fi

echo "=== Step 5: Verify ogham-mcp repo ==="
if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: Clone the repo first:"
    echo "  git clone https://github.com/ninthhousestudios/ogham-mcp.git ${REPO_DIR}"
    echo "  cd ${REPO_DIR} && git checkout worktree-colbert-reembed"
    exit 1
fi
cd "${REPO_DIR}"

echo "=== Step 6: Install uv + Python 3.13 + deps ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13
uv venv --python 3.13
uv sync --all-extras

echo "=== Step 7: Configure environment ==="
export DATABASE_URL="postgresql://postgres@localhost/ogham"
export DATABASE_BACKEND="postgres"
export EMBEDDING_PROVIDER="onnx"

cat > benchmarks/.env.local << 'ENVEOF'
EMBEDDING_PROVIDER=onnx
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://postgres@localhost/ogham
ENVEOF

echo "=== Step 8: Download ONNX model ==="
MODEL_DIR="${HOME}/.cache/ogham/bge-m3-onnx"
if [ ! -f "${MODEL_DIR}/bge_m3_model.onnx" ]; then
    echo "Downloading bge-m3 ONNX model (~2.2GB)..."
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('yuniko-software/bge-m3-onnx', local_dir='${MODEL_DIR}')
"
fi

echo "=== Step 9: Create table structure ==="
uv run python -c "
import os
os.environ.setdefault('DATABASE_URL', 'postgresql://postgres@localhost/ogham')
os.environ.setdefault('DATABASE_BACKEND', 'postgres')
os.environ.setdefault('EMBEDDING_PROVIDER', 'onnx')
from ogham.database import get_backend
b = get_backend()
print('Backend ready, table auto-migrated')
b._execute('ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_vectors_raw bytea', fetch='none')
print('colbert_vectors_raw column ready')
"

echo "=== Step 10: Ingest BEAM dataset ==="
echo "Ingesting all 100K bucket chats (dense embeddings)..."
uv run python benchmarks/beam_benchmark.py --ingest --bucket 100K --beam-dir /tmp/BEAM

echo "=== Step 11: Embed raw f32 ColBERT + sparse vectors for all profiles ==="
echo "This re-embeds all profiles with dense + sparse + raw f32 ColBERT..."
uv run python scripts/embed-colbert-raw.py --all

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the benchmark:"
echo "  cd ${REPO_DIR}"
echo "  uv run python scripts/run-benchmark-matrix.py --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40"
echo ""
echo "Then generate tables:"
echo "  uv run python scripts/generate-results-table.py"
