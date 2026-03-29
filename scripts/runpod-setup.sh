#!/bin/bash
# RunPod bootstrap for ColBERT compression benchmark.
#
# Docker image: postgres:16 (on RunPod, select this as the container image)
#   - PostgreSQL 16 ready out of the box
#   - Debian Bookworm base (Python 3.11 system, we install 3.13 via uv)
#
# No data upload needed — ingests BEAM dataset and embeds from scratch on the pod.
#
# Prerequisites:
#   1. Start a CPU pod with 48GB+ RAM on RunPod, image: postgres:16
#   2. SSH in and run:
#      apt-get update && apt-get install -y git
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
apt-get install -y -qq git build-essential curl postgresql-server-dev-16

echo "=== Step 2: pgvector ==="
if ! psql -U postgres -c "SELECT 1 FROM pg_available_extensions WHERE name='vector'" 2>/dev/null | grep -q 1; then
    cd /tmp
    git clone --depth 1 https://github.com/pgvector/pgvector.git
    cd pgvector
    make && make install
fi

echo "=== Step 3: Start postgres + create DB ==="
# postgres:16 image runs PG automatically, but ensure it's up
pg_isready -U postgres || pg_ctlcluster 16 main start || {
    # postgres:16 Docker image uses a different init
    su - postgres -c "pg_ctl -D /var/lib/postgresql/data start" || true
}
psql -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='ogham'" | grep -q 1 || createdb -U postgres ogham
psql -U postgres ogham -c 'CREATE EXTENSION IF NOT EXISTS vector'
echo "PostgreSQL $(psql -U postgres -tc 'SHOW server_version' | xargs) with pgvector ready"

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
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13
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
