#!/bin/bash
# RunPod GPU bootstrap for ColBERT compression benchmark.
#
# Docker image: runpod/pytorch:2.8.0-py3.13-cuda12.8.1-cudnn-devel-ubuntu22.04
#
# This is the GPU version — uses onnxruntime-gpu with CUDA for ~10-20x faster
# embedding vs CPU. See benchmarks/runpod-cpu-reference.md for the CPU version
# and a full log of every setup issue we hit.
#
# Usage:
#   1. Start a GPU pod on RunPod (any NVIDIA GPU with 8GB+ VRAM)
#   2. SSH in and run:
#      git clone https://github.com/ninthhousestudios/ogham-mcp.git /workspace/ogham-mcp
#      cd /workspace/ogham-mcp && git checkout worktree-colbert-reembed
#      bash scripts/runpod-gpu-setup.sh
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
# pgvector requires PG 13+. Ubuntu 22.04 ships PG 14, which is fine.
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
# Set password first — TCP auth uses scram-sha-256 by default, not trust.
# Adding trust to pg_hba.conf doesn't work (first matching rule wins, scram is above).
# ALTER USER is the reliable fix.
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='ogham'\" | grep -q 1 || createdb ogham"
su - postgres -c "psql ogham -c 'CREATE EXTENSION IF NOT EXISTS vector'"
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres'\""
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

echo "=== Step 6: Install uv + Python deps ==="
# The curl installer sometimes fails on RunPod. pip install is more reliable.
if ! command -v uv &>/dev/null; then
    pip install uv 2>/dev/null || pip3 install uv
fi
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL: uv aggressively downloads Python 3.14 beta which breaks pydantic:
#   TypeError: _eval_type() got an unexpected keyword argument 'prefer_fwd_module'
# pyproject.toml has requires-python = ">=3.13,<3.14" to prevent this.
# Belt and suspenders: pin to 3.13 and nuke any 3.14 uv already downloaded.
rm -rf /root/.local/share/uv/python/cpython-3.14*
uv python pin 3.13

# If venv exists from a previous run, remove it to avoid stale state.
rm -rf .venv
uv venv --python python3.13
# Install all extras. onnxruntime (CPU) comes from [onnx] extra.
uv sync --all-extras

# IMPORTANT: Replace CPU onnxruntime with GPU version.
# onnxruntime and onnxruntime-gpu conflict — must uninstall CPU first.
# The CUDA/cuDNN libs are already on the RunPod pytorch image.
echo "Swapping onnxruntime for onnxruntime-gpu..."
uv pip install --reinstall onnxruntime-gpu

echo "=== Step 7: Verify GPU is available to ONNX ==="
uv run python -c "
import sys
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Available ONNX providers: {providers}')
if 'CUDAExecutionProvider' not in providers:
    print('FATAL: CUDAExecutionProvider not available!')
    print('This benchmark requires GPU. Do not run on CPU.')
    print()
    print('Debugging:')
    print(f'  onnxruntime version: {ort.__version__}')
    print(f'  Available providers: {providers}')
    import subprocess
    r = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print(f'  nvidia-smi: {r.stdout[:200] if r.returncode == 0 else \"FAILED\"}')
    sys.exit(1)
print('GPU acceleration confirmed.')
"

echo "=== Step 8: Configure environment ==="
export DATABASE_URL="postgresql://postgres:postgres@localhost/ogham"
export DATABASE_BACKEND="postgres"
export EMBEDDING_PROVIDER="onnx"

cat > benchmarks/.env.local << 'ENVEOF'
EMBEDDING_PROVIDER=onnx
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://postgres:postgres@localhost/ogham
ENVEOF

echo "=== Step 9: Download ONNX model ==="
MODEL_DIR="${HOME}/.cache/ogham/bge-m3-onnx"
if [ ! -f "${MODEL_DIR}/bge_m3_model.onnx" ]; then
    echo "Downloading bge-m3 ONNX model (~2.2GB)..."
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('yuniko-software/bge-m3-onnx', local_dir='${MODEL_DIR}')
"
fi

echo "=== Step 10: Create table structure ==="
# get_backend() auto-migration doesn't create the table from scratch on a fresh DB.
# Create it explicitly via SQL so ingest doesn't fail with "relation does not exist".
su - postgres -c "psql ogham -c \"
CREATE TABLE IF NOT EXISTS memories (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content text NOT NULL,
    embedding vector(1024),
    profile text NOT NULL DEFAULT 'default',
    source text,
    tags text[] DEFAULT '{}',
    metadata jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    expires_at timestamptz,
    importance integer DEFAULT 5,
    access_count integer DEFAULT 0,
    last_accessed_at timestamptz,
    confidence real DEFAULT 0.7,
    compression_level integer DEFAULT 0,
    original_content text,
    sparse_embedding sparsevec(250002),
    colbert_vectors bytea,
    colbert_vectors_raw bytea
);\""
echo "memories table ready"

echo "=== Step 11: Ingest BEAM dataset ==="
echo "Ingesting all 100K bucket chats (dense + sparse embeddings)..."
echo "With GPU this should take ~20-30 minutes (vs ~5 hours on CPU)."
echo "Note: output appears per-chat, not per-batch. First chat takes ~1-2 min."
uv run python benchmarks/beam_benchmark.py --ingest --bucket 100K --beam-dir /tmp/BEAM

echo "=== Step 11b: Compact postgres WAL ==="
# Bulk inserts generate huge WAL. Checkpoint + vacuum to reclaim disk.
su - postgres -c "psql ogham -c 'CHECKPOINT; VACUUM;'"
echo "WAL compacted"

echo "=== Step 12: Embed raw f32 ColBERT + sparse vectors for all profiles ==="
echo "Re-embedding all profiles with raw f32 ColBERT..."
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
