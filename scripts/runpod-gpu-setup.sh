#!/bin/bash
# RunPod GPU bootstrap for ColBERT compression benchmark.
#
# Docker image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#
# This is the GPU version — uses onnxruntime-gpu with CUDA for ~10-20x faster
# embedding vs CPU. See benchmarks/runpod-cpu-reference.md for the CPU version
# and a full log of every setup issue we hit.
#
# Recommended pod: L4 GPU, 12+ vCPU, 80GB+ disk (not 20 or 30 — postgres
# WAL bloat from bulk updates can easily eat 40-50GB).
#
# Usage:
#   1. Start a GPU pod on RunPod with the image above and 80GB+ volume
#   2. SSH in and run:
#      git clone https://github.com/ninthhousestudios/ogham-mcp.git /workspace/ogham-mcp
#      cd /workspace/ogham-mcp && git checkout worktree-colbert-reembed
#      bash scripts/runpod-gpu-setup.sh
#
# After setup completes, run:
#   cd /workspace/ogham-mcp
#   OPENBLAS_NUM_THREADS=4 uv run python scripts/run-benchmark-matrix.py \
#       --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40
#   uv run python scripts/generate-results-table.py

set -euo pipefail

# OpenBLAS tries to spawn 64 threads by default — fails on pods with
# limited vCPU (pthread_create "Resource temporarily unavailable").
# Must be set before scipy is imported.
export OPENBLAS_NUM_THREADS=4

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/ogham-mcp"

echo "=== Step 1: System packages ==="
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib postgresql-server-dev-all git build-essential curl

echo "=== Step 2: pgvector ==="
# pgvector requires PG 13+. Ubuntu 22.04 ships PG 14, which is fine.
# Ubuntu 20.04 ships PG 12 — pgvector will compile but error at runtime.
if ! find /usr/lib/postgresql -name "vector.so" 2>/dev/null | grep -q .; then
    cd /tmp
    git clone --depth 1 https://github.com/pgvector/pgvector.git
    cd pgvector
    make && make install
fi

echo "=== Step 2b: VectorChord ==="
# VectorChord adds native MaxSim (ColBERT late-interaction) indexing to postgres.
# It installs alongside pgvector as a superset — reuses the vector type.
# Pre-built .deb packages are available from GitHub releases.
PG_VER_SHORT=$(pg_lsclusters -h | head -1 | awk '{print $1}')
if ! su - postgres -c "psql -tc \"SELECT 1 FROM pg_available_extensions WHERE name='vchord'\"" 2>/dev/null | grep -q 1; then
    echo "Installing VectorChord from pre-built .deb..."
    VCHORD_VERSION="1.1.1"
    VCHORD_TAG="v${VCHORD_VERSION}"
    VCHORD_DEB="vchord-pg${PG_VER_SHORT}_${VCHORD_VERSION}_amd64.deb"
    VCHORD_URL="https://github.com/tensorchord/VectorChord/releases/download/${VCHORD_TAG}/${VCHORD_DEB}"
    cd /tmp
    curl -fsSL -o "${VCHORD_DEB}" "${VCHORD_URL}" || {
        echo "Failed to download VectorChord .deb. Trying to build from source..."
        # Fallback: build from source (needs Rust toolchain)
        if ! command -v cargo &>/dev/null; then
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi
        git clone --depth 1 --branch "${VCHORD_VERSION}" https://github.com/tensorchord/VectorChord.git /tmp/VectorChord
        cd /tmp/VectorChord
        cargo install cargo-pgrx --version $(grep pgrx Cargo.toml | head -1 | grep -o '"[^"]*"' | tr -d '"') || true
        cargo pgrx init --pg${PG_VER_SHORT}=$(which pg_config)
        cargo pgrx install --sudo --release --pg-config=$(which pg_config)
    }
    dpkg -i "/tmp/${VCHORD_DEB}" 2>/dev/null || true
fi

echo "=== Step 3: Start postgres + create DB ==="
PG_VER=$(pg_lsclusters -h | head -1 | awk '{print $1}')
echo "Found PostgreSQL ${PG_VER}"
pg_ctlcluster "${PG_VER}" main start || true

# TCP auth uses scram-sha-256 by default. Adding trust to pg_hba.conf
# doesn't work (first matching rule wins, scram line is above).
# ALTER USER with a password is the only reliable fix.
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='ogham'\" | grep -q 1 || createdb ogham"
su - postgres -c "psql ogham -c 'CREATE EXTENSION IF NOT EXISTS vector'"
su - postgres -c "psql ogham -c 'CREATE EXTENSION IF NOT EXISTS vchord CASCADE'"
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres'\""

# Reduce WAL retention — default lets WAL grow unbounded under heavy writes.
# The benchmark does ~70K UPDATEs across 24 repooling configs.
su - postgres -c "psql -c \"ALTER SYSTEM SET max_wal_size = '2GB';\""
su - postgres -c "psql -c 'SELECT pg_reload_conf();'"
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

# Verify scipy + OpenBLAS work with our thread limit
uv run python -c "from scipy.cluster.hierarchy import linkage; print('scipy ok')"

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

echo "=== Step 10: Create table + schema ==="
# get_backend() auto-migration only adds columns — it doesn't create the table
# or the search functions. We need both before ingest can work.

# Create table first (IF NOT EXISTS so it's safe to re-run).
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
    colbert_vectors_raw bytea,
    colbert_tokens vector(128)[]
);\""
echo "memories table ready"

# Apply full schema — creates hybrid_search_memories and other functions.
# Uses ON_ERROR_STOP=0 because CREATE TYPE relationship_type will fail if
# it already exists, and we want to continue past that to create the functions.
su - postgres -c "psql -v ON_ERROR_STOP=0 ogham < ${REPO_DIR}/sql/schema_postgres.sql"
echo "Schema functions + indexes applied"

echo "=== Step 11: Ingest BEAM dataset ==="
echo "Ingesting all 100K bucket chats (dense + sparse embeddings)..."
echo "With GPU this takes ~3-5 minutes (~10s/chat)."
echo "Note: output appears per-chat, not per-batch."
uv run python benchmarks/beam_benchmark.py --ingest --bucket 100K --beam-dir /tmp/BEAM

echo "=== Step 11b: Compact postgres ==="
# Bulk inserts generate dead tuples + WAL. VACUUM FULL rewrites the table
# and actually reclaims disk. Regular VACUUM only marks space as reusable.
su - postgres -c "psql ogham -c 'CHECKPOINT;'"
su - postgres -c "psql ogham -c 'VACUUM FULL memories;'"
echo "Postgres compacted"

echo "=== Step 12: Embed raw f32 ColBERT vectors for all profiles ==="
echo "Re-embedding all profiles with raw f32 ColBERT..."
uv run python scripts/embed-colbert-raw.py --all

echo "=== Step 12b: Compact postgres again ==="
su - postgres -c "psql ogham -c 'CHECKPOINT;'"
su - postgres -c "psql ogham -c 'VACUUM FULL memories;'"
echo "Postgres compacted"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the ColBERT reranking benchmark (original):"
echo ""
echo "  cd ${REPO_DIR}"
echo "  OPENBLAS_NUM_THREADS=4 uv run python scripts/run-benchmark-matrix.py \\"
echo "      --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40"
echo ""
echo "Run the ColBERT retrieval benchmark (three-way RRF via VectorChord):"
echo ""
echo "  OPENBLAS_NUM_THREADS=4 uv run python scripts/run-colbert-retrieval-matrix.py \\"
echo "      --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40"
echo ""
echo "Then generate tables:"
echo "  uv run python scripts/generate-results-table.py"
echo ""
echo "TIP: If disk usage climbs during the benchmark, run in another terminal:"
echo "  su - postgres -c \"psql ogham -c 'CHECKPOINT;'\""
echo "  su - postgres -c \"psql ogham -c 'VACUUM FULL memories;'\""
