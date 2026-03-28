#!/bin/bash
# RunPod bootstrap for ColBERT compression benchmark.
#
# Prerequisites:
#   1. Start a CPU pod with 48GB+ RAM on RunPod
#   2. From laptop, dump and copy the DB:
#      pg_dump -U josh -Fc ogham -t memories > /tmp/ogham-memories.dump
#      scp -P <PORT> -i ~/.ssh/id_ed25519 /tmp/ogham-memories.dump root@<HOST>:/workspace/
#   3. SSH in and run this script:
#      bash scripts/runpod-setup.sh
#
# After setup, run:
#   uv run python scripts/run-benchmark-matrix.py --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/ogham-mcp"
DUMP_FILE="${WORKSPACE}/ogham-memories.dump"

echo "=== Step 1: System packages ==="
apt-get update -qq
apt-get install -y -qq postgresql postgresql-contrib git build-essential

echo "=== Step 2: pgvector ==="
if ! find /usr/lib/postgresql -name "vector.so" 2>/dev/null | grep -q .; then
    cd /tmp
    git clone --depth 1 https://github.com/pgvector/pgvector.git
    cd pgvector
    make && make install
fi

echo "=== Step 3: Start postgres + create DB ==="
pg_ctlcluster $(pg_lsclusters -h | head -1 | awk '{print $1, $2}') start || true
su - postgres -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='ogham'\" | grep -q 1 || createdb ogham"
su - postgres -c "psql ogham -c 'CREATE EXTENSION IF NOT EXISTS vector'"

echo "=== Step 4: Clone BEAM dataset ==="
if [ ! -d /tmp/BEAM ]; then
    git clone --depth 1 https://github.com/mohammadtavakoli78/BEAM /tmp/BEAM
fi

echo "=== Step 5: Clone/update ogham-mcp ==="
if [ ! -d "${REPO_DIR}" ]; then
    # Clone from the fork that has the colbert-reembed branch
    git clone https://github.com/ninthhousestudios/ogham-mcp.git "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git checkout worktree-colbert-reembed 2>/dev/null || git checkout main

echo "=== Step 6: Install Python deps ==="
pip install -q uv
uv sync --all-extras

echo "=== Step 7: Configure environment ==="
cat > benchmarks/.env.local << 'ENVEOF'
EMBEDDING_PROVIDER=onnx
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://postgres@localhost/ogham
ENVEOF

echo "=== Step 8: Download ONNX model (if not present) ==="
MODEL_DIR="${HOME}/.cache/ogham/bge-m3-onnx"
if [ ! -f "${MODEL_DIR}/bge_m3_model.onnx" ]; then
    echo "Downloading bge-m3 ONNX model..."
    pip install -q huggingface-hub
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('yuniko-software/bge-m3-onnx', local_dir='${MODEL_DIR}')
"
fi

echo "=== Step 9: Create table structure + add all columns ==="
uv run python -c "
import os
os.environ.setdefault('DATABASE_URL', 'postgresql://postgres@localhost/ogham')
os.environ.setdefault('DATABASE_BACKEND', 'postgres')
os.environ.setdefault('EMBEDDING_PROVIDER', 'onnx')
from ogham.database import get_backend
b = get_backend()
print('Backend ready, table auto-migrated')
# Add colbert_vectors_raw BEFORE restore so the dump data can populate it
b._execute('ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_vectors_raw bytea', fetch='none')
print('colbert_vectors_raw column ready')
"

echo "=== Step 10: Restore database dump ==="
if [ -f "${DUMP_FILE}" ]; then
    echo "Restoring from ${DUMP_FILE}..."
    su - postgres -c "pg_restore -d ogham --data-only --no-owner ${DUMP_FILE}" || {
        echo "pg_restore failed, trying with --clean..."
        su - postgres -c "pg_restore -d ogham --clean --no-owner ${DUMP_FILE}" || true
    }
else
    echo "WARNING: No dump file found at ${DUMP_FILE}"
    echo "Copy it from laptop: scp ogham-memories.dump root@<host>:/workspace/"
fi

echo "=== Step 11: Embed remaining profiles with raw f32 ColBERT ==="
cd "${REPO_DIR}"
# Export env vars for all subsequent Python scripts
export DATABASE_URL="postgresql://postgres@localhost/ogham"
export DATABASE_BACKEND="postgres"
export EMBEDDING_PROVIDER="onnx"

echo "Checking which profiles need raw f32 embedding..."
uv run python -c "
import os
os.environ.setdefault('DATABASE_URL', 'postgresql://postgres@localhost/ogham')
os.environ.setdefault('DATABASE_BACKEND', 'postgres')
os.environ.setdefault('EMBEDDING_PROVIDER', 'onnx')
from ogham.database import get_backend
b = get_backend()
rows = b._execute('''
    SELECT profile, count(*) as total,
           count(colbert_vectors_raw) as has_raw
    FROM memories WHERE profile LIKE 'beam_%%'
    GROUP BY profile ORDER BY profile
''', fetch='all')
need = 0
for r in rows:
    missing = r['total'] - r['has_raw']
    if missing > 0:
        need += missing
        print(f\"  {r['profile']:20s} total={r['total']:4d} has_raw={r['has_raw']:4d} NEED={missing}\")
    else:
        print(f\"  {r['profile']:20s} total={r['total']:4d} has_raw={r['has_raw']:4d} OK\")
print(f'\nTotal needing raw embedding: {need}')
"

echo ""
echo "Running raw f32 embedding for all profiles..."
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
echo ""
echo "Copy results back to laptop:"
echo "  scp -P <PORT> -i ~/.ssh/id_ed25519 root@<HOST>:${REPO_DIR}/benchmarks/beam_results/matrix_100K.json ."
