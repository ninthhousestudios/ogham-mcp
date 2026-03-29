# ColBERT Compression Benchmark — CPU RunPod Reference

Attempted 2026-03-29. Did not complete due to embedding speed on CPU (~15 min/chat, ~5 hours total for 20 chats).

## Pod Configuration

- **Image**: `runpod/base:0.7.0-ubuntu2204` (Ubuntu 22.04)
- **Type**: CPU pod, 48GB+ RAM
- **No GPU required** — ONNX BGE-M3 runs on CPU via onnxruntime

## Setup Script

See `scripts/runpod-setup.sh` — handles everything end-to-end:

1. System packages (postgresql, pgvector build deps)
2. Build + install pgvector from source (PG 14 on Ubuntu 22.04)
3. Start postgres, create `ogham` DB, enable vector extension
4. Clone BEAM dataset to `/tmp/BEAM`
5. Install uv, Python 3.13, project deps
6. Configure env (ONNX provider, postgres URL)
7. Download BGE-M3 ONNX model (~2.2GB from HuggingFace)
8. Create table structure via `get_backend()`
9. Ingest BEAM 100K bucket (20 chats, 2,960 memories) — dense + sparse embeddings
10. Embed raw f32 ColBERT vectors for all profiles
11. Ready to run benchmark matrix

## Performance on CPU

- ONNX BGE-M3 embedding: ~15 minutes per chat (148 memories avg)
- Total BEAM ingest: ~5 hours for 20 chats
- Raw ColBERT embedding (step 11): additional ~5 hours
- Benchmark matrix (26 configs): unknown, did not reach this step
- Process used ~764% CPU (8 threads), ~3.3GB RSS

## Troubleshooting Log — Everything That Went Wrong

### Attempt 1: Ubuntu 20.04 (`runpod/base:0.7.0-ubuntu2004`)

**pgvector build failed** — `postgres.h: No such file or directory`. Missing `postgresql-server-dev-all`.
Installed `postgresql-server-dev-12`, pgvector compiled but then errored:
```
#error "Requires PostgreSQL 13+"
```
Ubuntu 20.04 ships PG 12. pgvector needs PG 13+. Dead end.

Tried adding PG 16 via PGDG apt repo — `apt-key` succeeded but `focal-pgdg` 404'd.
`apt-get install postgresql-16` failed: "unable to locate package". Abandoned this image.

### Attempt 2: `postgres:16` Docker image

Tried using official Docker Hub postgres image. RunPod rate-limited by Docker Hub on image pull.
Pod wouldn't start. Abandoned.

### Attempt 3: Ubuntu 22.04 (`runpod/base:0.7.0-ubuntu2204`)

This worked. PG 14 in default repos, pgvector builds fine.

### Python 3.14 nightmare

uv aggressively downloads Python 3.14 beta even when 3.13 is available. pydantic breaks immediately:
```
TypeError: _eval_type() got an unexpected keyword argument 'prefer_fwd_module'
```

**Failed fixes (in order):**
1. `uv venv --python 3.13` — uv sync still pulled 3.14
2. `rm -rf .venv && uv venv --python 3.13` — same result
3. `UV_PYTHON=3.13 uv sync` — still downloaded 3.14
4. `rm -rf /root/.local/share/uv/python/cpython-3.14*` — uv re-downloaded it on sync
5. `uv python install 3.13 && uv venv --python /usr/bin/python3.13` — uv sync STILL went to 3.14
6. Various combinations of the above

**Actual fix**: cap `requires-python = ">=3.13,<3.14"` in `pyproject.toml`. Only way to prevent uv from using 3.14. The `rm -rf cpython-3.14*` also helps (prevents reinstall) but without the pyproject cap, uv re-downloads it.

### PostgreSQL authentication

Default postgres on Ubuntu uses scram-sha-256 for TCP and peer for unix sockets.

**Failed fixes:**
1. Added `host all all 127.0.0.1/32 trust` to bottom of `pg_hba.conf` — first matching rule wins, scram line was above it
2. `sed -i 's/scram-sha-256/trust/g' pg_hba.conf` + reload — still failed (unclear why, possibly cached connections)
3. `psql` via unix socket as root — `peer authentication failed` (running as root, user is postgres)

**Actual fix**: `su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres'\""` then use `postgresql://postgres:postgres@localhost/ogham` everywhere.

### SQLite `unixepoch()` not available

`embedding_cache.py` used `unixepoch('now')` which requires SQLite 3.38+.
RunPod's Ubuntu 22.04 has an older SQLite. Error: `unknown function: unixepoch()`.

**Fix**: replaced with `strftime('%s', 'now')` which works on all SQLite versions.

### Missing `memories` table on fresh DB

Step 9 originally ran `ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_vectors_raw bytea` before any data existed.
Error: `relation "memories" does not exist`.

**Fix**: moved ALTER TABLE to after BEAM ingest (which creates the table via `store_memories_batch`). Also `embed-colbert-raw.py` does its own ALTER TABLE, so the separate step in the setup script was redundant.

### BEAM ingest has no per-batch output

`beam_benchmark.py --ingest` logs per-chat, not per-batch. With ~15 min/chat on CPU, you see nothing for 15 minutes. Not a bug — just looks like it's hung.
Confirm it's working with `ps aux | grep python` (should show ~764% CPU, ~3.3GB RSS).

## Key Fixes Summary

- **Python 3.14**: `requires-python = ">=3.13,<3.14"` in pyproject.toml + `rm -rf cpython-3.14*`
- **SQLite**: `strftime('%s', 'now')` instead of `unixepoch('now')` in `embedding_cache.py`
- **PostgreSQL auth**: `ALTER USER postgres PASSWORD 'postgres'`, use password in DATABASE_URL
- **pgvector**: needs PG 13+, use Ubuntu 22.04+ (has PG 14)
- **Docker images**: use RunPod's own images, Docker Hub rate limits

## Commands After Setup

```bash
cd /workspace/ogham-mcp

# Run the 26-config benchmark matrix
uv run python scripts/run-benchmark-matrix.py --beam-dir /tmp/BEAM --bucket 100K --mem-limit-gb 40

# Generate results tables
uv run python scripts/generate-results-table.py
```

## Conclusion

CPU is too slow for this benchmark. Use GPU pod instead — see `scripts/runpod-gpu-setup.sh`.
