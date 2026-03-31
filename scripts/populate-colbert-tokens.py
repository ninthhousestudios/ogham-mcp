#!/usr/bin/env python3
"""Populate colbert_tokens (vector[]) column from raw ColBERT bytea vectors.

Converts ogham's custom-packed colbert_vectors_raw (float32 bytea) into
pgvector vector(128)[] arrays suitable for VectorChord's vchordrq MaxSim index.

Supports pooling and quantization — the same compression matrix as the
reranking benchmark, but stored in VectorChord's native format instead of
ogham's bytea packing.

Usage:
    # Populate with default settings (no pooling, f32):
    uv run python scripts/populate-colbert-tokens.py

    # Populate with specific pool factor and precision:
    uv run python scripts/populate-colbert-tokens.py --pool-factor 4 --precision int8_channel

    # Rebuild the MaxSim index after population:
    uv run python scripts/populate-colbert-tokens.py --reindex
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def vectors_to_pg_array(vectors) -> list[str]:
    """Convert [n_tokens, dim] numpy array to unquoted vector literal strings.

    Returns a list of strings like '[0.1,0.2,...]' — callers must wrap each
    in single quotes and ::vector(128) cast for SQL use.
    """
    parts = []
    for row in vectors:
        vec_str = "[" + ",".join(f"{v:.6f}" for v in row) + "]"
        parts.append(vec_str)
    return parts


def populate_all(backend, pool_factor: int, precision: str) -> tuple[int, float]:
    """Populate colbert_tokens for all beam memories from colbert_vectors_raw.

    Returns (count, avg_tokens_per_memory).
    """
    from ogham.onnx_embedder import pool_colbert, unpack_colbert

    import numpy as np

    rows = backend._execute(
        "SELECT id, colbert_vectors_raw FROM memories"
        " WHERE profile LIKE 'beam_%%' AND colbert_vectors_raw IS NOT NULL"
        " ORDER BY id",
        fetch="all",
    )
    total = len(rows)
    if total == 0:
        log.warning("No memories with colbert_vectors_raw found")
        return 0, 0.0

    log.info("Populating colbert_tokens for %d memories: pool=%d, precision=%s",
             total, pool_factor, precision)

    total_tokens = 0
    t0 = time.monotonic()

    for i, row in enumerate(rows):
        raw_vecs = unpack_colbert(row["colbert_vectors_raw"])

        # Apply pooling
        if pool_factor > 1:
            vecs = pool_colbert(raw_vecs, pool_factor)
        else:
            vecs = raw_vecs.astype(np.float32)

        # Apply quantization (dequantized back to float32 for storage —
        # we want to measure quantization's effect on retrieval quality,
        # and VectorChord stores vector(128) as float4 internally).
        if precision == "int8_row":
            row_max = np.max(np.abs(vecs), axis=1)
            scales = np.maximum(row_max / 127.0, 1e-8)
            quantized = np.round(vecs / scales[:, None]).clip(-127, 127).astype(np.int8)
            vecs = quantized.astype(np.float32) * scales[:, None]
        elif precision == "int8_channel":
            col_max = np.max(np.abs(vecs), axis=0)
            scales = np.maximum(col_max / 127.0, 1e-8)
            quantized = np.round(vecs / scales[None, :]).clip(-127, 127).astype(np.int8)
            vecs = quantized.astype(np.float32) * scales[None, :]
        elif precision == "f16":
            vecs = vecs.astype(np.float16).astype(np.float32)
        # f32: no-op

        # Guard against corrupt vectors
        if not np.all(np.isfinite(vecs)):
            log.warning("Non-finite values in ColBERT vectors for id %s — skipping", row["id"])
            continue

        if len(vecs) == 0:
            log.warning("Empty token list for id %s — skipping", row["id"])
            continue

        # Convert to postgres vector[] literal
        vec_literals = vectors_to_pg_array(vecs)
        n_tokens = len(vec_literals)
        total_tokens += n_tokens

        # Build the SQL array expression
        array_expr = "ARRAY[" + ",".join(
            f"'{v}'::vector(128)" for v in vec_literals
        ) + "]"

        backend._execute(
            f"UPDATE memories SET colbert_tokens = {array_expr} WHERE id = %(id)s",
            {"id": row["id"]},
            fetch="none",
        )

        if (i + 1) % 100 == 0:
            gc.collect()
            elapsed = time.monotonic() - t0
            log.info("  %d/%d (%.1f/s)", i + 1, total, (i + 1) / elapsed)

    gc.collect()
    elapsed = time.monotonic() - t0
    avg_tokens = total_tokens / total if total > 0 else 0
    log.info("Populated %d memories in %.0fs (avg %.0f tokens/memory)",
             total, elapsed, avg_tokens)
    return total, avg_tokens


def reindex(backend):
    """Drop and recreate the VectorChord MaxSim index."""
    log.info("Rebuilding MaxSim index...")
    t0 = time.monotonic()
    backend._execute("DROP INDEX IF EXISTS memories_colbert_maxsim_idx", fetch="none")
    backend._execute(
        "CREATE INDEX memories_colbert_maxsim_idx"
        " ON memories USING vchordrq (colbert_tokens vector_maxsim_ops)",
        fetch="none",
    )
    elapsed = time.monotonic() - t0
    log.info("Index rebuilt in %.1fs", elapsed)


def clear_tokens(backend):
    """Clear colbert_tokens for all beam memories (before repopulation)."""
    backend._execute(
        "UPDATE memories SET colbert_tokens = NULL WHERE profile LIKE 'beam_%%'",
        fetch="none",
    )
    log.info("Cleared colbert_tokens for all beam memories")


def main():
    parser = argparse.ArgumentParser(
        description="Populate colbert_tokens column for VectorChord MaxSim retrieval"
    )
    parser.add_argument("--pool-factor", type=int, default=1, help="Token pooling factor (default: 1 = no pooling)")
    parser.add_argument("--precision", default="f32",
                        choices=["f32", "f16", "int8_row", "int8_channel"],
                        help="Quantization precision (default: f32)")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the MaxSim index after population")
    parser.add_argument("--reindex-only", action="store_true", help="Only rebuild the index, don't populate")
    args = parser.parse_args()

    # Load benchmark env
    from pathlib import Path
    env_file = Path(__file__).parent.parent / "benchmarks" / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    from ogham.database import get_backend
    backend = get_backend()

    # Ensure column exists
    backend._execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_tokens vector(128)[]",
        fetch="none",
    )

    if args.reindex_only:
        reindex(backend)
        return

    clear_tokens(backend)
    count, avg_tokens = populate_all(backend, args.pool_factor, args.precision)

    if args.reindex or count > 0:
        reindex(backend)

    log.info("Done: %d memories, avg %.0f tokens, pool=%d, precision=%s",
             count, avg_tokens, args.pool_factor, args.precision)


if __name__ == "__main__":
    main()
