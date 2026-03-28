#!/usr/bin/env python3
"""Derive any (pool_factor, precision) ColBERT config from raw f32 vectors.

Reads colbert_vectors_raw (immutable source of truth), applies pooling +
quantization, writes result to colbert_vectors (working column) for benchmarking.

Usage:
    uv run python scripts/repool-requantize.py --pool-factor 4 --precision f16
    uv run python scripts/repool-requantize.py --pool-factor 1 --precision f32  # identity (quality ceiling)
    uv run python scripts/repool-requantize.py --pool-factor 2 --precision int8_row
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

PRECISIONS = ["f32", "f16", "int8_row", "int8_channel"]


def repool_profile(backend, profile: str, pool_factor: int, precision: str, chunk_size: int) -> dict:
    """Repool + requantize one profile's ColBERT vectors."""
    from ogham.onnx_embedder import repack_colbert, unpack_colbert

    rows = backend._execute(
        "SELECT id, colbert_vectors_raw FROM memories"
        " WHERE profile = %(profile)s AND colbert_vectors_raw IS NOT NULL"
        " ORDER BY id",
        {"profile": profile},
        fetch="all",
    )
    total = len(rows)
    if total == 0:
        log.info("  %s: no raw vectors, skipping", profile)
        return {"profile": profile, "total": 0, "processed": 0}

    log.info("  %s: repooling %d memories (pool=%d, precision=%s)", profile, total, pool_factor, precision)
    t0 = time.monotonic()
    total_bytes = 0
    processed = 0

    for chunk_start in range(0, total, chunk_size):
        chunk = rows[chunk_start : chunk_start + chunk_size]

        for row in chunk:
            raw_vecs = unpack_colbert(row["colbert_vectors_raw"])
            repacked = repack_colbert(raw_vecs, pool_factor, precision)
            total_bytes += len(repacked)

            backend._execute(
                "UPDATE memories SET colbert_vectors = %(packed)s WHERE id = %(id)s",
                {"id": row["id"], "packed": repacked},
                fetch="none",
            )

        processed += len(chunk)
        del chunk
        gc.collect()

        elapsed = time.monotonic() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        log.info("    %d/%d (%.1f/s)", processed, total, rate)

    elapsed = time.monotonic() - t0
    avg_bytes = total_bytes / total if total > 0 else 0
    log.info(
        "  %s: done -- %d memories, avg %.0f bytes/memory, %.0fs",
        profile, total, avg_bytes, elapsed,
    )
    return {
        "profile": profile,
        "total": total,
        "processed": processed,
        "avg_bytes": round(avg_bytes, 1),
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Repool + requantize ColBERT vectors from raw f32 source",
    )
    parser.add_argument("--pool-factor", type=int, required=True, help="Token pooling factor (1=none)")
    parser.add_argument(
        "--precision", required=True, choices=PRECISIONS,
        help="Quantization precision",
    )
    parser.add_argument("--profile", help="Single profile (default: all beam profiles)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Memories per chunk")
    args = parser.parse_args()

    from ogham.database import get_backend
    backend = get_backend()

    if args.profile:
        profiles = [args.profile]
    else:
        rows = backend._execute(
            "SELECT DISTINCT profile FROM memories"
            " WHERE profile LIKE 'beam_%%' AND colbert_vectors_raw IS NOT NULL"
            " ORDER BY profile",
            fetch="all",
        )
        profiles = [r["profile"] for r in rows]

    log.info(
        "Repooling %d profiles: pool_factor=%d, precision=%s",
        len(profiles), args.pool_factor, args.precision,
    )

    results = []
    t_start = time.monotonic()

    for profile in profiles:
        result = repool_profile(backend, profile, args.pool_factor, args.precision, args.chunk_size)
        results.append(result)

    total_elapsed = time.monotonic() - t_start
    total_processed = sum(r["processed"] for r in results)
    total_bytes_sum = sum(r.get("avg_bytes", 0) * r["total"] for r in results)
    overall_avg = total_bytes_sum / total_processed if total_processed > 0 else 0

    log.info(
        "All done: %d memories across %d profiles in %.0fs, avg %.0f bytes/memory",
        total_processed, len(profiles), total_elapsed, overall_avg,
    )
    # Print machine-readable summary for the matrix runner
    print(f"AVG_BYTES={overall_avg:.1f}")


if __name__ == "__main__":
    main()
