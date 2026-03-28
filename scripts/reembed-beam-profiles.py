#!/usr/bin/env python3
"""Re-embed all beam profiles with ONNX dense + sparse + ColBERT vectors.

Crash-safe: completed chunks are committed to DB immediately.
Skips memories that already have colbert_vectors set.

Usage:
    uv run python scripts/reembed-beam-profiles.py [--profile beam_100K_3]
    uv run python scripts/reembed-beam-profiles.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time

# Ensure the project src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def get_beam_profiles(backend, single: str | None = None) -> list[str]:
    """Get sorted list of beam profile names."""
    if single:
        return [single]
    rows = backend._execute(
        "SELECT DISTINCT profile FROM memories WHERE profile LIKE 'beam_%%' ORDER BY profile",
        fetch="all",
    )
    return [r["profile"] for r in rows]


def get_memories_needing_reembed(backend, profile: str) -> list[dict]:
    """Get memories that lack colbert_vectors."""
    return backend._execute(
        "SELECT id, content FROM memories"
        " WHERE profile = %(profile)s AND colbert_vectors IS NULL"
        " ORDER BY id",
        {"profile": profile},
        fetch="all",
    )


def reembed_profile(backend, profile: str, chunk_size: int, dry_run: bool) -> dict:
    """Re-embed one profile, returning stats."""
    memories = get_memories_needing_reembed(backend, profile)
    total = len(memories)
    if total == 0:
        log.info("  %s: nothing to do (all memories have colbert vectors)", profile)
        return {"profile": profile, "total": 0, "embedded": 0, "skipped": total}

    if dry_run:
        log.info("  %s: would re-embed %d memories", profile, total)
        return {"profile": profile, "total": total, "embedded": 0, "dry_run": True}

    from ogham.database import batch_update_embeddings
    from ogham.embeddings import generate_embeddings_batch_full

    log.info("  %s: re-embedding %d memories in chunks of %d", profile, total, chunk_size)

    embedded = 0
    sparse_count = 0
    colbert_count = 0
    t0 = time.monotonic()

    for chunk_start in range(0, total, chunk_size):
        chunk = memories[chunk_start : chunk_start + chunk_size]
        chunk_texts = [m["content"] for m in chunk]
        chunk_ids = [m["id"] for m in chunk]

        full_results = generate_embeddings_batch_full(
            chunk_texts, include_colbert=True
        )

        # Write dense
        batch_update_embeddings(chunk_ids, [emb for emb, _, _ in full_results])

        # Write sparse + colbert
        for mem_id, (_, sparse_str, colbert_bytes) in zip(chunk_ids, full_results):
            updates = []
            params: dict = {"id": mem_id}

            if sparse_str is not None:
                updates.append("sparse_embedding = %(sparse)s::sparsevec")
                params["sparse"] = sparse_str
                sparse_count += 1

            if colbert_bytes is not None:
                updates.append("colbert_vectors = %(colbert)s")
                params["colbert"] = colbert_bytes
                colbert_count += 1

            if updates:
                backend._execute(
                    f"UPDATE memories SET {', '.join(updates)} WHERE id = %(id)s",
                    params,
                    fetch="none",
                )

        # Free colbert bytes immediately
        del full_results
        gc.collect()

        embedded += len(chunk)
        elapsed = time.monotonic() - t0
        rate = embedded / elapsed if elapsed > 0 else 0
        eta = (total - embedded) / rate if rate > 0 else 0
        log.info(
            "    %d/%d done (%.2f/s, ETA %.0fm%.0fs)",
            embedded, total, rate, eta // 60, eta % 60,
        )

    elapsed = time.monotonic() - t0
    log.info(
        "  %s: done — %d memories in %.0fs (%.2f/s), sparse=%d, colbert=%d",
        profile, embedded, elapsed, embedded / elapsed if elapsed > 0 else 0,
        sparse_count, colbert_count,
    )
    return {
        "profile": profile,
        "total": total,
        "embedded": embedded,
        "sparse": sparse_count,
        "colbert": colbert_count,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-embed beam profiles with ONNX")
    parser.add_argument("--profile", help="Single profile to re-embed (default: all beam_*)")
    parser.add_argument("--chunk-size", type=int, default=5, help="Memories per chunk (default: 5)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    # Ensure colbert_vectors column exists
    from ogham.database import get_backend
    backend = get_backend()

    log.info("Adding colbert_vectors column if not present...")
    backend._execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_vectors bytea",
        fetch="none",
    )

    profiles = get_beam_profiles(backend, args.profile)
    log.info("Found %d beam profiles to process", len(profiles))

    results = []
    t_start = time.monotonic()

    for profile in profiles:
        result = reembed_profile(backend, profile, args.chunk_size, args.dry_run)
        results.append(result)

    total_elapsed = time.monotonic() - t_start
    total_embedded = sum(r.get("embedded", 0) for r in results)
    log.info(
        "All done: %d memories across %d profiles in %.0fs",
        total_embedded, len(profiles), total_elapsed,
    )


if __name__ == "__main__":
    main()
