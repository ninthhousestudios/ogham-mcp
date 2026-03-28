#!/usr/bin/env python3
"""Embed raw (unpooled, float32) ColBERT vectors for benchmark comparison.

Stores in colbert_vectors_raw column alongside existing pooled float16 vectors.
This is for benchmarking only — raw vectors are ~8x larger than pooled.

Usage:
    uv run python scripts/embed-colbert-raw.py [--profile beam_100K_3]
    uv run python scripts/embed-colbert-raw.py --dry-run
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

# Default profiles selected for benchmark: mix of sizes and content lengths
DEFAULT_PROFILES = [
    "beam_100K_4",   # 106 memories, short content
    "beam_100K_6",   # 129 memories, long content
    "beam_100K_13",  # 155 memories, short content
    "beam_100K_1",   # 188 memories, long content
    "beam_100K_11",  # 194 memories, medium content
]


def get_memories_needing_raw(backend, profile: str) -> list[dict]:
    """Get memories that lack colbert_vectors_raw."""
    return backend._execute(
        "SELECT id, content FROM memories"
        " WHERE profile = %(profile)s AND colbert_vectors_raw IS NULL"
        " ORDER BY id",
        {"profile": profile},
        fetch="all",
    )


def embed_profile_raw(backend, profile: str, chunk_size: int, dry_run: bool) -> dict:
    """Embed one profile with raw float32 ColBERT vectors."""
    memories = get_memories_needing_raw(backend, profile)
    total = len(memories)
    if total == 0:
        log.info("  %s: nothing to do", profile)
        return {"profile": profile, "total": 0, "embedded": 0}

    if dry_run:
        log.info("  %s: would embed %d memories", profile, total)
        return {"profile": profile, "total": total, "embedded": 0, "dry_run": True}

    import numpy as np
    from ogham.config import settings
    from ogham.onnx_embedder import _get_model, pack_colbert_raw

    log.info("  %s: embedding %d memories (raw f32) in chunks of %d", profile, total, chunk_size)

    model_path = settings.onnx_model_path or None
    tokenizer, session = _get_model(model_path)
    embedded = 0
    t0 = time.monotonic()

    for chunk_start in range(0, total, chunk_size):
        chunk = memories[chunk_start : chunk_start + chunk_size]

        for mem in chunk:
            encoded = tokenizer.encode(mem["content"])
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
            outputs = session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            colbert_vecs = outputs[2][0].astype(np.float32)
            raw_bytes = pack_colbert_raw(colbert_vecs)

            backend._execute(
                "UPDATE memories SET colbert_vectors_raw = %(raw)s WHERE id = %(id)s",
                {"id": mem["id"], "raw": raw_bytes},
                fetch="none",
            )

        del chunk
        gc.collect()

        embedded += len(memories[chunk_start : chunk_start + chunk_size])
        elapsed = time.monotonic() - t0
        rate = embedded / elapsed if elapsed > 0 else 0
        eta = (total - embedded) / rate if rate > 0 else 0
        log.info(
            "    %d/%d done (%.2f/s, ETA %.0fm%.0fs)",
            embedded, total, rate, eta // 60, eta % 60,
        )

    elapsed = time.monotonic() - t0
    log.info(
        "  %s: done -- %d memories in %.0fs (%.2f/s)",
        profile, embedded, elapsed, embedded / elapsed if elapsed > 0 else 0,
    )
    return {"profile": profile, "total": total, "embedded": embedded, "elapsed_s": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Embed raw ColBERT vectors for benchmarking")
    parser.add_argument("--profile", help="Single profile (default: 5 selected profiles)")
    parser.add_argument("--all", action="store_true", help="Process ALL beam profiles (not just default 5)")
    parser.add_argument("--chunk-size", type=int, default=5, help="Memories per progress report")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from ogham.database import get_backend
    backend = get_backend()

    log.info("Adding colbert_vectors_raw column if not present...")
    backend._execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_vectors_raw bytea",
        fetch="none",
    )

    if args.profile:
        profiles = [args.profile]
    elif args.all:
        rows = backend._execute(
            "SELECT DISTINCT profile FROM memories WHERE profile LIKE 'beam_%%' ORDER BY profile",
            fetch="all",
        )
        profiles = [r["profile"] for r in rows]
    else:
        profiles = DEFAULT_PROFILES
    log.info("Processing %d profiles", len(profiles))

    results = []
    t_start = time.monotonic()

    for profile in profiles:
        result = embed_profile_raw(backend, profile, args.chunk_size, args.dry_run)
        results.append(result)

    total_elapsed = time.monotonic() - t_start
    total_embedded = sum(r.get("embedded", 0) for r in results)
    log.info("All done: %d memories across %d profiles in %.0fs", total_embedded, len(profiles), total_elapsed)


if __name__ == "__main__":
    main()
