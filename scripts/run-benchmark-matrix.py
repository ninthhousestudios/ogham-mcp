#!/usr/bin/env python3
"""Run the full ColBERT compression benchmark matrix.

Iterates over all (precision, pool_factor) combinations, repooling and
requantizing from raw f32 source vectors, then running BEAM evaluation.

Produces a single JSON with all results for comparison.

Usage:
    uv run python scripts/run-benchmark-matrix.py --beam-dir /tmp/BEAM --bucket 100K
    uv run python scripts/run-benchmark-matrix.py --beam-dir /tmp/BEAM --bucket 100K --configs f32:1,f16:2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PRECISIONS = ["f32", "f16", "int8_row", "int8_channel"]
POOL_FACTORS = [1, 2, 3, 4, 6, 8]

RESULTS_DIR = Path(__file__).parent.parent / "benchmarks" / "beam_results"


def repool_all(backend, pool_factor: int, precision: str):
    """Repool + requantize all beam profiles' colbert_vectors from raw f32."""
    from ogham.onnx_embedder import repack_colbert, unpack_colbert

    rows = backend._execute(
        "SELECT id, colbert_vectors_raw FROM memories"
        " WHERE profile LIKE 'beam_%%' AND colbert_vectors_raw IS NOT NULL"
        " ORDER BY id",
        fetch="all",
    )
    total = len(rows)
    log.info("Repooling %d memories: pool=%d, precision=%s", total, pool_factor, precision)

    if total == 0:
        log.warning("No memories with colbert_vectors_raw found")
        return 0.0

    total_bytes = 0
    t0 = time.monotonic()

    for i, row in enumerate(rows):
        raw_vecs = unpack_colbert(row["colbert_vectors_raw"])
        repacked = repack_colbert(raw_vecs, pool_factor, precision)
        total_bytes += len(repacked)

        backend._execute(
            "UPDATE memories SET colbert_vectors = %(packed)s WHERE id = %(id)s",
            {"id": row["id"], "packed": repacked},
            fetch="none",
        )

        if (i + 1) % 200 == 0:
            gc.collect()
            elapsed = time.monotonic() - t0
            log.info("  %d/%d (%.1f/s)", i + 1, total, (i + 1) / elapsed)

    gc.collect()
    elapsed = time.monotonic() - t0
    avg_bytes = total_bytes / total
    log.info("Repooled %d memories in %.0fs, avg %.0f bytes/memory", total, elapsed, avg_bytes)
    return avg_bytes


def measure_storage(backend) -> float:
    """Get average colbert_vectors size in bytes across beam profiles."""
    rows = backend._execute(
        "SELECT avg(length(colbert_vectors)) as avg_bytes"
        " FROM memories WHERE profile LIKE 'beam_%%' AND colbert_vectors IS NOT NULL",
        fetch="all",
    )
    return float(rows[0]["avg_bytes"]) if rows and rows[0]["avg_bytes"] else 0


def run_eval(beam_dir: Path, bucket: str, search_mode: str, top_k: int = 10,
             cached_embeddings: dict | None = None) -> dict:
    """Run BEAM evaluation and return results dict."""
    from beam_benchmark import evaluate_bucket
    return evaluate_bucket(
        beam_dir, bucket, search_mode=search_mode, top_k=top_k,
        cached_embeddings=cached_embeddings,
    )


def precompute_query_embeddings(beam_dir: Path, bucket: str) -> dict:
    """Pre-compute all query embeddings once for reuse across configs."""
    from beam_benchmark import list_chats, load_probing_questions, ALL_CATEGORIES, EMBEDDING_BATCH_SIZE, _with_retry
    from ogham.embeddings import generate_embeddings_batch
    from ogham.onnx_embedder import encode_query_colbert

    all_questions = []
    for chat_id in list_chats(beam_dir, bucket):
        questions = load_probing_questions(beam_dir, bucket, chat_id)
        for cat in ALL_CATEGORIES:
            for q in questions.get(cat, []):
                all_questions.append(q)

    query_texts = [q["question"] for q in all_questions]
    log.info("Pre-computing %d query embeddings (dense + ColBERT)...", len(query_texts))

    # Dense embeddings
    dense = _with_retry(generate_embeddings_batch, query_texts, batch_size=EMBEDDING_BATCH_SIZE)
    log.info("Dense query embeddings ready")

    # ColBERT query vectors
    colbert = [None] * len(query_texts)
    for i, qt in enumerate(query_texts):
        colbert[i] = encode_query_colbert(qt)
        if (i + 1) % 50 == 0:
            log.info("  ColBERT query %d/%d", i + 1, len(query_texts))
    log.info("ColBERT query embeddings ready")

    return {
        "dense": dense,
        "sparse": [None] * len(query_texts),
        "colbert": colbert,
    }


def parse_configs(config_str: str) -> list[tuple[str, int]]:
    """Parse 'f32:1,f16:2,int8_row:4' into [(precision, pool_factor), ...]."""
    configs = []
    for item in config_str.split(","):
        precision, factor = item.strip().split(":")
        configs.append((precision, int(factor)))
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run full ColBERT compression benchmark matrix")
    parser.add_argument("--beam-dir", type=Path, default=Path("/tmp/BEAM"), help="Path to BEAM repo")
    parser.add_argument("--bucket", default="100K", help="BEAM bucket (default: 100K)")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for retrieval")
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Specific configs as 'precision:pool_factor,...' (default: full matrix)",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip tsvector and sparse baselines",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON path (default: beam_results/matrix_{bucket}.json)",
    )
    parser.add_argument(
        "--mem-limit-gb", type=int, default=8,
        help="Virtual memory limit in GB (default: 8)",
    )
    args = parser.parse_args()

    # Memory cap — use RLIMIT_DATA instead of RLIMIT_AS to avoid killing
    # ONNX runtime's mmap allocations (AS counts virtual space, DATA is heap only)
    mem_bytes = args.mem_limit_gb * 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
        log.info("Memory limit (RLIMIT_DATA) set to %d GB", args.mem_limit_gb)
    except (ValueError, OSError) as e:
        log.warning("Could not set memory limit: %s (continuing without limit)", e)

    # Load benchmark env
    env_file = Path(__file__).parent.parent / "benchmarks" / ".env.local"
    if env_file.exists():
        log.info("Loading env from %s", env_file)
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    output_path = args.output or RESULTS_DIR / f"matrix_{args.bucket}.json"

    # Build config list
    if args.configs:
        colbert_configs = parse_configs(args.configs)
    else:
        colbert_configs = [
            (precision, pool_factor)
            for precision in PRECISIONS
            for pool_factor in POOL_FACTORS
        ]

    from ogham.database import get_backend
    backend = get_backend()

    # Pre-compute query embeddings once
    log.info("=" * 70)
    log.info("PRE-COMPUTING QUERY EMBEDDINGS")
    log.info("=" * 70)
    cached = precompute_query_embeddings(args.beam_dir, args.bucket)

    all_results = {}
    total_start = time.monotonic()

    # Baselines
    if not args.skip_baselines:
        for mode in ["tsvector", "sparse"]:
            log.info("=" * 70)
            log.info("BASELINE: %s", mode)
            log.info("=" * 70)
            try:
                results = run_eval(args.beam_dir, args.bucket, search_mode=mode,
                                  top_k=args.top_k, cached_embeddings=cached)
                all_results[mode] = {
                    "config": {"search_mode": mode},
                    "overall": results.get("overall", {}),
                    "latency": results.get("latency", {}),
                    "per_category": results.get("per_category", {}),
                    "questions_evaluated": results.get("questions_evaluated", 0),
                    "elapsed_seconds": results.get("elapsed_seconds", 0),
                }
            except Exception as e:
                log.error("Baseline %s failed: %s", mode, e)
                all_results[mode] = {"error": str(e)}
            gc.collect()

    # ColBERT configs
    for i, (precision, pool_factor) in enumerate(colbert_configs):
        config_name = f"{precision}_pool{pool_factor}"
        log.info("=" * 70)
        log.info("CONFIG %d/%d: %s", i + 1, len(colbert_configs), config_name)
        log.info("=" * 70)

        try:
            # Step 1: Repool/requantize
            avg_bytes = repool_all(backend, pool_factor, precision)

            # Step 2: Run eval (always uses colbert_pooled mode which reads colbert_vectors)
            results = run_eval(
                args.beam_dir, args.bucket,
                search_mode="colbert_pooled", top_k=args.top_k,
                cached_embeddings=cached,
            )

            all_results[config_name] = {
                "config": {
                    "precision": precision,
                    "pool_factor": pool_factor,
                    "search_mode": "colbert_pooled",
                },
                "storage_bytes_avg": round(avg_bytes, 1),
                "overall": results.get("overall", {}),
                "latency": results.get("latency", {}),
                "per_category": results.get("per_category", {}),
                "questions_evaluated": results.get("questions_evaluated", 0),
                "elapsed_seconds": results.get("elapsed_seconds", 0),
            }
        except Exception as e:
            log.error("Config %s failed: %s", config_name, e, exc_info=True)
            all_results[config_name] = {"error": str(e)}

        gc.collect()

    total_elapsed = time.monotonic() - total_start

    # Summary
    summary = {
        "benchmark": "BEAM ColBERT compression matrix",
        "bucket": args.bucket,
        "top_k": args.top_k,
        "total_configs": len(colbert_configs) + (0 if args.skip_baselines else 2),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": all_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Matrix results saved to %s", output_path)

    # Print quick summary table
    print("\n" + "=" * 90)
    print("MATRIX SUMMARY")
    print("=" * 90)
    print(f"{'Config':<20s} {'R@5':>8s} {'R@10':>8s} {'R@20':>8s} {'nDCG@10':>8s} {'MRR':>8s} {'Bytes':>8s}")
    print("-" * 90)
    for name, data in all_results.items():
        if "error" in data:
            print(f"{name:<20s} ERROR: {data['error'][:60]}")
            continue
        o = data.get("overall", {})
        storage = data.get("storage_bytes_avg", 0)
        print(
            f"{name:<20s} {o.get('recall@5', 0):>8.4f} {o.get('recall@10', 0):>8.4f} "
            f"{o.get('recall@20', 0):>8.4f} {o.get('ndcg@10', 0):>8.4f} "
            f"{o.get('mrr', 0):>8.4f} {storage:>8.0f}"
        )
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
