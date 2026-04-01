#!/usr/bin/env python3
"""Run ColBERT-as-retrieval benchmark matrix via VectorChord MaxSim.

Tests ColBERT as a primary retrieval leg (three-way RRF: dense + keyword +
ColBERT MaxSim) instead of as a reranker. Uses VectorChord's vchordrq index
for native MaxSim ANN search.

For each (precision, pool_factor) config:
  1. Populate colbert_tokens vector[] column from raw f32 source
  2. Rebuild the VectorChord MaxSim index
  3. Run BEAM evaluation via hybrid_search_memories_colbert()
  4. Record results

Produces a JSON with all results for comparison against the reranking matrix.

Usage:
    uv run python scripts/run-colbert-retrieval-matrix.py --beam-dir /tmp/BEAM --bucket 100K
    uv run python scripts/run-colbert-retrieval-matrix.py --beam-dir /tmp/BEAM --bucket 100K --configs f32:1,f16:2
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


def populate_colbert_tokens(backend, pool_factor: int, precision: str) -> tuple[float, float]:
    """Populate colbert_tokens column for all beam memories.

    Returns (avg_tokens_per_memory, elapsed_seconds).
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
        return 0.0, 0.0

    log.info("Populating colbert_tokens: pool=%d, precision=%s (%d memories)",
             pool_factor, precision, total)

    # Clear existing tokens
    backend._execute(
        "UPDATE memories SET colbert_tokens = NULL WHERE profile LIKE 'beam_%%'",
        fetch="none",
    )

    total_tokens = 0
    t0 = time.monotonic()

    for i, row in enumerate(rows):
        raw_vecs = unpack_colbert(row["colbert_vectors_raw"])

        # Apply pooling
        if pool_factor > 1:
            vecs = pool_colbert(raw_vecs, pool_factor)
        else:
            vecs = raw_vecs.astype(np.float32)

        # Apply quantization (dequantized for storage — VectorChord
        # stores float4 internally, but we want to measure the lossy
        # effect of quantize->dequantize on retrieval quality)
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

        if not np.all(np.isfinite(vecs)):
            log.warning("Non-finite values in ColBERT vectors for id %s — skipping", row["id"])
            continue

        if len(vecs) == 0:
            log.warning("Empty token list for id %s — skipping", row["id"])
            continue

        total_tokens += len(vecs)

        # Build postgres vector[] literal
        vec_parts = []
        for v in vecs:
            vec_str = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
            vec_parts.append(f"'{vec_str}'::vector(1024)")
        array_expr = "ARRAY[" + ",".join(vec_parts) + "]"

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
    log.info("Populated %d memories in %.0fs (avg %.0f tokens/mem)", total, elapsed, avg_tokens)
    return avg_tokens, elapsed


def rebuild_maxsim_index(backend) -> float:
    """Drop and recreate the VectorChord MaxSim index. Returns elapsed seconds."""
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
    return elapsed


def query_colbert_to_pg_array(query_vectors) -> str:
    """Convert query ColBERT vectors [n_tokens, 128] to postgres ARRAY literal."""
    if len(query_vectors) == 0:
        return "ARRAY[]::vector(1024)[]"
    parts = []
    for v in query_vectors:
        vec_str = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
        parts.append(f"'{vec_str}'::vector(1024)")
    return "ARRAY[" + ",".join(parts) + "]"


def evaluate_colbert_retrieval(
    beam_dir: Path,
    bucket: str,
    backend,
    cached_embeddings: dict,
    top_k: int = 10,
) -> dict:
    """Run BEAM evaluation using three-way RRF (dense + keyword + ColBERT MaxSim).

    Calls hybrid_search_memories_colbert() directly via SQL.
    """
    import math

    from beam_benchmark import (
        ALL_CATEGORIES,
        list_chats,
        load_probing_questions,
    )

    all_questions = []
    for chat_id in list_chats(beam_dir, bucket):
        questions = load_probing_questions(beam_dir, bucket, chat_id)
        for cat in ALL_CATEGORIES:
            for q in questions.get(cat, []):
                all_questions.append((chat_id, cat, q))

    total = len(all_questions)
    assert total == len(cached_embeddings["dense"]), (
        f"Question count mismatch: {total} questions vs "
        f"{len(cached_embeddings['dense'])} cached embeddings"
    )
    log.info("Evaluating %d questions via three-way RRF", total)

    all_metrics = []
    category_metrics: dict[str, list] = {}
    search_times = []
    t0 = time.monotonic()

    for i, (chat_id, cat, question) in enumerate(all_questions):
        profile = f"beam_{bucket}_{chat_id}"
        query_text = question["question"]

        # Get pre-computed embeddings
        dense_emb = cached_embeddings["dense"][i]
        colbert_vecs = cached_embeddings["colbert"][i]

        # Build the ColBERT query array expression
        colbert_array = query_colbert_to_pg_array(colbert_vecs)

        # Call three-way RRF search
        search_t0 = time.monotonic()
        results = backend._execute(
            f"""SELECT * FROM hybrid_search_memories_colbert(
                %(query_text)s,
                %(query_embedding)s::vector,
                {colbert_array},
                %(match_count)s,
                %(profile)s
            )""",
            {
                "query_text": query_text,
                "query_embedding": str(dense_emb),
                "match_count": max(top_k, 50),
                "profile": profile,
            },
            fetch="all",
        )
        search_ms = (time.monotonic() - search_t0) * 1000
        search_times.append(search_ms)

        # Score against gold — same logic as beam_benchmark.evaluate_question
        gold_msg_set = set()
        source_ids = question.get("source_chat_ids", {})
        if isinstance(source_ids, dict):
            for key, ids in source_ids.items():
                if isinstance(ids, list):
                    gold_msg_set.update(str(mid) for mid in ids)
        elif isinstance(source_ids, list):
            gold_msg_set.update(str(mid) for mid in source_ids)

        # Recall@K
        hit_at = {}
        for k in [5, 10, 20, 30, 50]:
            top_k_msgs = set()
            for r in results[:k]:
                meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
                top_k_msgs.update(str(m) for m in meta.get("msg_ids", []))
            if gold_msg_set:
                hit_at[k] = len(gold_msg_set & top_k_msgs) / len(gold_msg_set)
            else:
                hit_at[k] = 1.0 if cat in ("abstention", "summarization") else 0.0

        # MRR
        mrr = 0.0
        for j, r in enumerate(results):
            meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
            r_msgs = set(str(m) for m in meta.get("msg_ids", []))
            if r_msgs & gold_msg_set:
                mrr = 1.0 / (j + 1)
                break

        # NDCG@10
        dcg = 0.0
        for j, r in enumerate(results[:10]):
            meta = r.get("metadata", {}) if isinstance(r.get("metadata"), dict) else {}
            r_msgs = set(str(m) for m in meta.get("msg_ids", []))
            if r_msgs & gold_msg_set:
                dcg += 1.0 / math.log2(j + 2)
        idcg = sum(1.0 / math.log2(j + 2) for j in range(min(len(gold_msg_set), 10)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics = {
            "chat_id": chat_id,
            "category": cat,
            "question": query_text[:120],
            "recall@5": hit_at.get(5, 0.0),
            "recall@10": hit_at.get(10, 0.0),
            "recall@20": hit_at.get(20, 0.0),
            "recall@50": hit_at.get(50, 0.0),
            "ndcg@10": ndcg,
            "mrr": mrr,
            "search_ms": round(search_ms, 1),
        }
        all_metrics.append(metrics)
        category_metrics.setdefault(cat, []).append(metrics)

        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            log.info("  %d/%d (%.1f/s, avg search %.0fms)",
                     i + 1, total, (i + 1) / elapsed,
                     sum(search_times[-50:]) / min(50, len(search_times)))

    elapsed = time.monotonic() - t0

    # Aggregate
    avg = {
        "recall@5": sum(m["recall@5"] for m in all_metrics) / len(all_metrics),
        "recall@10": sum(m["recall@10"] for m in all_metrics) / len(all_metrics),
        "recall@20": sum(m["recall@20"] for m in all_metrics) / len(all_metrics),
        "recall@50": sum(m["recall@50"] for m in all_metrics) / len(all_metrics),
        "ndcg@10": sum(m["ndcg@10"] for m in all_metrics) / len(all_metrics),
        "mrr": sum(m["mrr"] for m in all_metrics) / len(all_metrics),
    }

    cat_avgs = {}
    for cat, mlist in sorted(category_metrics.items()):
        cat_avgs[cat] = {
            "recall@5": sum(m["recall@5"] for m in mlist) / len(mlist),
            "recall@10": sum(m["recall@10"] for m in mlist) / len(mlist),
            "ndcg@10": sum(m["ndcg@10"] for m in mlist) / len(mlist),
            "mrr": sum(m["mrr"] for m in mlist) / len(mlist),
            "count": len(mlist),
        }

    latency = {
        "avg_search_ms": round(sum(search_times) / len(search_times), 1),
        "p50_search_ms": round(sorted(search_times)[len(search_times) // 2], 1),
        "p95_search_ms": round(sorted(search_times)[int(len(search_times) * 0.95)], 1),
        "max_search_ms": round(max(search_times), 1),
    }

    return {
        "overall": avg,
        "per_category": cat_avgs,
        "latency": latency,
        "questions_evaluated": len(all_metrics),
        "elapsed_seconds": round(elapsed, 1),
        "per_question": all_metrics,
    }


def precompute_query_embeddings(beam_dir: Path, bucket: str) -> dict:
    """Pre-compute all query embeddings once for reuse across configs."""
    from beam_benchmark import (
        ALL_CATEGORIES,
        EMBEDDING_BATCH_SIZE,
        _with_retry,
        list_chats,
        load_probing_questions,
    )
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

    dense = _with_retry(generate_embeddings_batch, query_texts, batch_size=EMBEDDING_BATCH_SIZE)
    log.info("Dense query embeddings ready")

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


def run_two_way_baseline(beam_dir: Path, bucket: str, top_k: int,
                         cached_embeddings: dict) -> dict:
    """Run dense+keyword two-way baseline (no ColBERT) for comparison."""
    from beam_benchmark import evaluate_bucket
    return evaluate_bucket(beam_dir, bucket, search_mode="tsvector", top_k=top_k,
                          cached_embeddings=cached_embeddings)


def _save_results(output_path: Path, all_results: dict, args,
                   colbert_configs: list, total_start: float) -> None:
    """Save current results to disk (incremental after each config)."""
    total_elapsed = time.monotonic() - total_start
    summary = {
        "benchmark": "BEAM ColBERT retrieval matrix (three-way RRF via VectorChord)",
        "bucket": args.bucket,
        "top_k": args.top_k,
        "total_configs": len(colbert_configs) + (0 if args.skip_baseline else 1),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "results": all_results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    completed = sum(1 for v in all_results.values() if "error" not in v)
    log.info("Progress saved: %d/%d configs -> %s",
             completed, summary["total_configs"], output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run ColBERT-as-retrieval benchmark matrix (VectorChord MaxSim)")
    parser.add_argument("--beam-dir", type=Path, default=Path("/tmp/BEAM"))
    parser.add_argument("--bucket", default="100K")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--configs", type=str, default=None,
                        help="Specific configs as 'precision:pool_factor,...'")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip dense+keyword two-way baseline")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mem-limit-gb", type=int, default=30)
    args = parser.parse_args()

    # Memory cap
    mem_bytes = args.mem_limit_gb * 1024 * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
        log.info("Memory limit (RLIMIT_DATA) set to %d GB", args.mem_limit_gb)
    except (ValueError, OSError) as e:
        log.warning("Could not set memory limit: %s", e)

    # Load benchmark env
    env_file = Path(__file__).parent.parent / "benchmarks" / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    output_path = args.output or RESULTS_DIR / f"retrieval_matrix_{args.bucket}.json"

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

    # Ensure column and migration are applied
    backend._execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_tokens vector(1024)[]",
        fetch="none",
    )

    # Apply the three-way RRF function via psql (the migration contains plpgsql
    # with semicolons inside $$ blocks — can't split on ';' safely)
    migration_path = Path(__file__).parent.parent / "sql" / "migrations" / "018_colbert_retrieval.sql"
    if migration_path.exists():
        log.info("Applying migration 018_colbert_retrieval.sql via psql...")
        import subprocess
        db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost/ogham")
        result = subprocess.run(
            ["psql", "-v", "ON_ERROR_STOP=0", db_url, "-f", str(migration_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.warning("Migration had errors (may be safe to ignore on re-run):\n%s",
                        result.stderr[:500])
        else:
            log.info("Migration applied successfully")

    # Resume support
    all_results = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            all_results = existing.get("results", {})
            completed = [k for k, v in all_results.items() if "error" not in v]
            log.info("Resuming: %d completed configs from %s", len(completed), output_path)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Could not load existing results: %s", e)
            all_results = {}

    # Pre-compute query embeddings
    log.info("=" * 70)
    log.info("PRE-COMPUTING QUERY EMBEDDINGS")
    log.info("=" * 70)
    cached = precompute_query_embeddings(args.beam_dir, args.bucket)

    total_start = time.monotonic()

    # Baseline: dense+keyword two-way (no ColBERT)
    if not args.skip_baseline:
        if "baseline_two_way" in all_results and "error" not in all_results["baseline_two_way"]:
            log.info("SKIPPING BASELINE (already completed)")
        else:
            log.info("=" * 70)
            log.info("BASELINE: dense + keyword (two-way, no ColBERT)")
            log.info("=" * 70)
            try:
                results = run_two_way_baseline(args.beam_dir, args.bucket,
                                               args.top_k, cached)
                all_results["baseline_two_way"] = {
                    "config": {"search_mode": "tsvector", "legs": 2},
                    "overall": results.get("overall", {}),
                    "latency": results.get("latency", {}),
                    "per_category": results.get("per_category", {}),
                    "questions_evaluated": results.get("questions_evaluated", 0),
                    "elapsed_seconds": results.get("elapsed_seconds", 0),
                }
            except Exception as e:
                log.error("Baseline failed: %s", e, exc_info=True)
                all_results["baseline_two_way"] = {"error": str(e)}
            gc.collect()
            _save_results(output_path, all_results, args, colbert_configs, total_start)

    # ColBERT retrieval configs
    for i, (precision, pool_factor) in enumerate(colbert_configs):
        config_name = f"retrieval_{precision}_pool{pool_factor}"

        if config_name in all_results and "error" not in all_results[config_name]:
            log.info("SKIPPING CONFIG %d/%d: %s (already completed)",
                     i + 1, len(colbert_configs), config_name)
            continue

        log.info("=" * 70)
        log.info("CONFIG %d/%d: %s (three-way RRF)", i + 1, len(colbert_configs), config_name)
        log.info("=" * 70)

        try:
            # Step 1: Populate colbert_tokens with this config
            avg_tokens, populate_time = populate_colbert_tokens(
                backend, pool_factor, precision)

            # Step 2: Rebuild MaxSim index
            index_time = rebuild_maxsim_index(backend)

            # Step 3: Evaluate via three-way RRF
            results = evaluate_colbert_retrieval(
                args.beam_dir, args.bucket, backend, cached, args.top_k)

            all_results[config_name] = {
                "config": {
                    "precision": precision,
                    "pool_factor": pool_factor,
                    "search_mode": "colbert_retrieval",
                    "legs": 3,
                },
                "avg_tokens_per_memory": round(avg_tokens, 1),
                "populate_seconds": round(populate_time, 1),
                "index_build_seconds": round(index_time, 1),
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
        _save_results(output_path, all_results, args, colbert_configs, total_start)

    total_elapsed = time.monotonic() - total_start
    _save_results(output_path, all_results, args, colbert_configs, total_start)

    # Summary table
    print("\n" + "=" * 100)
    print("COLBERT RETRIEVAL MATRIX SUMMARY (three-way RRF)")
    print("=" * 100)
    print(f"{'Config':<30s} {'R@5':>8s} {'R@10':>8s} {'R@20':>8s} "
          f"{'nDCG@10':>8s} {'MRR':>8s} {'AvgTok':>8s} {'SearchMs':>8s}")
    print("-" * 100)

    for name, data in all_results.items():
        if "error" in data:
            print(f"{name:<30s} ERROR: {data['error'][:60]}")
            continue
        o = data.get("overall", {})
        avg_tok = data.get("avg_tokens_per_memory", 0)
        lat = data.get("latency", {})
        search_ms = lat.get("avg_search_ms", 0)
        print(
            f"{name:<30s} {o.get('recall@5', 0):>8.4f} {o.get('recall@10', 0):>8.4f} "
            f"{o.get('recall@20', 0):>8.4f} {o.get('ndcg@10', 0):>8.4f} "
            f"{o.get('mrr', 0):>8.4f} {avg_tok:>8.0f} {search_ms:>8.1f}"
        )

    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
