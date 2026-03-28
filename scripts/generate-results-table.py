#!/usr/bin/env python3
"""Generate comparison tables from benchmark matrix results.

Reads the consolidated JSON from run-benchmark-matrix.py and produces
markdown tables matching the style of Clavié et al. (arXiv:2409.14683).

Usage:
    uv run python scripts/generate-results-table.py
    uv run python scripts/generate-results-table.py --input benchmarks/beam_results/matrix_100K.json
    uv run python scripts/generate-results-table.py --csv  # also output CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def format_table(data: dict, output_csv: Path | None = None):
    """Print markdown table and optionally write CSV."""
    results = data["results"]

    # Find the quality ceiling (f32_pool1) for relative performance
    ceiling = results.get("f32_pool1", {}).get("overall", {})
    ceiling_ndcg = ceiling.get("ndcg@10", 1)

    # Collect rows
    rows = []
    for name, entry in results.items():
        if "error" in entry:
            continue
        o = entry.get("overall", {})
        config = entry.get("config", {})
        lat = entry.get("latency", {})

        precision = config.get("precision", "-")
        pool_factor = config.get("pool_factor", "-")
        storage = entry.get("storage_bytes_avg", 0)

        ndcg = o.get("ndcg@10", 0)
        rel_pct = (ndcg / ceiling_ndcg * 100) if ceiling_ndcg > 0 else 0

        rows.append({
            "name": name,
            "precision": str(precision),
            "pool": str(pool_factor),
            "recall@5": o.get("recall@5", 0),
            "recall@10": o.get("recall@10", 0),
            "recall@20": o.get("recall@20", 0),
            "recall@50": o.get("recall@50", 0),
            "ndcg@10": ndcg,
            "mrr": o.get("mrr", 0),
            "bytes": storage,
            "rel%": rel_pct,
            "search_ms": lat.get("search_ms_avg", 0),
            "rerank_ms": lat.get("rerank_ms_avg", 0),
        })

    # Sort: baselines first, then by precision and pool factor
    precision_order = {"-": 0, "f32": 1, "f16": 2, "int8_row": 3, "int8_channel": 4}
    rows.sort(key=lambda r: (precision_order.get(r["precision"], 99), int(r["pool"]) if r["pool"] != "-" else 0))

    # Markdown table
    print("\n## Overall Results")
    print()
    print(f"| {'Config':<20s} | {'Prec':>5s} | {'Pool':>4s} | {'R@5':>6s} | {'R@10':>6s} | "
          f"{'R@20':>6s} | {'R@50':>6s} | {'nDCG':>6s} | {'MRR':>6s} | "
          f"{'Bytes':>7s} | {'%ceil':>5s} | {'Srch':>5s} | {'Rnk':>5s} |")
    print(f"|{'-'*22}|{'-'*7}|{'-'*6}|{'-'*8}|{'-'*8}|"
          f"{'-'*8}|{'-'*8}|{'-'*8}|{'-'*8}|"
          f"{'-'*9}|{'-'*7}|{'-'*7}|{'-'*7}|")

    for r in rows:
        print(
            f"| {r['name']:<20s} | {r['precision']:>5s} | {r['pool']:>4s} | "
            f"{r['recall@5']:>6.3f} | {r['recall@10']:>6.3f} | "
            f"{r['recall@20']:>6.3f} | {r['recall@50']:>6.3f} | "
            f"{r['ndcg@10']:>6.3f} | {r['mrr']:>6.3f} | "
            f"{r['bytes']:>7.0f} | {r['rel%']:>5.1f} | "
            f"{r['search_ms']:>5.0f} | {r['rerank_ms']:>5.0f} |"
        )

    # Storage comparison
    print("\n## Storage Comparison")
    print()
    print(f"| {'Config':<20s} | {'Bytes/mem':>10s} | {'vs f32_pool1':>12s} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*14}|")

    ceiling_bytes = next((r["bytes"] for r in rows if r["name"] == "f32_pool1"), 1)
    for r in rows:
        if r["bytes"] > 0:
            ratio = r["bytes"] / ceiling_bytes * 100 if ceiling_bytes > 0 else 0
            print(f"| {r['name']:<20s} | {r['bytes']:>10.0f} | {ratio:>11.1f}% |")

    # Paper comparison (hierarchical pooling, f16)
    paper_data = {2: 100.62, 3: 99.03, 4: 97.03, 6: 90.67}
    our_f16 = {r["pool"]: r["rel%"] for r in rows if r["precision"] == "f16"}

    if our_f16:
        print("\n## Comparison with Clavié et al.")
        print()
        print(f"| {'Pool factor':>11s} | {'Paper (%)':>9s} | {'Ours (%)':>9s} | {'Delta':>6s} |")
        print(f"|{'-'*13}|{'-'*11}|{'-'*11}|{'-'*8}|")
        for pf in ["1", "2", "3", "4", "6", "8"]:
            paper_val = paper_data.get(int(pf), "-")
            our_val = our_f16.get(pf, None)
            if our_val is not None:
                paper_str = f"{paper_val:.2f}" if isinstance(paper_val, float) else str(paper_val)
                delta = f"{our_val - paper_val:+.2f}" if isinstance(paper_val, float) else "-"
                print(f"| {pf:>11s} | {paper_str:>9s} | {our_val:>9.2f} | {delta:>6s} |")

    # Per-category for best and worst configs
    print("\n## Per-Category Breakdown (best vs worst ColBERT configs)")
    print()

    colbert_rows = [r for r in rows if r["precision"] not in ("-",)]
    if len(colbert_rows) >= 2:
        best = max(colbert_rows, key=lambda r: r["ndcg@10"])
        worst = min(colbert_rows, key=lambda r: r["ndcg@10"])

        best_cats = results.get(best["name"], {}).get("per_category", {})
        worst_cats = results.get(worst["name"], {}).get("per_category", {})

        if best_cats and worst_cats:
            cats = sorted(set(best_cats) | set(worst_cats))
            print(f"Best: {best['name']}  |  Worst: {worst['name']}")
            print()
            print(f"| {'Category':<30s} | {'Best R@10':>9s} | {'Worst R@10':>10s} | {'Delta':>6s} |")
            print(f"|{'-'*32}|{'-'*11}|{'-'*12}|{'-'*8}|")
            for cat in cats:
                b = best_cats.get(cat, {}).get("recall@10", 0)
                w = worst_cats.get(cat, {}).get("recall@10", 0)
                print(f"| {cat:<30s} | {b:>9.4f} | {w:>10.4f} | {b-w:>+6.4f} |")

    # CSV output
    if output_csv and rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison tables")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent.parent / "benchmarks" / "beam_results" / "matrix_100K.json",
        help="Input matrix JSON",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Also output CSV to this path")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    data = load_results(args.input)
    format_table(data, output_csv=args.csv)


if __name__ == "__main__":
    main()
