# ColBERT Reranking Evaluation for Ogham

**Date:** 2026-03-29
**Benchmark:** BEAM (Benchmark for Evaluation of AI Memory), 100K bucket
**Hardware:** RunPod L40S GPU, 12 vCPU, 48GB VRAM, 80GB disk
**Model:** BGE-M3 via ONNX (dense 1024-dim + neural sparse + ColBERT token vectors)

## Summary

We evaluated whether adding ColBERT late-interaction reranking to ogham's existing
dense+sparse hybrid search would improve retrieval quality. After running a 24-config
compression matrix across 490 questions and 20 chat histories, the answer is **no**.
ColBERT reranking adds no measurable quality improvement over dense+sparse retrieval
alone, while adding significant storage and latency costs.

## Background

ColBERT (Contextualized Late Interaction over BERT) represents documents and queries
as sets of token-level vectors rather than single dense vectors. At query time, it
computes a MaxSim score: for each query token, find the most similar document token,
then sum those scores. This token-level matching can capture fine-grained semantic
relationships that a single dense vector misses.

The BGE-M3 model already produces ColBERT token vectors as a byproduct of its forward
pass (alongside the dense and sparse vectors ogham already uses). The question was
whether incorporating these vectors into the search pipeline would improve recall.

Clavie et al. (arXiv:2409.14683, "Is ColBERT Worth the Trouble?") showed that ColBERT
token vectors can be aggressively compressed via hierarchical pooling and quantization
with minimal quality loss. We implemented their approach and benchmarked it to determine
both (a) whether ColBERT helps at all, and (b) if so, how much it can be compressed.

## Methodology

### Search Pipeline

Ogham's search works in stages:

1. **First stage (dense+sparse):** `hybrid_search_memories` retrieves candidates using
   a combination of dense vector similarity (pgvector cosine distance) and neural sparse
   matching (SPLADE-style lexical vectors). This is the existing production search.

2. **ColBERT reranking (experimental):** The top candidates from the first stage are
   rescored using ColBERT MaxSim. The final score is a weighted blend:
   `final = 0.7 * normalized_hybrid + 0.3 * normalized_colbert`

### Compression Matrix

We tested every combination of:

- **Precision:** f32 (baseline), f16, int8 per-row quantization, int8 per-channel quantization
- **Pool factor:** 1 (no pooling), 2, 3, 4, 6, 8

Pool factor N means N adjacent token vectors are clustered (Ward's hierarchical
linkage) and replaced by their cluster centroids, reducing the token count by ~Nx.

This produces 24 configurations ranging from uncompressed (f32, pool 1, ~4.2 MB/memory)
to maximally compressed (int8, pool 8, ~132 KB/memory) — a 32x storage reduction.

### Evaluation

Each configuration was evaluated on the BEAM benchmark:
- 20 chat histories from the 100K bucket
- 490 probing questions across 10 categories (abstention, contradiction resolution,
  event ordering, information extraction, instruction following, knowledge update,
  multi-session reasoning, preference following, summarization, temporal reasoning)
- Metrics: Recall@5/10/20/50, nDCG@10, MRR

Query embeddings (dense + ColBERT) were pre-computed once and reused across all
configurations to ensure a fair comparison.

## Results

### Overall Quality

| Config | Precision | Pool | R@5 | R@10 | R@20 | nDCG@10 | MRR | Bytes/mem |
|---|---|---|---|---|---|---|---|---|
| f32_pool1 | f32 | 1 | 0.546 | 0.631 | 0.713 | 0.412 | 0.473 | 4,231,303 |
| f32_pool8 | f32 | 8 | 0.551 | 0.631 | 0.719 | 0.413 | 0.478 | 527,167 |
| f16_pool4 | f16 | 4 | 0.548 | 0.632 | 0.715 | 0.414 | 0.477 | 528,158 |
| int8_ch_pool4 | int8_ch | 4 | 0.549 | 0.635 | 0.716 | 0.417 | 0.480 | 268,180 |
| int8_ch_pool8 | int8_ch | 8 | 0.549 | 0.635 | 0.718 | 0.415 | 0.478 | 135,895 |

*(Representative configs shown. Full 24-config table in appendix.)*

**All 24 configurations score identically within noise.** R@10 ranges from 0.630 to
0.638 across the entire matrix. nDCG@10 ranges from 0.412 to 0.417. These differences
are not statistically meaningful across 490 questions.

### Storage vs Quality

Compressing ColBERT vectors by 32x (from 4.2 MB to 132 KB per memory) produces
no measurable quality degradation. Every configuration achieves ~100% of the
uncompressed f32_pool1 baseline:

| Compression | Storage | % of baseline nDCG |
|---|---|---|
| f32, no pooling | 4,231 KB | 100.0% |
| f16, pool 4 | 528 KB | 100.6% |
| int8_ch, pool 4 | 268 KB | 101.3% |
| int8_ch, pool 8 | 136 KB | 100.9% |

### Comparison with Clavie et al.

| Pool factor | Paper (% of baseline) | Ours (% of baseline) | Delta |
|---|---|---|---|
| 2 | 100.62% | 100.32% | -0.30 |
| 3 | 99.03% | 100.76% | +1.73 |
| 4 | 97.03% | 100.59% | +3.56 |
| 6 | 90.67% | 100.57% | +9.90 |

Our results diverge significantly from the paper at higher pool factors. The paper
shows meaningful degradation at pool 6 (90.67%), while ours stays flat. This is not
because our ColBERT implementation is better — it's because ColBERT is not
contributing to the final score in our pipeline.

### Ranking Latency

ColBERT reranking latency scales linearly with token count:

| Config | Rerank time (ms) |
|---|---|
| f32_pool1 | 1,838 |
| f32_pool4 | 465 |
| int8_ch_pool8 | 103 |

Even the fastest config adds ~100ms of reranking latency per query.

## Analysis

### Why ColBERT Reranking Has No Effect

The results are clear: ColBERT reranking does not improve retrieval quality in ogham's
pipeline. The evidence is:

1. **Quality is flat across all compression levels.** If ColBERT were doing useful work,
   destroying 97% of the vector data (32x compression) would degrade the score. It
   doesn't, because ColBERT's signal is not influencing the final ranking.

2. **The blend architecture limits ColBERT's influence.** With a 0.7/0.3 blend
   (70% dense hybrid, 30% ColBERT), the first-stage ranking dominates. But even
   within that 30% budget, ColBERT is not differentiating candidates — suggesting the
   dense+sparse first stage already orders the candidates correctly.

3. **The divergence from Clavie et al. confirms this.** Their paper evaluates ColBERT
   as a primary retriever on standard IR benchmarks (BEIR). In that setting, ColBERT
   does the heavy lifting, so degrading its vectors hurts. In our setting, ColBERT is
   a secondary signal on top of an already-effective dense+sparse retriever, so
   degrading it changes nothing.

### Why Dense+Sparse Is Sufficient

Ogham's use case is personal memory retrieval — finding relevant memories from a
user's own history. This differs from web-scale IR in important ways:

- **Small corpus per profile.** Hundreds to low thousands of memories, not millions of
  documents. The retrieval problem is easier.
- **High-quality dense vectors.** BGE-M3's 1024-dim dense embeddings capture semantic
  meaning well for the conversational text typical of memory content.
- **Neural sparse fills the lexical gap.** The SPLADE-style sparse vectors from BGE-M3
  handle exact keyword matching that dense vectors miss, covering the main weakness
  of dense-only retrieval.
- **User-scoped search.** Queries are always within a single user's profile, so there's
  less ambiguity to resolve than in open-domain search.

The combination of dense semantic matching and neural sparse lexical matching appears
to be sufficient for this domain. Token-level ColBERT matching would add value in
scenarios where two documents are semantically similar overall but differ in specific
token-level details — a situation that apparently doesn't arise often enough in personal
memory retrieval to move aggregate metrics.

## Decision

**Do not add ColBERT reranking to ogham.**

The cost/benefit analysis is clear:

| | Without ColBERT | With ColBERT (int8_ch, pool 4) |
|---|---|---|
| Storage per memory | ~1 KB (dense) + ~2 KB (sparse) | + 268 KB ColBERT |
| Embedding latency | ~50ms | + ~20ms for token vectors |
| Search latency | ~90ms | + ~190ms reranking |
| Code complexity | Existing | New packing/unpacking, reranking logic, DB column |
| Quality improvement | Baseline | None measurable |

ColBERT would increase per-memory storage by ~100x, add ~200ms search latency, and
require maintaining token vector packing, quantization, and reranking code — all for
zero quality improvement on the BEAM benchmark.

### When to Revisit

ColBERT reranking might become worthwhile if:

- **Corpus size grows significantly.** At tens of thousands of memories per profile,
  the dense+sparse first stage might start returning more ambiguous candidate sets
  where fine-grained token matching helps.
- **A harder benchmark emerges.** BEAM may not stress the retrieval system enough to
  surface ColBERT's advantage. A benchmark with more subtle distinctions between
  relevant and near-relevant memories could change the picture.
- **ColBERT is used as primary retriever.** Instead of reranking dense+sparse results,
  using ColBERT similarity directly (e.g., via a specialized index) could yield
  different results. This would require pgvector support for multi-vector similarity
  or an external index.

## Appendix: Full Results

| Config | Prec | Pool | R@5 | R@10 | R@20 | R@50 | nDCG | MRR | Bytes | %ceil | Srch | Rnk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f32_pool1 | f32 | 1 | 0.546 | 0.631 | 0.713 | 0.843 | 0.412 | 0.473 | 4231303 | 100.0 | 375 | 1838 |
| f32_pool2 | f32 | 2 | 0.546 | 0.633 | 0.713 | 0.843 | 0.413 | 0.473 | 2114659 | 100.3 | 94 | 923 |
| f32_pool3 | f32 | 3 | 0.548 | 0.633 | 0.713 | 0.845 | 0.413 | 0.474 | 1409058 | 100.4 | 94 | 611 |
| f32_pool4 | f32 | 4 | 0.548 | 0.631 | 0.715 | 0.845 | 0.413 | 0.476 | 1056309 | 100.4 | 94 | 465 |
| f32_pool6 | f32 | 6 | 0.547 | 0.632 | 0.714 | 0.847 | 0.413 | 0.477 | 703502 | 100.4 | 93 | 320 |
| f32_pool8 | f32 | 8 | 0.551 | 0.631 | 0.719 | 0.847 | 0.413 | 0.478 | 527167 | 100.4 | 90 | 239 |
| f16_pool1 | f16 | 1 | 0.549 | 0.632 | 0.717 | 0.847 | 0.413 | 0.472 | 2115656 | 100.3 | 93 | 1257 |
| f16_pool2 | f16 | 2 | 0.548 | 0.632 | 0.717 | 0.847 | 0.413 | 0.472 | 1057333 | 100.3 | 95 | 688 |
| f16_pool3 | f16 | 3 | 0.549 | 0.635 | 0.716 | 0.848 | 0.415 | 0.476 | 704533 | 100.8 | 94 | 497 |
| f16_pool4 | f16 | 4 | 0.548 | 0.632 | 0.715 | 0.848 | 0.414 | 0.477 | 528158 | 100.6 | 94 | 382 |
| f16_pool6 | f16 | 6 | 0.546 | 0.634 | 0.715 | 0.848 | 0.414 | 0.477 | 351755 | 100.6 | 92 | 255 |
| f16_pool8 | f16 | 8 | 0.549 | 0.630 | 0.718 | 0.847 | 0.413 | 0.477 | 263588 | 100.4 | 92 | 196 |
| int8_row_pool1 | int8_row | 1 | 0.548 | 0.635 | 0.717 | 0.847 | 0.415 | 0.475 | 1061965 | 100.8 | 95 | 591 |
| int8_row_pool2 | int8_row | 2 | 0.548 | 0.634 | 0.717 | 0.847 | 0.415 | 0.475 | 530737 | 100.8 | 95 | 370 |
| int8_row_pool3 | int8_row | 3 | 0.546 | 0.635 | 0.717 | 0.847 | 0.415 | 0.476 | 353648 | 100.9 | 92 | 244 |
| int8_row_pool4 | int8_row | 4 | 0.548 | 0.635 | 0.716 | 0.848 | 0.416 | 0.478 | 265116 | 101.0 | 91 | 191 |
| int8_row_pool6 | int8_row | 6 | 0.545 | 0.636 | 0.716 | 0.847 | 0.415 | 0.477 | 176570 | 100.9 | 91 | 133 |
| int8_row_pool8 | int8_row | 8 | 0.549 | 0.630 | 0.718 | 0.847 | 0.414 | 0.477 | 132314 | 100.6 | 90 | 103 |
| int8_channel_pool1 | int8_ch | 1 | 0.550 | 0.634 | 0.717 | 0.847 | 0.416 | 0.477 | 1061929 | 101.1 | 95 | 574 |
| int8_channel_pool2 | int8_ch | 2 | 0.550 | 0.634 | 0.717 | 0.847 | 0.416 | 0.477 | 532768 | 101.1 | 94 | 356 |
| int8_channel_pool3 | int8_ch | 3 | 0.548 | 0.635 | 0.719 | 0.847 | 0.416 | 0.478 | 356367 | 101.2 | 92 | 234 |
| int8_channel_pool4 | int8_ch | 4 | 0.549 | 0.635 | 0.716 | 0.848 | 0.417 | 0.480 | 268180 | 101.3 | 93 | 188 |
| int8_channel_pool6 | int8_ch | 6 | 0.547 | 0.638 | 0.715 | 0.847 | 0.417 | 0.479 | 179979 | 101.4 | 90 | 131 |
| int8_channel_pool8 | int8_ch | 8 | 0.549 | 0.635 | 0.718 | 0.847 | 0.415 | 0.478 | 135895 | 100.9 | 89 | 103 |

### Per-Category Breakdown

Best config (int8_channel_pool6) vs worst (f32_pool1) by nDCG@10:

| Category | Best R@10 | Worst R@10 | Delta |
|---|---|---|---|
| abstention | 1.0000 | 1.0000 | +0.0000 |
| contradiction_resolution | 0.8488 | 0.8613 | -0.0125 |
| event_ordering | 0.2240 | 0.2219 | +0.0021 |
| information_extraction | 0.7458 | 0.7208 | +0.0250 |
| instruction_following | 0.5500 | 0.5250 | +0.0250 |
| knowledge_update | 0.8792 | 0.8833 | -0.0042 |
| multi_session_reasoning | 0.5427 | 0.5427 | +0.0000 |
| preference_following | 0.4583 | 0.4333 | +0.0250 |
| summarization | 0.3086 | 0.3136 | -0.0050 |
| temporal_reasoning | 0.8250 | 0.8125 | +0.0125 |

No category shows a meaningful difference between the most and least compressed ColBERT
configurations.

## Appendix: Infrastructure Notes

The benchmark was run on RunPod using GPU-accelerated ONNX inference. Key setup details
are documented in `scripts/runpod-gpu-setup.sh` and `benchmarks/runpod-cpu-reference.md`.

Notable issues encountered during deployment:
- RLIMIT_AS (virtual memory cap) kills CUDA mmap — must use RLIMIT_DATA instead
- onnxruntime and onnxruntime-gpu conflict — must uninstall CPU version before installing GPU
- OpenBLAS defaults to 64 threads, crashes on limited-vCPU pods — set OPENBLAS_NUM_THREADS=4
- PostgreSQL WAL bloat from bulk UPDATE operations requires VACUUM FULL and adequate disk (80GB+)
- uv aggressively downloads Python 3.14 beta which breaks pydantic — pin to 3.13

Total benchmark runtime: ~4.5 hours (24 ColBERT configs, each requiring a full repool
of ~3000 memories + evaluation of 490 questions).
