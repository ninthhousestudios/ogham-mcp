# ColBERT Primary Retrieval Evaluation for Ogham

**Date:** 2026-04-01
**Benchmark:** BEAM (Benchmark for Evaluation of AI Memory), 100K bucket
**Hardware:** RunPod L40S GPU, 12 vCPU, 48GB VRAM, 50GB disk
**Model:** BGE-M3 via ONNX (dense 1024-dim + tsvector keyword + ColBERT 1024-dim token vectors)

## Summary

We evaluated whether adding ColBERT as a **primary retrieval leg** (not just a reranker)
via three-way Reciprocal Rank Fusion (dense HNSW + keyword tsvector + ColBERT MaxSim)
would improve retrieval quality over the existing two-way fusion (dense + keyword).
The answer is **no** — ColBERT retrieval significantly **degrades** performance, dropping
R@10 by 23% and nDCG@10 by 33%.

This follows the [reranking evaluation](colbert-evaluation.md) which found ColBERT
reranking adds no value either. ColBERT is not useful for ogham's retrieval pipeline
in any configuration tested.

## Background

The previous evaluation tested ColBERT as a **reranker** — rescoring candidates already
retrieved by dense+sparse search. It found zero improvement across all 24 compression
configs, but the architecture limited ColBERT's influence (0.7/0.3 blend weight).

This evaluation tests a stronger hypothesis: ColBERT as an **independent retrieval leg**
using VectorChord's native MaxSim index. Instead of reranking existing results, ColBERT
retrieves its own candidate set, which is fused with dense and keyword results via RRF.
This gives ColBERT equal weight in the final ranking.

### Key difference from reranking evaluation

- **Reranking:** dense+sparse retrieves candidates, ColBERT rescores them (secondary signal)
- **Retrieval:** ColBERT retrieves its own candidates via MaxSim index (primary signal, equal footing)

### ONNX model gap: missing colbert_linear projection

During setup we discovered that the ONNX export of BGE-M3 (`yuniko-software/bge-m3-onnx`)
omits the `colbert_linear` projection layer. The third output (`colbert_vectors`) returns
raw 1024-dim hidden states instead of properly projected ColBERT vectors.

We fixed this by downloading `colbert_linear.pt` (~2MB) from the BAAI/bge-m3 HuggingFace
repo and applying the projection in Python: `projected = hidden_states @ W.T + b`, followed
by L2 normalization. The projection is [1024, 1024] — BGE-M3's ColBERT dimension is 1024,
not 128 as commonly assumed from ColBERTv2.

This means all ColBERT vectors from previous benchmarks (the reranking evaluation) were
computed **without** the colbert_linear projection. Those results measured MaxSim on raw
hidden states, not proper ColBERT vectors. The reranking evaluation's conclusion (no
improvement) still holds — if anything, proper projection should have helped, and it didn't.

## Methodology

### Search Pipeline

Three-way RRF via `hybrid_search_memories_colbert()`:

1. **Dense leg:** HNSW cosine similarity on 1024-dim halfvec embeddings → top 30 candidates
2. **Keyword leg:** `ts_rank_cd` on tsvector full-text search → top 30 candidates
3. **ColBERT leg:** VectorChord MaxSim (`@#` operator) on `vector(1024)[]` → top 30 candidates

Results are fused via Reciprocal Rank Fusion (k=60):

```
rrf_score = 1/(k + dense_rank) + 1/(k + keyword_rank) + 1/(k + colbert_rank)
```

### Configuration tested

Given the reranking evaluation already showed ColBERT quality is flat across all 24
compression configs, we tested a single practical configuration:

- **f32, pool factor 4:** Full-precision vectors with 4x token pooling. This reduces
  token count by 4x via hierarchical clustering while preserving vector quality.
  Average 258 tokens/memory after pooling.

If the best practical config can't beat the baseline, more compressed configs won't either.

### Infrastructure: VectorChord

VectorChord 1.1.1 was installed as a postgres extension, providing:

- The `@#` MaxSim operator for ColBERT late-interaction scoring
- The `vchordrq` index type (IVF + RaBitQ) for approximate MaxSim search
- Native `vector(1024)[]` storage for token vector arrays

### Storage impact

The `colbert_tokens` column with 1024-dim unpooled vectors (f32, pool 1) inflated the
memories table from 12GB to 43GB — a 3.6x increase. This bloat slowed **all** queries,
including the baseline which doesn't use ColBERT. Pool factor 4 reduces this substantially
but the storage cost remains significant.

### Evaluation

- 20 chat histories from the BEAM 100K bucket
- 400 probing questions across 10 categories
- Metrics: Recall@5/10/20/50, nDCG@10, MRR
- Query embeddings (dense + ColBERT) pre-computed once and reused

## Results

### Overall

| Config | Legs | R@5 | R@10 | R@20 | R@50 | nDCG@10 | MRR | Avg Search (ms) |
|---|---|---|---|---|---|---|---|---|
| **baseline (dense+keyword)** | 2 | **0.520** | **0.597** | **0.681** | **0.763** | **0.385** | **0.442** | 61 |
| ColBERT f32 pool4 | 3 | 0.390 | 0.460 | 0.575 | 0.723 | 0.259 | 0.303 | 242 |

ColBERT retrieval is worse on every metric:

| Metric | Baseline | ColBERT | Delta | Change |
|---|---|---|---|---|
| R@5 | 0.520 | 0.390 | -0.130 | **-25%** |
| R@10 | 0.597 | 0.460 | -0.137 | **-23%** |
| R@20 | 0.681 | 0.575 | -0.106 | **-16%** |
| R@50 | 0.763 | 0.723 | -0.040 | -5% |
| nDCG@10 | 0.385 | 0.259 | -0.126 | **-33%** |
| MRR | 0.442 | 0.303 | -0.139 | **-31%** |

### Per-Category Breakdown

| Category | Baseline R@10 | ColBERT R@10 | Delta | Baseline nDCG | ColBERT nDCG | Delta |
|---|---|---|---|---|---|---|
| abstention | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| contradiction_resolution | 0.806 | 0.597 | **-0.209** | 0.687 | 0.445 | **-0.242** |
| event_ordering | 0.153 | 0.108 | -0.045 | 0.137 | 0.092 | -0.045 |
| information_extraction | 0.621 | 0.433 | **-0.188** | 0.429 | 0.316 | **-0.113** |
| instruction_following | 0.525 | 0.363 | **-0.163** | 0.315 | 0.203 | **-0.113** |
| knowledge_update | 0.842 | 0.621 | **-0.221** | 0.732 | 0.530 | **-0.202** |
| multi_session_reasoning | 0.469 | 0.350 | **-0.120** | 0.360 | 0.250 | **-0.110** |
| preference_following | 0.421 | 0.325 | **-0.096** | 0.292 | 0.223 | **-0.069** |
| summarization | 0.304 | 0.265 | -0.039 | 0.174 | 0.138 | -0.036 |
| temporal_reasoning | 0.825 | 0.538 | **-0.288** | 0.724 | 0.396 | **-0.328** |

ColBERT hurts every category except abstention (which is already at ceiling). The worst
damage is to temporal_reasoning (-0.288 R@10, -0.328 nDCG) and knowledge_update (-0.221 R@10).

### Latency

| Config | Avg (ms) | P50 (ms) | P95 (ms) |
|---|---|---|---|
| Baseline | 61 | 61 | 123 |
| ColBERT f32 pool4 | 242 | 230 | 375 |

ColBERT adds ~180ms average search latency (4x slower than baseline).

## Analysis

### Why ColBERT retrieval hurts

ColBERT as a retrieval leg actively **degrades** results. This is worse than the reranking
evaluation (which showed no effect). The mechanism:

1. **RRF gives ColBERT equal voting weight.** In three-way RRF, each leg contributes
   equally. If ColBERT's candidate set is poor, it dilutes the good rankings from
   dense and keyword search.

2. **ColBERT retrieves different (wrong) candidates.** MaxSim scoring finds documents
   that match well at the token level but aren't actually relevant. These irrelevant
   documents get RRF votes and push down the correct results.

3. **The damage is proportional to ColBERT's independence.** Categories where dense+keyword
   already works well (knowledge_update, temporal_reasoning) suffer the most, because
   ColBERT's independent candidate set displaces results that both other legs agree on.

### Comparison with reranking evaluation

| Approach | Effect on R@10 | Effect on nDCG@10 |
|---|---|---|
| ColBERT reranking (0.7/0.3 blend) | +0.000 (no effect) | +0.000 (no effect) |
| ColBERT retrieval (three-way RRF) | **-0.137 (-23%)** | **-0.126 (-33%)** |

Giving ColBERT more influence makes things worse, not better. The reranking evaluation
showed ColBERT adds no signal; the retrieval evaluation confirms it adds **noise**.

### Note on the colbert_linear projection

The reranking evaluation computed ColBERT vectors without the colbert_linear projection
(raw hidden states). This retrieval evaluation uses properly projected vectors. The
projected vectors perform worse as a retrieval signal than the unprojected ones did as
a reranking signal — though this comparison isn't apples-to-apples since the search
architecture differs.

## Decision

**Do not add ColBERT to ogham's retrieval pipeline in any form.**

Two evaluations, two architectures, same conclusion:

| Evaluation | Architecture | Result |
|---|---|---|
| Reranking (2026-03-29) | Rerank dense+sparse candidates with ColBERT | No improvement |
| Retrieval (2026-04-01) | Three-way RRF with ColBERT as independent leg | 23% degradation |

The dense+keyword two-way fusion is ogham's best retrieval strategy. Adding ColBERT
in any configuration — as a reranker or as a retrieval leg — does not help and may
actively harm quality.

### When to revisit

The conditions from the reranking evaluation still apply, but with a stronger prior
against ColBERT:

- A fundamentally different embedding model with better ColBERT vectors
- A corpus where dense+keyword genuinely struggles (not the case for personal memory)
- Evidence from other personal-memory systems showing ColBERT benefit

### What to keep

The infrastructure work has value beyond ColBERT:

- **VectorChord integration** and `runpod-gpu-setup.sh` fixes (tag naming, .deb naming,
  shared_preload_libraries) are useful for future postgres extension experiments
- **colbert_linear projection fix** in `onnx_embedder.py` corrects a real bug — the ONNX
  model was returning unprojected hidden states labeled as ColBERT vectors
- **BEAM benchmark tooling** (resume support, incremental saves) is reusable

## Appendix: Infrastructure Notes

### VectorChord setup issues resolved

- Tag naming: VectorChord uses bare semver tags (`1.1.1`), not `v`-prefixed (`v1.1.1`)
- Deb naming: changed from `vchord-pg{VER}_{VERSION}_amd64.deb` to
  `postgresql-{VER}-vchord_{VERSION}-1_amd64.deb` (Debian convention)
- VectorChord requires `shared_preload_libraries = 'vchord'` in postgresql.conf
- PL/pgSQL ambiguous column reference: `SELECT id FROM cte` conflicts with function
  return column `id` — must qualify as `SELECT cte.id FROM cte`

### Storage bloat

1024-dim ColBERT vectors at pool factor 1 inflated the memories table from 12GB to 43GB,
slowing all queries including non-ColBERT baselines. Pool factor 4 is more manageable
but still adds significant overhead. This storage cost alone makes ColBERT impractical
for ogham's target deployment (14GB laptop with limited disk).

Total benchmark runtime: ~25 minutes (1 baseline + 1 ColBERT config, including
14 minutes for token population and index build).
