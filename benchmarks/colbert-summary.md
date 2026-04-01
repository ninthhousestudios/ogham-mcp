# ColBERT Evaluation Summary for Ogham

**TL;DR:** We tested ColBERT token-level matching in two configurations. It doesn't
help and can actively hurt retrieval quality. Ogham's existing dense+keyword search
is the right approach.

## What we tested

BGE-M3 produces three types of embeddings in a single forward pass: dense vectors
(1024-dim), sparse lexical vectors, and ColBERT token vectors (1024-dim per token).
Ogham currently uses dense + keyword (tsvector) search. We investigated whether
adding ColBERT would improve retrieval.

We ran two experiments on the BEAM benchmark (400 questions, 20 chat histories,
~2900 memories) using a RunPod L40S GPU:

**Test 1 — ColBERT as reranker (March 29):**
Retrieve candidates with dense+sparse, then rescore using ColBERT MaxSim.
24 configs tested (4 precisions x 6 compression levels).

**Test 2 — ColBERT as retrieval leg (April 1):**
Three-way Reciprocal Rank Fusion — dense, keyword, and ColBERT each independently
retrieve candidates, results are fused. Uses VectorChord's native MaxSim index.

## Results

| Approach | R@10 | nDCG@10 | MRR | Latency |
|---|---|---|---|---|
| **Baseline (dense + keyword)** | **0.597** | **0.385** | **0.442** | **61ms** |
| + ColBERT reranking | 0.597 | 0.385 | 0.442 | +190ms |
| + ColBERT retrieval | 0.460 | 0.259 | 0.303 | +180ms |

- **Reranking:** zero effect. All 24 compression configs score identically to baseline.
  Compressing ColBERT vectors by 32x doesn't change anything — because ColBERT isn't
  contributing signal in the first place.
- **Retrieval:** 23% worse R@10, 33% worse nDCG. ColBERT's independent candidate set
  is poor enough that it dilutes the good results from dense+keyword when fused via RRF.

## Why ColBERT doesn't help here

ColBERT excels at distinguishing between semantically similar documents that differ
in specific token-level details. In web-scale IR with millions of documents, this
matters. In ogham's setting it doesn't, because:

- **Small corpus** — hundreds to low thousands of memories per user, not millions
- **Dense+keyword already sufficient** — BGE-M3's 1024-dim dense vectors handle
  semantic matching well, keyword search covers lexical gaps
- **User-scoped search** — queries are within one person's memory, reducing ambiguity

The token-level matching signal ColBERT provides is redundant with what dense+keyword
already captures at this scale.

## Cost of adding ColBERT

Even if it helped, the costs are steep:

| | Without ColBERT | With ColBERT |
|---|---|---|
| Storage per memory | ~3 KB | + 260 KB to 4 MB |
| Search latency | 61ms | + 180-240ms |
| DB table size (2900 memories) | 12 GB | up to 43 GB |
| New dependencies | none | VectorChord extension, Rust toolchain |

Ogham targets a 14GB laptop. ColBERT storage alone would be prohibitive.

## Decision

Do not add ColBERT to ogham. Dense + keyword two-way search is the right architecture
for personal memory retrieval at this scale. The infrastructure work (VectorChord
integration, ONNX embedding fixes, benchmark tooling) is preserved in the codebase
for any future experiments.

Full details: [reranking evaluation](colbert-evaluation.md) | [retrieval evaluation](colbert-retrieval-evaluation.md)
