# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.9.2] - 2026-04-12

### Fixed
- **Hybrid search upgrade-path regression** — on every `sql/upgrade.sh` run, `hybrid_search_memories` was left with a raw-score linear-combination fusion formula (`semantic_weight * similarity + full_text_weight * keyword_rank`) instead of true Reciprocal Rank Fusion. Two sources contributed: `sql/migrations/013_halfvec_compression.sql` recreated the function body with the broken formula, and the unnumbered `sql/migrations/update_search_function.sql` sorted after `017_rrf_bm25.sql` alphabetically and reinforced it. Meanwhile `017_rrf_bm25.sql` itself was comment-only and did nothing. Users who fresh-installed via `sql/schema.sql` or `sql/schema_postgres.sql` were unaffected; users who upgraded via `upgrade.sh` between v0.8.3 and v0.9.1 received degraded hybrid search. The reported v0.9.1 benchmark numbers (R@10 0.972, LongMemEval QA 0.918) apply to fresh installs only. Fix: `update_search_function.sql` removed, `017_rrf_bm25.sql` rewritten as a functional migration that restores RRF, drops the 9-param overload introduced by 013 to prevent Postgres function-resolution ambiguity, and is a halfvec-gated safe no-op on Supabase Cloud (768-dim) deployments.
- **`test_extract_entities_cap`** asserted `<= 15` but `extract_entities()` caps at 20. Assertion and fixture corrected to match code.

### Changed
- **`sql/schema_selfhost_supabase.sql` hybrid_search_memories defaults** — `full_text_weight=1.0, semantic_weight=1.0, rrf_k=60` → `0.3, 0.7, 10`. Empirically-tuned values from `schema.sql` / `schema_postgres.sql` had never been propagated to the self-hosted-Supabase schema. Formula shape (Cormack-original RRF via UNION+GROUP+SUM) unchanged — it is functionally valid RRF, just a different variant than the FULL-OUTER-JOIN form in the other two schemas.
- **`geotext` import** in `src/ogham/extraction.py` hoisted to top-level. The soft-import guard (`try: from geotext import GeoText / except ImportError: _GeoText = None`) was vestigial — `geotext>=0.4` has been a hard dependency since v0.9.0 (`pyproject.toml:17`).

### Migration housekeeping
- Duplicate `008` collision resolved: `sql/migrations/008_ccf_search.sql` → `008a_ccf_search.sql`. `008_memory_relationships.sql` creates the `memory_relationships` table; `008a_ccf_search.sql` creates `hybrid_search_memories` which references that table. Alphabetical sort now executes them in the correct order.

### Schema upgrade note
Users affected by the hybrid-search regression should re-run `sql/migrations/017_rrf_bm25.sql` after pulling v0.9.2. No data migration required. Fresh installs do nothing.

## [0.9.1] - 2026-04-08

### Fixed
- **ONNX provider**: `_embed_onnx()` was using `result["dense"]` against an `OnnxResult` dataclass and failed on cache misses with `'OnnxResult' object is not subscriptable`. Now uses `result.dense`. Cached queries worked because the bug only triggered on the embedding path -- thanks to Josh for the report.

### Added
- **Preference extraction** -- new `preference:` entity tag, with 280 trigger words across all 18 supported languages (English, German, French, Spanish, Italian, Portuguese, Dutch, Russian, Polish, Turkish, Irish, Arabic, Hindi, Japanese, Korean, Chinese, Ukrainian, plus pt-BR variants). Detects "prefer", "favorite", "like better", "rather", "always get", "go-to" and equivalents at ingest time. Extracted memories get tagged automatically and surface for recommendation-style queries.
- **Multi-word phrase matching** in entity extraction `_match()` -- phrases like "always get", "better than", "tercih ederim" now match via substring lookup. Single Latin words still use word-boundary matching to avoid partial false positives.
- **`format_results_with_sessions()`** in `service.py` -- formats search results with timeline table at the top, session boundary headers (`=== SESSION: 2024-04-12 ===`), entity and date annotations per memory. Used by gateway chat endpoints and benchmark scripts. Produces the same enriched context that drove the LongMemEval 91.8% and BEAM 100K 0.554 results.
- **`RERANK_MODEL` config option** -- choose between `flashrank` (default, 33M params, CPU-only) and `bge` (BAAI/bge-reranker-v2-m3, 568M params, multilingual, via `sentence-transformers`). Both disabled by default. Benchmark experiments showed neither helps when retrieval is already above 95% R@10, but the plumbing is here for users who want to test it on their own data.
- **BEAM batch QA harness** (`benchmarks/beam_batch.py`) -- three-phase pipeline (`prepare` / `submit` / `judge`) using OpenAI Batch API for 50% cost savings. Implements the BEAM paper's exact Appendix G nugget judge prompt for direct comparability to published numbers. Skips event_ordering (which uses Kendall tau-b in the paper, needs separate equivalence-detector pipeline).

### Changed
- BEAM benchmark default `EMBEDDING_BATCH_SIZE` is now `None` (uses provider default) instead of hardcoded 1000. Gemini caps batches at 100 requests; Voyage allows 1000. The previous hardcoded value broke ingest on Gemini.
- Added comment to `BEAM_GRAPH_DEPTH` env var explaining why graph augmentation is disabled by default. Experiments showed that graph_depth=1 with the current memory-similarity graph hurts retrieval significantly (-13 to -44pp across all categories), because similarity edges duplicate vector-search hits and displace diverse evidence. The win requires a real entity-relationship graph (memory→entity edges), planned for v0.10.

### Documentation
- New blog post: ["BEAM benchmark -- a fair look at where we stand on long-term memory"](https://ogham-mcp.dev/blog/beam-benchmark-v090/) -- 0.554 nugget score vs paper's 0.358, honest gaps in retrieval, what we're fixing next.
- New blog post: ["From 62% to 92% -- what we learned about reading, not retrieval"](https://ogham-mcp.dev/blog/longmemeval-92/) -- the LongMemEval journey from 62% to 91.8% via context engineering.

### Benchmark numbers (v0.9.1)
- **LongMemEval QA accuracy**: 0.918 (459/500), gpt-5.4-mini reader with reasoning, paper Appendix G judge
- **LongMemEval R@10**: 0.972
- **BEAM 100K QA nugget score**: 0.554 (vs paper best 0.358)
- **BEAM 100K R@10**: 0.737

## [0.3.0] - 2026-03-07 — Relationship Graph

### Added
- **Relationship graph** — `memory_relationships` table with typed, weighted edges (`similar`, `supports`, `contradicts`, `related`, `follows`, `derived_from`). Built entirely in PostgreSQL with recursive CTEs — no separate graph database or LLM needed
- **Auto-linking** — new memories are automatically linked to similar existing memories on store via HNSW vector search (threshold 0.85, max 5 links). One database round-trip, no LLM in the write path
- `explore_knowledge` tool — hybrid search seeds + recursive CTE graph traversal. Finds memories by meaning, then expands via relationship edges to pull in connected context
- `find_related` tool — traverses the graph outward from a known memory ID for impact analysis
- `store_decision` tool — stores architectural decisions with structured metadata (rationale, alternatives) and `supports` edges to related memories
- `link_unlinked` tool — backfills auto-links for memories that predate the relationship graph. Configurable threshold and batch size
- `store_memory` gains `auto_link` parameter (default `True`) and returns `links_created` count
- Stress test script (`tests/bench_stress.py`) — imports 1000+ memories, verifies dedup, backfills auto-links, benchmarks graph operations at scale
- Benchmark coverage for auto-link, explore graph, and get related operations

### Fixed
- `link_unlinked_memories` RPC infinite loop — `PERFORM` discarded return value, causing memories with no similar neighbors to be reprocessed indefinitely. Fixed with `SELECT INTO` and conditional increment

### Schema Changes
- Run `sql/migrations/008_memory_relationships.sql`, `009_graph_explorer.sql`, `010_impact_analysis.sql` in order, or re-run `sql/schema.sql` for fresh installs
- New table: `memory_relationships` (source_id, target_id, relationship, strength, metadata)
- New enum: `relationship_type`
- New RPCs: `auto_link_memory`, `link_unlinked_memories`, `explore_memory_graph`, `get_related_memories`

## [0.2.1] - 2026-03-05 — Supabase Best Practices

### Changed
- **RLS hardened** — `FORCE ROW LEVEL SECURITY` on `memories` and `profile_settings`, deny `anon` role access by default
- **Composite index** — replaced separate `profile` and `created_at` indexes with `(profile, created_at DESC)` for faster filtered queries
- **Keyset pagination** — `get_all_memories_full` and `get_all_memories_content` now use cursor-based pagination instead of OFFSET
- **Batch re-embed** — `re_embed_all` uses new `batch_update_embeddings` RPC (batches of 50) instead of individual updates

### Added
- `batch_update_embeddings` RPC function for efficient bulk embedding updates
- `CHECK` constraint on `profile_settings.ttl_days` (`>= 1` or NULL)
- RLS policies on `profile_settings` table

### Schema Changes
- Run `sql/migrate_best_practices.sql` to upgrade existing installs

## [0.2.0] - 2026-03-05 — Quality of Life Release

### Added
- **LRU embedding cache** with configurable max size (`EMBEDDING_CACHE_MAX_SIZE`, default 1000) and `get_cache_stats()` tool for monitoring
- **Memory expiration** with profile-bound TTLs — `set_profile_ttl()` sets days until memories expire, `cleanup_expired()` permanently removes them
- **Profile settings table** (`profile_settings`) for storing per-profile TTL configuration
- **MCP prompts** — `summarize-recent`, `find-decisions`, `profile-overview`, `cleanup-check`
- **HTTP health endpoint** — optional `GET /health` on configurable port (`ENABLE_HTTP_HEALTH`, `HEALTH_PORT`)
- **CLI tool** — `ogham` command with subcommands: serve, health, profiles, stats, search, list, cleanup, export, import, openapi
- **OpenAPI spec generation** — `ogham openapi` generates `docs/openapi.json` from MCP tool definitions
- **Export/import** — `export_profile()` exports as JSON or Markdown, `import_memories_tool()` imports with dedup

### Changed
- `match_memories`, `get_profile_counts`, `get_memory_stats_sql` RPCs now filter expired memories
- `list_recent_memories` now filters expired memories and returns `expires_at`
- `store_memory` now computes `expires_at` from profile TTL when set
- Entry point changed: `ogham` now runs CLI (use `ogham serve` for MCP server, or `ogham-serve`)
- Embedding cache changed from unbounded dict to OrderedDict-based LRU

### Schema Changes
- New table: `profile_settings` (profile, ttl_days, created_at, updated_at)
- New column: `memories.expires_at` (timestamptz, nullable)
- New RPCs: `cleanup_expired_memories()`, `count_expired_memories()`
- Run `sql/migrate_expiration.sql` to upgrade existing installs

## [0.1.1] - 2026-03-05

### Added
- `health_check` tool for diagnosing connection issues
- Startup validation -- exits with clear errors if Supabase or Ollama aren't reachable
- Retry with exponential backoff on transient failures (embedding: 3 attempts, DB reads: 2)
- Input validation for content, threshold, and limit parameters
- Embedding cache (SHA256-keyed) to skip re-embedding identical content
- SQL aggregation functions (`get_profile_counts`, `get_memory_stats_sql`) replacing Python-side counting
- Progress reporting for `re_embed_all`
- Tool execution timing logged to stderr

### Fixed
- `update_memory` / `store_memory` crash when Supabase returns empty results
- `delete_memory` / `update_memory` could affect memories in other profiles

### Changed
- `get_stats()` returns `total` instead of `total_memories`
- OpenAI is now an optional dependency (`uv sync --extra openai`)
- New RPC functions required -- re-run `sql/schema.sql`

## [0.1.0] - 2026-03-05

### Added
- Initial release
- MCP server with stdio transport
- Supabase PostgreSQL + pgvector backend
- Ollama and OpenAI embedding providers
- Memory profiles (Severance-style partitioning)
- Semantic search with cosine similarity
- Tag and source filtering
- Docker and native (UV) deployment options
- Support for Claude Desktop, Cursor, Claude Code, VS Code, Codex, Windsurf
