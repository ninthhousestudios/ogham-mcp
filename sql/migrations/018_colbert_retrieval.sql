-- Migration 018: Three-way RRF search with ColBERT MaxSim retrieval
--
-- Adds hybrid_search_memories_colbert() — a three-leg retrieval function
-- that fuses dense (HNSW), keyword (tsvector), and ColBERT MaxSim (VectorChord)
-- results via Reciprocal Rank Fusion.
--
-- Requires:
--   - pgvector (vector type, HNSW index)
--   - VectorChord (vchord extension, @# MaxSim operator, vchordrq index)
--   - colbert_tokens vector(1024)[] column on memories table
--
-- Apply with:
--   psql -v ON_ERROR_STOP=0 ogham < sql/migrations/018_colbert_retrieval.sql

-- Add colbert_tokens column if not present
ALTER TABLE memories ADD COLUMN IF NOT EXISTS colbert_tokens vector(1024)[];

-- Create VectorChord MaxSim index (vchordrq = IVF + RaBitQ)
-- This index accelerates the @# operator for ColBERT late-interaction search.
-- DROP + CREATE because CREATE INDEX IF NOT EXISTS doesn't support USING vchordrq.
DROP INDEX IF EXISTS memories_colbert_maxsim_idx;
CREATE INDEX memories_colbert_maxsim_idx
    ON memories USING vchordrq (colbert_tokens vector_maxsim_ops);

-- Three-way RRF search function: dense + keyword + ColBERT MaxSim
CREATE OR REPLACE FUNCTION hybrid_search_memories_colbert(
    query_text text,
    query_embedding vector(1024),
    query_colbert_tokens vector(1024)[],
    match_count integer DEFAULT 10,
    filter_profile text DEFAULT 'default',
    filter_tags text[] DEFAULT NULL,
    filter_source text DEFAULT NULL,
    rrf_k integer DEFAULT 60
)
RETURNS TABLE(
    id uuid,
    content text,
    metadata jsonb,
    source text,
    profile text,
    tags text[],
    similarity float,
    keyword_rank float,
    colbert_score float,
    relevance float,
    access_count integer,
    last_accessed_at timestamptz,
    confidence float,
    created_at timestamptz,
    updated_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH semantic AS (
        SELECT
            m.id,
            row_number() OVER (
                ORDER BY (m.embedding::halfvec(1024) <=> query_embedding::halfvec(1024))
            )::integer AS rank,
            (1 - (m.embedding::halfvec(1024) <=> query_embedding::halfvec(1024)))::float AS sim
        FROM memories m
        WHERE m.profile = filter_profile
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND (filter_tags IS NULL OR m.tags && filter_tags)
          AND (filter_source IS NULL OR m.source = filter_source)
        ORDER BY m.embedding::halfvec(1024) <=> query_embedding::halfvec(1024)
        LIMIT match_count * 3
    ),
    keyword AS (
        SELECT
            m.id,
            row_number() OVER (
                ORDER BY ts_rank_cd(m.fts, websearch_to_tsquery('english', query_text), 34) DESC
            )::integer AS rank,
            ts_rank_cd(m.fts, websearch_to_tsquery('english', query_text), 34)::float AS kw_rank
        FROM memories m
        WHERE m.fts @@ websearch_to_tsquery('english', query_text)
          AND m.profile = filter_profile
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND (filter_tags IS NULL OR m.tags && filter_tags)
          AND (filter_source IS NULL OR m.source = filter_source)
        ORDER BY ts_rank_cd(m.fts, websearch_to_tsquery('english', query_text), 34) DESC
        LIMIT match_count * 3
    ),
    colbert AS (
        SELECT
            m.id,
            row_number() OVER (
                ORDER BY m.colbert_tokens @# query_colbert_tokens DESC
            )::integer AS rank,
            (m.colbert_tokens @# query_colbert_tokens)::float AS cb_score
        FROM memories m
        WHERE m.profile = filter_profile
          AND m.colbert_tokens IS NOT NULL
          AND (m.expires_at IS NULL OR m.expires_at > now())
          AND (filter_tags IS NULL OR m.tags && filter_tags)
          AND (filter_source IS NULL OR m.source = filter_source)
        ORDER BY m.colbert_tokens @# query_colbert_tokens DESC
        LIMIT match_count * 3
    ),
    all_ids AS (
        SELECT semantic.id FROM semantic
        UNION
        SELECT keyword.id FROM keyword
        UNION
        SELECT colbert.id FROM colbert
    ),
    fused AS (
        SELECT
            a.id,
            coalesce(1.0 / (rrf_k + s.rank), 0)::float
                + coalesce(1.0 / (rrf_k + k.rank), 0)::float
                + coalesce(1.0 / (rrf_k + c.rank), 0)::float AS rrf_score,
            coalesce(s.sim, 0)::float AS similarity,
            coalesce(k.kw_rank, 0)::float AS keyword_rank,
            coalesce(c.cb_score, 0)::float AS colbert_score
        FROM all_ids a
        LEFT JOIN semantic s ON a.id = s.id
        LEFT JOIN keyword k ON a.id = k.id
        LEFT JOIN colbert c ON a.id = c.id
    )
    SELECT
        m.id,
        m.content,
        m.metadata,
        m.source,
        m.profile,
        m.tags,
        f.similarity,
        f.keyword_rank,
        f.colbert_score,
        (f.rrf_score * m.confidence)::float AS relevance,
        m.access_count,
        m.last_accessed_at,
        m.confidence,
        m.created_at,
        m.updated_at
    FROM fused f
    JOIN memories m ON m.id = f.id
    ORDER BY relevance DESC
    LIMIT match_count;
END;
$$;
