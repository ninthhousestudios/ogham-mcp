-- Migration 020: Dim-aware halfvec casts in all RPC functions and HNSW index.
--
-- Background
-- ----------
-- Migrations 013 and 017 hardcode `halfvec(512)` casts inside the function
-- bodies of auto_link_memory, match_memories, hybrid_search_memories,
-- batch_check_duplicates and in the memories_embedding_idx HNSW index.
--
-- That blocks first-class support for non-512 embedding dimensions
-- (1024 for openai/mistral/voyage defaults, 3072 for openai-large, etc.)
-- even though src/ogham/config.py advertises those as supported.
--
-- See https://github.com/ninthhousestudios/ogham-mcp/issues/24
--
-- This migration
-- ---------------
-- Introspects the actual `memories.embedding` column dimension via
-- pg_attribute + format_type, then uses format() + EXECUTE to (re)create
-- every halfvec-using function with the correct halfvec(N) cast.
--
-- The HNSW index is *not* rebuilt by default — index recreation on a
-- populated table is expensive. Set `ogham.rebuild_hnsw = 'on'` in the
-- session before running the migration to opt in:
--
--     SET ogham.rebuild_hnsw = 'on';
--     \i sql/migrations/020_dim_aware_halfvec.sql
--
-- Idempotent. No data migration. Safe no-op on non-halfvec deployments
-- (e.g. Supabase Cloud at 768 dims using plain vector ops).
--
-- IMPORTANT for future editors: function bodies inside the format() calls
-- below must not contain any literal `%` character. format() treats `%` as
-- a format-spec sigil; a stray `%` (e.g. in a LIKE pattern) will fail at
-- migration time. Use `%%` to embed a literal percent.

do $mig$
declare
    col_type    text;
    embed_dim   int;
    rebuild_idx text;
begin
    -- 1. Introspect column type and dimension.
    select format_type(a.atttypid, a.atttypmod) into col_type
    from pg_attribute a
    join pg_class c on c.oid = a.attrelid
    where c.relname = 'memories'
      and a.attname = 'embedding'
      and a.attnum > 0;

    if col_type is null then
        raise notice 'Migration 020: memories.embedding not found; run your schema file first. Skipping.';
        return;
    end if;

    if col_type not like '%halfvec%' and col_type not like 'vector(%' then
        raise notice 'Migration 020: memories.embedding is % (no typmod or non-halfvec); skipping. Re-run sql/schema.sql for this backend.', col_type;
        return;
    end if;

    -- Extract dim from "vector(N)" or "halfvec(N)" — pgvector stores N raw in atttypmod.
    embed_dim := nullif(substring(col_type from '\((\d+)\)'), '')::int;

    if embed_dim is null then
        raise notice 'Migration 020: could not parse dimension from column type %; skipping.', col_type;
        return;
    end if;

    raise notice 'Migration 020: detected memories.embedding dimension = %', embed_dim;

    -- 1b. Drop every existing overload of the functions we are about to recreate.
    --
    -- CREATE OR REPLACE FUNCTION treats `vector(512)` and `vector(1024)` as
    -- distinct signatures. A user who altered their column from 512 -> 1024
    -- and re-ran this migration would otherwise end up with two overloads
    -- side by side and ambiguous function-resolution errors at runtime.
    -- Enumerating pg_proc by name and dropping each overload is exhaustive
    -- and covers historical drift (e.g. the 9-param vs 10-param hybrid_search
    -- overloads from migrations 013/017).
    declare
        r record;
    begin
        for r in
            select p.proname, pg_get_function_identity_arguments(p.oid) as args
            from pg_proc p
            join pg_namespace n on n.oid = p.pronamespace
            where n.nspname = 'public'
              and p.proname in ('auto_link_memory', 'match_memories',
                                'hybrid_search_memories', 'batch_check_duplicates')
        loop
            execute format('drop function if exists public.%I(%s)', r.proname, r.args);
        end loop;
    end;

    -- 2. auto_link_memory
    execute format($fn$
        create or replace function auto_link_memory(
            new_memory_id uuid,
            new_embedding vector(%1$s),
            link_threshold float default 0.85,
            max_links int default 5,
            filter_profile text default 'default'
        )
        returns integer
        language sql
        security invoker
        set search_path = public, extensions
        as $func$
            with candidates as (
                select m.id, (1 - (m.embedding::halfvec(%1$s) <=> new_embedding::halfvec(%1$s)))::float as similarity
                from memories m
                where m.id != new_memory_id
                  and m.profile = filter_profile
                  and (m.expires_at is null or m.expires_at > now())
                  and 1 - (m.embedding::halfvec(%1$s) <=> new_embedding::halfvec(%1$s)) > link_threshold
                order by m.embedding::halfvec(%1$s) <=> new_embedding::halfvec(%1$s)
                limit max_links
            ),
            inserted as (
                insert into memory_relationships (source_id, target_id, relationship, strength, created_by)
                select new_memory_id, c.id, 'similar', c.similarity, 'auto'
                from candidates c
                on conflict (source_id, target_id, relationship) do nothing
                returning 1
            )
            select count(*)::integer from inserted;
        $func$
    $fn$, embed_dim);

    -- 3. match_memories
    execute format($fn$
        create or replace function match_memories(
            query_embedding vector(%1$s),
            match_threshold float default 0.7,
            match_count int default 10,
            filter_tags text[] default null,
            filter_source text default null,
            filter_profile text default 'default'
        )
        returns table (
            id uuid, content text, metadata jsonb, source text, profile text, tags text[],
            similarity float, relevance float,
            access_count integer, last_accessed_at timestamptz, confidence float,
            created_at timestamptz, updated_at timestamptz
        )
        language plpgsql
        security invoker
        set search_path = public, extensions
        as $func$
        begin
            return query
            select
                m.id, m.content, m.metadata, m.source, m.profile, m.tags,
                (1 - (m.embedding::halfvec(%1$s) <=> query_embedding::halfvec(%1$s)))::float as similarity,
                (
                    (1 - (m.embedding::halfvec(%1$s) <=> query_embedding::halfvec(%1$s))) *
                    ln(1.0 + exp(
                        ln(m.access_count + 1.0) -
                        0.5 * ln(
                            greatest(
                                extract(epoch from now() - coalesce(m.last_accessed_at, m.created_at)) / 86400.0,
                                0.01
                            ) / (m.access_count + 1.0)
                        )
                    ))
                    * m.confidence
                    * (1.0 + g.graph_boost * 0.2)
                )::float as relevance,
                m.access_count, m.last_accessed_at, m.confidence, m.created_at, m.updated_at
            from public.memories m
            left join lateral (
                select coalesce(sum(r.strength), 0.0) as graph_boost
                from memory_relationships r
                where r.target_id = m.id or r.source_id = m.id
            ) g on true
            where
                1 - (m.embedding::halfvec(%1$s) <=> query_embedding::halfvec(%1$s)) > match_threshold
                and (filter_tags is null or m.tags && filter_tags)
                and (filter_source is null or m.source = filter_source)
                and m.profile = filter_profile
                and (m.expires_at is null or m.expires_at > now())
            order by relevance desc
            limit match_count;
        end;
        $func$
    $fn$, embed_dim);

    -- 4. hybrid_search_memories — preserve the canonical RRF formula from migration 017.
    -- (All prior overloads were dropped in step 1b.)
    execute format($fn$
        create or replace function hybrid_search_memories(
            query_text text,
            query_embedding vector,
            match_count integer default 10,
            filter_profile text default 'default',
            filter_tags text[] default null,
            filter_source text default null,
            full_text_weight float default 0.3,
            semantic_weight float default 0.7,
            rrf_k integer default 10,
            filter_profiles text[] default null
        )
        returns table(
            id uuid, content text, metadata jsonb, source text, profile text, tags text[],
            similarity float, keyword_rank float, relevance float,
            access_count integer, last_accessed_at timestamptz, confidence float,
            created_at timestamptz, updated_at timestamptz
        )
        language sql
        set search_path = public, extensions
        as $func$
        with semantic as (
            select
                m.id,
                (1 - (m.embedding::halfvec(%1$s) <=> query_embedding::halfvec(%1$s)))::float as similarity,
                row_number() over (order by m.embedding::halfvec(%1$s) <=> query_embedding::halfvec(%1$s)) as rank_ix
            from memories m
            where (filter_profiles is not null and m.profile = any(filter_profiles)
                   or filter_profiles is null and m.profile = filter_profile)
              and (filter_tags is null or m.tags && filter_tags)
              and (filter_source is null or m.source = filter_source)
              and (m.expires_at is null or m.expires_at > now())
            order by m.embedding::halfvec(%1$s) <=> query_embedding::halfvec(%1$s)
            limit match_count * 3
        ),
        keyword as (
            select
                m.id,
                ts_rank_cd(m.fts, websearch_to_tsquery(query_text), 34)::float as keyword_rank,
                row_number() over (order by ts_rank_cd(m.fts, websearch_to_tsquery(query_text), 34) desc) as rank_ix
            from memories m
            where (filter_profiles is not null and m.profile = any(filter_profiles)
                   or filter_profiles is null and m.profile = filter_profile)
              and m.fts @@ websearch_to_tsquery(query_text)
              and (filter_tags is null or m.tags && filter_tags)
              and (filter_source is null or m.source = filter_source)
              and (m.expires_at is null or m.expires_at > now())
            order by keyword_rank desc
            limit match_count * 3
        ),
        fused as (
            select
                coalesce(s.id, k.id) as id,
                coalesce(s.similarity, 0.0) as similarity,
                coalesce(k.keyword_rank, 0.0) as keyword_rank,
                -- Reciprocal Rank Fusion: position-based, score-agnostic.
                (
                    semantic_weight * (1.0 / (rrf_k + coalesce(s.rank_ix, match_count * 3)))
                    + full_text_weight * (1.0 / (rrf_k + coalesce(k.rank_ix, match_count * 3)))
                ) as score
            from semantic s
            full outer join keyword k on s.id = k.id
        )
        select
            m.id, m.content, m.metadata, m.source, m.profile, m.tags,
            f.similarity, f.keyword_rank,
            (
                f.score
                * (1.0 + ln(m.access_count + 1.0) * 0.1)
                * m.confidence
                * (1.0 + g.graph_boost * 0.2)
            )::float as relevance,
            m.access_count, m.last_accessed_at, m.confidence, m.created_at, m.updated_at
        from fused f
        join memories m on m.id = f.id
        left join lateral (
            select coalesce(sum(r.strength), 0.0) as graph_boost
            from memory_relationships r
            where r.target_id = m.id or r.source_id = m.id
        ) g on true
        order by relevance desc
        limit match_count
        $func$
    $fn$, embed_dim);

    -- 5. batch_check_duplicates
    execute format($fn$
        create or replace function batch_check_duplicates(
            query_embeddings vector(%1$s)[],
            match_threshold float default 0.8,
            filter_profile text default 'default'
        )
        returns boolean[]
        language plpgsql
        security invoker
        set search_path = public, extensions
        as $func$
        declare
            results boolean[];
            i integer;
            found boolean;
        begin
            perform set_config('hnsw.ef_search', '40', true);
            results := array[]::boolean[];
            for i in 1..array_length(query_embeddings, 1) loop
                select exists(
                    select 1 from memories m
                    where m.profile = filter_profile
                      and (m.expires_at is null or m.expires_at > now())
                      and 1 - (m.embedding::halfvec(%1$s) <=> query_embeddings[i]::halfvec(%1$s)) > match_threshold
                    limit 1
                ) into found;
                results := array_append(results, found);
            end loop;
            return results;
        end;
        $func$
    $fn$, embed_dim);

    -- 6. HNSW index rebuild (opt-in).
    rebuild_idx := current_setting('ogham.rebuild_hnsw', true);

    if rebuild_idx = 'on' then
        raise notice 'Migration 020: rebuilding memories_embedding_idx HNSW at dim=% (this may take a while on large tables)', embed_dim;
        execute 'drop index if exists memories_embedding_idx';
        execute format(
            'create index memories_embedding_idx on memories using hnsw ((embedding::halfvec(%1$s)) halfvec_cosine_ops) with (m = 16, ef_construction = 64)',
            embed_dim
        );
        raise notice 'Migration 020: HNSW index rebuilt at dim=%.', embed_dim;
    else
        raise notice 'Migration 020: HNSW index NOT rebuilt. The existing memories_embedding_idx still casts to the original dim and will NOT be used for queries at dim=%. To rebuild: SET ogham.rebuild_hnsw = ''on''; then re-run this migration.', embed_dim;
    end if;

    raise notice 'Migration 020: dim-aware functions installed at dim=%.', embed_dim;
end
$mig$;
