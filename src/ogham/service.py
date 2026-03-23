"""Shared memory service pipeline.

Used by both the MCP tool layer (tools/memory.py) and the gateway REST API.
Handles: content validation, date extraction, entity extraction,
importance scoring, embedding generation, surprise scoring,
storage, and auto-linking.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ogham.database import auto_link_memory as db_auto_link
from ogham.database import get_profile_ttl as db_get_profile_ttl
from ogham.database import hybrid_search_memories, record_access
from ogham.database import store_memory as db_store
from ogham.embeddings import generate_embedding
from ogham.extraction import (
    compute_importance,
    extract_dates,
    extract_entities,
    extract_query_anchors,
    extract_recurrence,
    has_temporal_intent,
    is_multi_hop_temporal,
    is_ordering_query,
    resolve_temporal_query,
)

logger = logging.getLogger(__name__)


def store_memory_enriched(
    content: str,
    profile: str,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    auto_link: bool = True,
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Full store pipeline: validation, extraction, embedding, scoring, store, link.

    Returns the stored memory dict with id, created_at, links_created, etc.
    """
    # Lazy import to avoid circular dependency with tools/memory.py
    from ogham.tools.memory import _require_content

    _require_content(content)

    # Mask secrets before storing (protects all paths: hooks, MCP tools, gateway, CLI)
    from ogham.hooks import _mask_secrets

    content = _mask_secrets(content)

    # Auto-extract dates into metadata
    dates = extract_dates(content)
    if dates:
        if metadata is None:
            metadata = {}
        metadata["dates"] = dates

    # Auto-extract entities as tags
    entity_tags = extract_entities(content)
    if entity_tags:
        if tags is None:
            tags = []
        else:
            tags = list(tags)
        tags.extend(entity_tags)

    # Auto-extract recurrence (multilingual, 16 languages)
    recurrence_days = extract_recurrence(content)

    # Compute importance score from content signals
    importance = compute_importance(content, tags)

    # Generate embedding (skip if pre-computed, e.g. from gateway cache)
    if embedding is None:
        embedding = generate_embedding(content)

    # Compute surprise score: how novel is this vs existing memories?
    surprise = 0.5
    try:
        existing = hybrid_search_memories(
            query_text=content[:200],
            query_embedding=embedding,
            profile=profile,
            limit=3,
        )
        if existing:
            max_sim = max(r.get("similarity", 0) for r in existing)
            surprise = round(1.0 - max_sim, 3)
    except Exception:
        logger.debug("Surprise scoring skipped: search failed, using default 0.5")

    # TTL
    ttl_days = db_get_profile_ttl(profile)
    expires_at = None
    if ttl_days is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()

    # Store
    result = db_store(
        content=content,
        embedding=embedding,
        profile=profile,
        metadata=metadata,
        source=source,
        tags=tags,
        expires_at=expires_at,
        importance=importance,
        recurrence_days=recurrence_days,
        surprise=surprise,
    )

    response: dict[str, Any] = {
        "status": "stored",
        "id": result["id"],
        "profile": profile,
        "created_at": result["created_at"],
        "expires_at": expires_at,
        "importance": importance,
        "surprise": surprise,
    }

    # Auto-link
    if auto_link:
        links_created = db_auto_link(
            memory_id=result["id"],
            embedding=embedding,
            profile=profile,
        )
        response["links_created"] = links_created

    return response


def search_memories_enriched(
    query: str,
    profile: str,
    limit: int = 10,
    tags: list[str] | None = None,
    source: str | None = None,
    graph_depth: int = 0,
    embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Full search pipeline: embed query, search, optional graph traversal, record access.

    If the query has temporal intent, resolves date references and boosts
    results whose dates fall within the resolved range.
    """
    if embedding is None:
        embedding = generate_embedding(query)

    # Ordering queries: broad search, chronological sort by content date
    if is_ordering_query(query):
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit * 5,
            tags=tags,
            source=source,
        )
        if results:
            # Sort by date extracted from content, then trim
            for r in results:
                r["_sort_date"] = _extract_memory_date(r) or "9999"
            results.sort(key=lambda r: r["_sort_date"])
            results = results[:limit]
            record_access([r["id"] for r in results])
        return results

    # Multi-hop temporal: entity-centric bridge retrieval
    if is_multi_hop_temporal(query):
        bridge_results = _bridge_retrieval(query, profile, limit, tags, source)
        if bridge_results:
            # Merge bridge results with standard search
            results = _merge_bridge_results(
                bridge_results, query, embedding, profile, limit, tags, source
            )
            # No temporal re-ranking on bridge results -- tested in runs 10-11,
            # added noise without improving scores. Bridge path stays clean.
            if results:
                record_access([r["id"] for r in results])
            return results

    # Standard search path
    fetch_limit = limit * 3 if has_temporal_intent(query) else limit

    if graph_depth > 0:
        from ogham.database import graph_augmented_search

        results = graph_augmented_search(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=fetch_limit,
            graph_depth=graph_depth,
            tags=tags,
            source=source,
        )
    else:
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=fetch_limit,
            tags=tags,
            source=source,
        )

    # Single-anchor temporal re-ranking
    if results and has_temporal_intent(query):
        results = _temporal_rerank(results, query)
        results = results[:limit]

    if results:
        record_access([r["id"] for r in results])

    return results


def _bridge_retrieval(
    query: str,
    profile: str,
    limit: int,
    tags: list[str] | None,
    source: str | None,
) -> list[dict[str, Any]]:
    """Entity-centric bridge retrieval for multi-hop temporal queries.

    Extracts entity anchors from the query, runs separate keyword searches
    for each anchor, and returns results grouped by anchor.
    """
    anchors = extract_query_anchors(query)
    if not anchors:
        return []

    # Split limit evenly across anchors
    per_anchor_limit = max(limit, 10) // len(anchors) if anchors else limit

    all_results = []
    for anchor in anchors:
        # Path A: Semantic + keyword hybrid search
        anchor_embedding = generate_embedding(anchor)
        results = hybrid_search_memories(
            query_text=anchor,
            query_embedding=anchor_embedding,
            profile=profile,
            limit=per_anchor_limit * 2,
            tags=tags,
            source=source,
        )

        if results:
            for r in results:
                r["_bridge_anchor"] = anchor
            all_results.append(results)

    if not all_results:
        return []

    # Interleave results from each anchor (round-robin)
    interleaved = []
    max_len = max(len(group) for group in all_results)
    for i in range(max_len):
        for group in all_results:
            if i < len(group):
                interleaved.append(group[i])

    return interleaved


def _merge_bridge_results(
    bridge_results: list[dict[str, Any]],
    original_query: str,
    original_embedding: list[float],
    profile: str,
    limit: int,
    tags: list[str] | None,
    source: str | None,
) -> list[dict[str, Any]]:
    """Merge bridge retrieval results with standard search, deduplicating.

    Bridge results get a 1.5x boost. The final list is deduped by ID
    and trimmed to limit.
    """
    # Also run the standard search as fallback
    standard_results = hybrid_search_memories(
        query_text=original_query,
        query_embedding=original_embedding,
        profile=profile,
        limit=limit * 2,
        tags=tags,
        source=source,
    )

    # Logarithmic boost for bridge results (run 7 config -- best MRR)
    import math

    for r in bridge_results:
        if "relevance" in r and r["relevance"] is not None:
            ccf = r["relevance"]
            r["relevance"] = ccf + 0.3 * math.log1p(ccf)

    # Merge: bridge results first, then standard, dedup by ID
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for r in bridge_results + standard_results:
        rid = r.get("id", "")
        if rid not in seen_ids:
            seen_ids.add(rid)
            merged.append(r)

    # Timestamp tiebreaker for same-score results
    for r in merged:
        created = r.get("created_at", "")
        if created and "relevance" in r and r["relevance"] is not None:
            try:
                ts = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                r["relevance"] += ts.timestamp() * 1e-15
            except (ValueError, TypeError):
                pass

    # Sort by relevance
    merged.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return merged[:limit]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _exact_content_search(anchor: str, profile: str, limit: int) -> list[dict[str, Any]]:
    """Direct content search using ILIKE -- bypasses ts_rank normalisation.

    Finds entity mentions buried in long documents where tsvector
    scores them too low (e.g. "Emma" in 18K chars of marketing content).
    """
    from ogham.database import get_backend

    backend = get_backend()

    # Extract the most specific word from the anchor (longest, likely a name)
    words = [w for w in anchor.split() if len(w) > 2]
    if not words:
        return []

    # Use the most distinctive word (longest, likely a proper noun)
    search_term = max(words, key=len)

    try:
        if hasattr(backend, "_execute"):
            # Postgres backend -- direct SQL
            sql = """
                SELECT id, content, metadata, source, profile, tags,
                       confidence, created_at, updated_at,
                       0.5::float AS similarity, 0.5::float AS relevance
                FROM memories
                WHERE profile = %(profile)s
                  AND content ILIKE %(pattern)s
                  AND (expires_at IS NULL OR expires_at > now())
                ORDER BY created_at DESC
                LIMIT %(limit)s
            """
            results = backend._execute(
                sql,
                {"profile": profile, "pattern": f"%{search_term}%", "limit": limit},
                fetch="all",
            )
            return results if results else []
        elif hasattr(backend, "_get_client"):
            # Supabase/PostgREST backend
            result = (
                backend._get_client()
                .from_("memories")
                .select("*")
                .eq("profile", profile)
                .ilike("content", f"*{search_term}*")
                .limit(limit)
                .execute()
            )
            return result.data if result.data else []
    except Exception as e:
        logger.debug("Exact content search failed for '%s': %s", search_term, e)

    return []


_CONTENT_DATE_RE = re.compile(r"\[Date:\s*(\d{4}-\d{2}-\d{2})\]")

# Direction keywords for asymmetric temporal decay
_AFTER_WORDS = frozenset({"after", "since", "following", "subsequent", "later"})
_BEFORE_WORDS = frozenset({"before", "prior", "previous", "earlier", "preceding"})


def _extract_memory_date(r: dict[str, Any]) -> str | None:
    """Extract a date from a memory (metadata > content prefix > created_at)."""
    meta_dates = r.get("metadata", {}).get("dates", [])
    if meta_dates:
        return meta_dates[0]

    content = r.get("content", "")
    date_match = _CONTENT_DATE_RE.search(content)
    if date_match:
        return date_match.group(1)

    created = str(r.get("created_at", ""))[:10]
    if created and len(created) == 10:
        return created

    return None


def _detect_direction(query: str) -> str:
    """Detect temporal direction: 'future', 'past', or 'near'."""
    words = set(query.lower().split())
    if words & _AFTER_WORDS:
        return "future"
    if words & _BEFORE_WORDS:
        return "past"
    return "near"


def _temporal_rerank(
    results: list[dict[str, Any]], query: str, sigma: float = 3.0
) -> list[dict[str, Any]]:
    """Gaussian decay + directional hard penalty temporal re-ranking.

    Uses Gaussian decay centered on the anchor date. σ controls the window
    width (3 = tight week-scale, 7 = loose). Wrong-direction results get
    hard 0.1x penalty (the "temporal cliff"). Sub-day precision via
    fractional days from timestamps.

    Decay: exp(-delta²/2σ²) concentrates boost near anchor
    Directional: 0.1x for wrong side of anchor (squash, not just decay)
    Same-day grace: delta < 1 day gets 1.5x boost
    Tiebreaker: 1e-6 * timestamp for same-score results
    """
    import math

    date_range = resolve_temporal_query(query)
    if not date_range:
        return results

    range_start, range_end = date_range
    try:
        anchor_start = datetime.fromisoformat(range_start)
        anchor_end = datetime.fromisoformat(range_end)
        anchor = anchor_start + (anchor_end - anchor_start) / 2
    except (ValueError, TypeError):
        return results

    direction = _detect_direction(query)

    for r in results:
        mem_date_str = _extract_memory_date(r)
        if not mem_date_str:
            continue

        try:
            # Use full timestamp precision when available
            if len(mem_date_str) > 10:
                mem_dt = datetime.fromisoformat(mem_date_str.replace("Z", "+00:00"))
            else:
                mem_dt = datetime.fromisoformat(mem_date_str)
        except (ValueError, TypeError):
            continue

        # Delta in fractional days (sub-day precision)
        delta_days = (mem_dt - anchor).total_seconds() / 86400.0

        # 1. Directional hard penalty (the "cliff")
        dir_multiplier = 1.0
        if direction == "future" and delta_days < -0.5:
            dir_multiplier = 0.1
        elif direction == "past" and delta_days > 0.5:
            dir_multiplier = 0.1

        # 2. Gaussian decay (proximity boost)
        if abs(delta_days) < 0.5:
            # Same-day grace period
            decay = 1.5
        else:
            decay = 1.0 + 0.5 * math.exp(-(delta_days**2) / (2 * sigma**2))

        # 3. Tiebreaker: tiny timestamp fraction
        tiebreak = mem_dt.timestamp() * 1e-15 if hasattr(mem_dt, "timestamp") else 0

        # 4. Apply
        if "relevance" in r and r["relevance"] is not None:
            r["relevance"] = r["relevance"] * dir_multiplier * decay + tiebreak

    results.sort(key=lambda r: r.get("relevance", 0), reverse=True)
    return results
