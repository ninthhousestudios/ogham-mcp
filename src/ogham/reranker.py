"""Optional cross-encoder reranking for search results.

Uses FlashRank (ms-marco-MiniLM-L-12-v2, 21MB) for fast CPU-only
reranking. Lazy-loaded on first use -- no cost if not enabled.

Enable via RERANK_ENABLED=true in environment.
"""

import logging

logger = logging.getLogger(__name__)

_ranker = None


def _get_ranker():
    """Lazy-load FlashRank singleton."""
    global _ranker
    if _ranker is not None:
        return _ranker

    try:
        from flashrank import Ranker

        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        logger.info("FlashRank reranker loaded")
        return _ranker
    except ImportError:
        logger.debug("flashrank not installed, reranking disabled")
        return None


def rerank_results(
    query: str,
    results: list[dict],
    top_k: int = 10,
    alpha: float = 0.55,
) -> list[dict]:
    """Rerank search results using FlashRank cross-encoder.

    Blends the original retrieval score with the cross-encoder score:
        final = (1 - alpha) * retrieval_score + alpha * ce_score

    Args:
        query: The search query text.
        results: List of memory dicts with 'content' and 'relevance' keys.
        top_k: Number of results to return after reranking.
        alpha: Weight for cross-encoder score (0 = retrieval only, 1 = CE only).

    Returns:
        Reranked list of memory dicts, truncated to top_k.
    """
    ranker = _get_ranker()
    if ranker is None or not results:
        return results[:top_k]

    from flashrank import RerankRequest

    passages = [{"id": i, "text": r.get("content", "")} for i, r in enumerate(results)]

    try:
        req = RerankRequest(query=query, passages=passages)
        ranked = ranker.rerank(req)
    except Exception:
        logger.exception("FlashRank reranking failed, returning original order")
        return results[:top_k]

    # Build reranked list with blended scores
    reranked = []
    for item in ranked:
        idx = item["id"]
        if idx < len(results):
            result = results[idx].copy()
            ce_score = float(item["score"])
            retrieval_score = float(result.get("relevance", 0))
            result["relevance"] = (1 - alpha) * retrieval_score + alpha * ce_score
            result["ce_score"] = ce_score
            reranked.append(result)

    return reranked[:top_k]
