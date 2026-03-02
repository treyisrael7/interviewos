"""Retrieval: embed query, search document_chunks, MMR diversification."""

import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import DocumentChunk
from app.services.ingestion import _create_embeddings

# Query keywords -> suggested section types for JD filtering
QUERY_SECTION_HINTS: dict[str, list[str]] = {
    "skill": ["qualifications", "tools_technologies", "preferred_qualifications"],
    "qualification": ["qualifications", "preferred_qualifications"],
    "responsibilit": ["responsibilities"],
    "requirement": ["qualifications"],
    "salary": ["compensation", "about"],
    "pay": ["compensation", "about"],
    "compensation": ["compensation", "about"],
    "location": ["location", "about"],
    "remote": ["location", "about"],
    "company": ["company_info", "about"],
    "role": ["position_summary", "about"],
    "job": ["position_summary", "about"],
}


def suggest_section_filters(query: str) -> list[str] | None:
    """If query suggests specific sections, return section_types to filter."""
    q = query.lower().strip()
    words = set(re.findall(r"\b\w+\b", q))
    suggested: set[str] = set()
    for hint, sections in QUERY_SECTION_HINTS.items():
        if hint in q or any(hint in w for w in words):
            suggested.update(sections)
    return list(suggested) if suggested else None


def embed_query(query: str) -> list[float]:
    """Embed a single query string. Returns embedding vector."""
    embeddings = _create_embeddings([query])
    return embeddings[0]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity (dot product for normalized vectors)."""
    return sum(x * y for x, y in zip(a, b))


def _mmr_select(
    candidates: list[dict],
    query_embedding: list[float],
    top_k: int,
    lambda_: float,
) -> list[dict]:
    """
    Maximal Marginal Relevance: select diverse top_k from candidates.
    candidates have: id, page_number, content, embedding, score (sim to query).
    """
    if len(candidates) <= top_k:
        for c in candidates[:top_k]:
            c.pop("embedding", None)
        return candidates[:top_k]

    selected: list[dict] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_mmr = float("-inf")
        for i, c in enumerate(remaining):
            sim_q = c["score"]
            max_sim_sel = 0.0
            if selected:
                for s in selected:
                    sim_d = _cosine_sim(c["embedding"], s["embedding"])
                    max_sim_sel = max(max_sim_sel, sim_d)
            mmr = lambda_ * sim_q - (1 - lambda_) * max_sim_sel
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        if best_idx < 0:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)

    for c in selected:
        c.pop("embedding", None)
    return selected


async def retrieve_chunks(
    db: AsyncSession,
    document_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int,
    include_low_signal: bool = False,
    section_types: list[str] | None = None,
    doc_domain: str | None = None,
) -> list[dict]:
    """
    Search document_chunks by cosine similarity.
    Fetches top top_n_candidates, filters low-signal, applies MMR for diversity.
    By default excludes is_low_signal chunks; pass include_low_signal=true for contact queries.
    Returns list of {chunk_id, page_number, snippet, score, is_low_signal}.
    """
    distance_col = DocumentChunk.embedding.cosine_distance(query_embedding)
    score_col = (1 - distance_col).label("score")
    limit = max(top_k, settings.top_n_candidates)

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.page_number,
            DocumentChunk.content,
            DocumentChunk.embedding,
            DocumentChunk.is_low_signal,
            DocumentChunk.section_type,
            score_col,
        )
        .where(DocumentChunk.document_id == document_id)
        .where(DocumentChunk.embedding.isnot(None))
        .order_by(distance_col.asc())
        .limit(limit)
    )
    if not include_low_signal:
        stmt = stmt.where(DocumentChunk.is_low_signal == False)
    if section_types:
        stmt = stmt.where(DocumentChunk.section_type.in_(section_types))
    if doc_domain:
        stmt = stmt.where(DocumentChunk.doc_domain == doc_domain)

    result = await db.execute(stmt)
    rows = result.all()

    candidates = [
        {
            "chunk_id": str(row.id),
            "page_number": row.page_number,
            "snippet": row.content,
            "score": round(float(row.score), 6),
            "is_low_signal": bool(row.is_low_signal),
            "section_type": getattr(row, "section_type", None),
            "embedding": row.embedding,
        }
        for row in rows
    ]

    diversified = _mmr_select(
        candidates,
        query_embedding,
        top_k,
        settings.mmr_lambda,
    )

    return [
        {
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "snippet": c["snippet"],
            "score": c["score"],
            "is_low_signal": c.get("is_low_signal", False),
            "section_type": c.get("section_type"),
        }
        for c in diversified
    ]
