"""Retrieval: embed query, search document_chunks, hybrid merge, MMR diversification."""

import logging
import re
import uuid
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.models import DocumentChunk, InterviewSource

# Backward compat: expand canonical section types to legacy job description section names in DB
SECTION_TYPE_EXPANSION: dict[str, list[str]] = {
    "tools": ["tools", "tools_technologies"],
    "qualifications": ["qualifications", "preferred_qualifications"],
    "about": ["about", "position_summary", "company_info"],
    "other": ["other", "location", "company_info"],
}
from app.services.ingestion import _create_embeddings

logger = logging.getLogger(__name__)

RetrievalMode = Literal["hybrid", "semantic", "keyword"]

# Query keywords -> suggested section types (canonical: responsibilities, qualifications, tools, compensation, about, other)
QUERY_SECTION_HINTS: dict[str, list[str]] = {
    "skill": ["qualifications", "tools"],
    "qualification": ["qualifications"],
    "responsibilit": ["responsibilities"],
    "requirement": ["qualifications"],
    "salary": ["compensation"],
    "salaries": ["compensation"],
    "pay": ["compensation"],
    "compensation": ["compensation"],
    "benefits": ["compensation"],
    "wage": ["compensation"],
    "how much": ["compensation"],
    "location": ["about", "other"],
    "remote": ["about", "other"],
    "company": ["about", "other"],
    "role": ["about"],
    "job": ["about"],
    "tool": ["tools"],
    "tech": ["tools"],
}

KEYWORD_VARIANT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfront\s*-\s*end\b|\bfrontend\b", re.IGNORECASE), '(frontend OR "front-end")'),
    (re.compile(r"\bback\s*-\s*end\b|\bbackend\b", re.IGNORECASE), '(backend OR "back-end")'),
    (re.compile(r"\bfull\s*-\s*stack\b|\bfull\s+stack\b", re.IGNORECASE), '("full stack" OR "full-stack")'),
]

KEYWORD_TECH_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\w)c\+\+(?!\w)", re.IGNORECASE), '("c++" OR cpp)'),
    (re.compile(r"(?<!\w)c#(?!\w)", re.IGNORECASE), '("c#" OR csharp)'),
    (re.compile(r"(?<!\w)(?:\.net|dotnet)(?!\w)", re.IGNORECASE), '(".net" OR dotnet)'),
    (re.compile(r"(?<!\w)(?:node\.js|nodejs)(?!\w)", re.IGNORECASE), '("node.js" OR nodejs)'),
    (re.compile(r"(?<!\w)(?:next\.js|nextjs)(?!\w)", re.IGNORECASE), '("next.js" OR nextjs)'),
    (re.compile(r"(?<!\w)(?:react\.js|reactjs)(?!\w)", re.IGNORECASE), '("react.js" OR reactjs)'),
    (re.compile(r"(?<!\w)(?:postgresql|postgres)(?!\w)", re.IGNORECASE), "(postgresql OR postgres)"),
    (re.compile(r"(?<!\w)pgvector(?!\w)", re.IGNORECASE), '(pgvector OR "pg vector")'),
    (re.compile(r"(?<!\w)aws(?!\w)", re.IGNORECASE), '(aws OR "amazon web services")'),
]


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


def _expanded_section_types(section_types: list[str] | None) -> list[str] | None:
    """Expand canonical section filters to match legacy section names stored in DB."""
    if not section_types:
        return None

    expanded: set[str] = set()
    for st in section_types:
        expanded.add(st)
        expanded.update(SECTION_TYPE_EXPANSION.get(st, []))
    return list(expanded)


def _normalize_keyword_query_text(query_text: str) -> str:
    """
    Light preprocessing for PostgreSQL full-text retrieval.

    Goals:
    - collapse punctuation/whitespace noise common in user questions
    - preserve important technical tokens used in job descriptions
    - add a few hand-maintained spelling/format variants that improve keyword recall
    """
    normalized = query_text.strip()
    if not normalized:
        return ""

    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    normalized = re.sub(r"[^\S\r\n]+", " ", normalized)
    normalized = re.sub(r"[?!,:;()\[\]{}]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for pattern, replacement in KEYWORD_TECH_SYNONYMS:
        normalized = pattern.sub(f" {replacement} ", normalized)
    for pattern, replacement in KEYWORD_VARIANT_PATTERNS:
        normalized = pattern.sub(f" {replacement} ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _chunk_payload_from_row(row) -> dict:
    """Normalize a SQL row into the shared retrieval payload used across retrieval paths."""
    source_type_val = getattr(row, "src_type", None) or "jd"
    source_title_val = getattr(row, "src_title", None) or ""
    payload = {
        "chunk_id": str(row.id),
        "chunkId": str(row.id),
        "page_number": row.page_number,
        "page": row.page_number,
        "snippet": row.content,
        "text": row.content,
        "score": round(float(row.score), 6),
        "is_low_signal": bool(row.is_low_signal),
        "section_type": getattr(row, "section_type", None),
        "sourceType": source_type_val,
        "sourceTitle": source_title_val,
    }
    embedding = getattr(row, "embedding", None)
    if embedding is not None:
        payload["embedding"] = embedding
    content_hash = getattr(row, "content_hash", None)
    if content_hash:
        payload["content_hash"] = content_hash
    return payload


def _with_retrieval_source_defaults(candidates: list[dict], retrieval_source: str) -> list[dict]:
    """Attach transparency metadata for single-source retrieval results."""
    result = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["retrieval_source"] = retrieval_source
        enriched["retrievalSource"] = retrieval_source
        if retrieval_source == "semantic":
            enriched["semantic_score"] = enriched.get("semantic_score", enriched.get("score"))
            enriched["keyword_score"] = enriched.get("keyword_score")
        elif retrieval_source == "keyword":
            enriched["semantic_score"] = enriched.get("semantic_score")
            enriched["keyword_score"] = enriched.get("keyword_score", enriched.get("score"))
        enriched["final_score"] = enriched.get("final_score", enriched.get("score"))
        result.append(enriched)
    return result


def _finalize_chunks(candidates: list[dict]) -> list[dict]:
    """Return caller-facing chunk dicts while preserving internal debugging metadata."""
    return [
        {
            "chunk_id": c["chunk_id"],
            "chunkId": c["chunkId"],
            "page_number": c["page_number"],
            "page": c["page"],
            "snippet": c["snippet"],
            "text": c["text"],
            "score": c["score"],
            "sourceType": c["sourceType"],
            "sourceTitle": c["sourceTitle"],
            "is_low_signal": c.get("is_low_signal", False),
            "section_type": c.get("section_type"),
            "retrieval_source": c.get("retrieval_source", "semantic"),
            "retrievalSource": c.get("retrievalSource", c.get("retrieval_source", "semantic")),
            "semantic_score": c.get("semantic_score", c["score"] if c.get("retrieval_source", "semantic") == "semantic" else None),
            "keyword_score": c.get("keyword_score"),
            "final_score": c.get("final_score", c["score"]),
        }
        for c in candidates
    ]


def _normalize_scores(candidates: list[dict], score_key: str) -> None:
    """Normalize a retrieval list's scores into 0..1 so vector and FTS results can be merged."""
    positive_scores = [max(float(c.get(score_key, 0.0)), 0.0) for c in candidates]
    max_score = max(positive_scores, default=0.0)
    for c in candidates:
        raw = max(float(c.get(score_key, 0.0)), 0.0)
        c[f"{score_key}_norm"] = (raw / max_score) if max_score > 0 else 0.0


def _hybrid_score(semantic_norm: float | None, keyword_norm: float | None) -> float:
    """
    Blend semantic and keyword scores into one ranking score.

    Strategy:
    - semantic-only keeps its normalized semantic score
    - keyword-only gets a slight discount so exact-term matches complement, rather than dominate,
      vector retrieval
    - if both hit the same chunk, add a small corroboration bonus
    """
    if semantic_norm is not None and keyword_norm is not None:
        return round(
            min(1.0, max(semantic_norm, keyword_norm * 0.9) + 0.1 * min(semantic_norm, keyword_norm)),
            6,
        )
    if semantic_norm is not None:
        return round(float(semantic_norm), 6)
    if keyword_norm is not None:
        return round(float(keyword_norm) * 0.9, 6)
    return 0.0


def _merge_retrieval_candidates(
    semantic_candidates: list[dict],
    keyword_candidates: list[dict],
) -> list[dict]:
    """Merge semantic + keyword candidates and deduplicate by chunk id, falling back to content hash."""
    _normalize_scores(semantic_candidates, "score")
    for c in semantic_candidates:
        c["semantic_score"] = c["score"]
        c["semantic_score_norm"] = c.pop("score_norm")
        c["retrieval_source"] = "semantic"
        c["retrievalSource"] = "semantic"

    _normalize_scores(keyword_candidates, "score")
    for c in keyword_candidates:
        c["keyword_score"] = c["score"]
        c["keyword_score_norm"] = c.pop("score_norm")
        c["retrieval_source"] = "keyword"
        c["retrievalSource"] = "keyword"

    merged: dict[str, dict] = {}
    hash_to_chunk_id: dict[str, str] = {}

    for candidate in semantic_candidates + keyword_candidates:
        chunk_id = candidate["chunk_id"]
        content_hash = candidate.get("content_hash")
        existing_key = chunk_id
        if content_hash and content_hash in hash_to_chunk_id:
            existing_key = hash_to_chunk_id[content_hash]

        existing = merged.get(existing_key)
        if not existing:
            merged[existing_key] = dict(candidate)
            if content_hash:
                hash_to_chunk_id[content_hash] = existing_key
            continue

        if candidate.get("semantic_score") is not None:
            existing["semantic_score"] = candidate["semantic_score"]
            existing["semantic_score_norm"] = candidate.get("semantic_score_norm")
        if candidate.get("keyword_score") is not None:
            existing["keyword_score"] = candidate["keyword_score"]
            existing["keyword_score_norm"] = candidate.get("keyword_score_norm")
        if existing.get("embedding") is None and candidate.get("embedding") is not None:
            existing["embedding"] = candidate["embedding"]
        if not existing.get("content_hash") and content_hash:
            existing["content_hash"] = content_hash
            hash_to_chunk_id[content_hash] = existing_key

        existing_source = existing.get("retrieval_source")
        candidate_source = candidate.get("retrieval_source")
        if existing_source == "both" or candidate_source == "both":
            existing["retrieval_source"] = "both"
            existing["retrievalSource"] = "both"
        elif existing_source and candidate_source and existing_source != candidate_source:
            existing["retrieval_source"] = "both"
            existing["retrievalSource"] = "both"

    merged_candidates = list(merged.values())
    for c in merged_candidates:
        c["final_score"] = _hybrid_score(
            c.get("semantic_score_norm"),
            c.get("keyword_score_norm"),
        )
        c["score"] = c["final_score"]

    merged_candidates.sort(key=lambda c: c["score"], reverse=True)
    return merged_candidates


def _log_retrieval_summary(
    *,
    document_id: uuid.UUID,
    semantic_hits: int,
    keyword_hits: int,
    deduped_hits: int,
    final_hits: int,
    hybrid_enabled: bool,
) -> None:
    """Emit a compact structured log for hybrid retrieval debugging."""
    logger.info(
        "hybrid_retrieval_summary document_id=%s hybrid_enabled=%s semantic_hits=%s keyword_hits=%s deduped_hits=%s final_hits=%s",
        document_id,
        hybrid_enabled,
        semantic_hits,
        keyword_hits,
        deduped_hits,
        final_hits,
    )


def _finalize_single_source_candidates(
    candidates: list[dict],
    *,
    query_embedding: list[float] | None,
    top_k: int,
    retrieval_source: str,
) -> list[dict]:
    """Apply shared post-processing for semantic-only or keyword-only candidate lists."""
    ranked = candidates
    if query_embedding is not None and candidates:
        ranked = _mmr_select(candidates, query_embedding, top_k, settings.mmr_lambda)
    else:
        ranked = candidates[:top_k]
        for c in ranked:
            c.pop("embedding", None)
    return _finalize_chunks(_with_retrieval_source_defaults(ranked, retrieval_source))


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
            if selected and c.get("embedding") is not None:
                for s in selected:
                    if s.get("embedding") is None:
                        continue
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


async def _retrieve_semantic_candidates(
    db: AsyncSession,
    document_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int,
    include_low_signal: bool = False,
    section_types: list[str] | None = None,
    doc_domain: str | None = None,
    source_types: list[str] | None = None,
    additional_document_ids: list[uuid.UUID] | None = None,
) -> list[dict]:
    """Fetch semantic candidates before final ranking/diversification."""
    distance_col = DocumentChunk.embedding.cosine_distance(query_embedding)
    score_col = (1 - distance_col).label("score")
    limit = max(top_k, settings.top_n_candidates)

    expanded_section_types = _expanded_section_types(section_types)

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.page_number,
            DocumentChunk.content,
            DocumentChunk.embedding,
            DocumentChunk.content_hash,
            DocumentChunk.is_low_signal,
            DocumentChunk.section_type,
            InterviewSource.source_type.label("src_type"),
            InterviewSource.title.label("src_title"),
            score_col,
        )
        .join(InterviewSource, DocumentChunk.source_id == InterviewSource.id)
        .where(
            DocumentChunk.document_id.in_([document_id] + (additional_document_ids or []))
        )
        .where(DocumentChunk.embedding.isnot(None))
        .order_by(distance_col.asc())
        .limit(limit)
    )
    if not include_low_signal:
        stmt = stmt.where(DocumentChunk.is_low_signal.is_(False))
    if expanded_section_types:
        stmt = stmt.where(DocumentChunk.section_type.in_(expanded_section_types))
    if doc_domain:
        stmt = stmt.where(DocumentChunk.doc_domain == doc_domain)
    if source_types:
        stmt = stmt.where(InterviewSource.source_type.in_(source_types))

    result = await db.execute(stmt)
    rows = result.all()
    return [_chunk_payload_from_row(row) for row in rows]


def get_default_retrieval_mode() -> RetrievalMode:
    """Return the retrieval mode implied by current app settings."""
    return "hybrid" if settings.hybrid_retrieval_enabled else "semantic"


async def retrieve_chunks_keyword(
    db: AsyncSession,
    document_id: uuid.UUID,
    query_text: str,
    top_k: int,
    include_low_signal: bool = False,
    section_types: list[str] | None = None,
    doc_domain: str | None = None,
    source_types: list[str] | None = None,
    additional_document_ids: list[uuid.UUID] | None = None,
) -> list[dict]:
    """
    Search document_chunks by PostgreSQL full-text search ranking.

    This is the keyword retrieval path that complements vector search:
    - converts the raw query text into a PostgreSQL tsquery
    - filters the same chunk corpus as semantic retrieval
    - ranks matches with ts_rank_cd(search_vector, tsquery)

    Returns the same chunk payload shape as retrieve_chunks() so callers can
    later merge keyword and semantic candidates without extra mapping.
    """
    normalized_query = _normalize_keyword_query_text(query_text)
    if not normalized_query:
        return []

    expanded_section_types = _expanded_section_types(section_types)

    # websearch_to_tsquery is more forgiving for natural user input than
    # plainto_tsquery, and the preprocessor above keeps JD/technical tokens
    # readable while expanding a few high-value keyword variants.
    tsquery = func.websearch_to_tsquery("english", normalized_query)
    rank_col = func.ts_rank_cd(DocumentChunk.search_vector, tsquery).label("score")

    stmt = (
        select(
            DocumentChunk.id,
            DocumentChunk.page_number,
            DocumentChunk.content,
            DocumentChunk.embedding,
            DocumentChunk.content_hash,
            DocumentChunk.is_low_signal,
            DocumentChunk.section_type,
            InterviewSource.source_type.label("src_type"),
            InterviewSource.title.label("src_title"),
            rank_col,
        )
        .join(InterviewSource, DocumentChunk.source_id == InterviewSource.id)
        .where(
            DocumentChunk.document_id.in_([document_id] + (additional_document_ids or []))
        )
        .where(DocumentChunk.search_vector.op("@@")(tsquery))
        .order_by(rank_col.desc(), DocumentChunk.page_number.asc(), DocumentChunk.chunk_index.asc())
        .limit(top_k)
    )
    if not include_low_signal:
        stmt = stmt.where(DocumentChunk.is_low_signal.is_(False))
    if expanded_section_types:
        stmt = stmt.where(DocumentChunk.section_type.in_(expanded_section_types))
    if doc_domain:
        stmt = stmt.where(DocumentChunk.doc_domain == doc_domain)
    if source_types:
        stmt = stmt.where(InterviewSource.source_type.in_(source_types))

    result = await db.execute(stmt)
    rows = result.all()
    return [_chunk_payload_from_row(row) for row in rows]


async def retrieve_chunks_for_mode(
    db: AsyncSession,
    document_id: uuid.UUID,
    query_embedding: list[float] | None,
    top_k: int,
    mode: RetrievalMode = "hybrid",
    include_low_signal: bool = False,
    section_types: list[str] | None = None,
    doc_domain: str | None = None,
    source_types: list[str] | None = None,
    additional_document_ids: list[uuid.UUID] | None = None,
    query_text: str | None = None,
) -> list[dict]:
    """
    Shared retrieval entry point used by eval tooling and production callers.

    Behavior:
    - `semantic`: vector retrieval + MMR
    - `keyword`: PostgreSQL full-text retrieval (+ MMR when a query embedding is provided)
    - `hybrid`: semantic retrieval plus keyword augmentation and merge
    """
    normalized_query_text = (query_text or "").strip()

    if mode not in ("hybrid", "semantic", "keyword"):
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    if mode in ("hybrid", "semantic") and query_embedding is None:
        raise ValueError(f"query_embedding is required for retrieval mode '{mode}'")

    if mode == "semantic":
        semantic_candidates = await _retrieve_semantic_candidates(
            db=db,
            document_id=document_id,
            query_embedding=query_embedding,
            top_k=top_k,
            include_low_signal=include_low_signal,
            section_types=section_types,
            doc_domain=doc_domain,
            source_types=source_types,
            additional_document_ids=additional_document_ids,
        )
        final_candidates = _with_retrieval_source_defaults(
            _mmr_select(semantic_candidates, query_embedding, top_k, settings.mmr_lambda)
            if semantic_candidates else [],
            "semantic",
        )
        _log_retrieval_summary(
            document_id=document_id,
            semantic_hits=len(semantic_candidates),
            keyword_hits=0,
            deduped_hits=len(semantic_candidates),
            final_hits=len(final_candidates),
            hybrid_enabled=False,
        )
        return _finalize_chunks(final_candidates)

    if mode == "keyword":
        if not normalized_query_text:
            _log_retrieval_summary(
                document_id=document_id,
                semantic_hits=0,
                keyword_hits=0,
                deduped_hits=0,
                final_hits=0,
                hybrid_enabled=False,
            )
            return []
        keyword_candidates = await retrieve_chunks_keyword(
            db=db,
            document_id=document_id,
            query_text=normalized_query_text,
            top_k=max(top_k, settings.top_n_candidates),
            include_low_signal=include_low_signal,
            section_types=section_types,
            doc_domain=doc_domain,
            source_types=source_types,
            additional_document_ids=additional_document_ids,
        )
        final_candidates = _finalize_single_source_candidates(
            keyword_candidates,
            query_embedding=query_embedding,
            top_k=top_k,
            retrieval_source="keyword",
        )
        _log_retrieval_summary(
            document_id=document_id,
            semantic_hits=0,
            keyword_hits=len(keyword_candidates),
            deduped_hits=len(keyword_candidates),
            final_hits=len(final_candidates),
            hybrid_enabled=False,
        )
        return final_candidates

    semantic_candidates = await _retrieve_semantic_candidates(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        top_k=top_k,
        include_low_signal=include_low_signal,
        section_types=section_types,
        doc_domain=doc_domain,
        source_types=source_types,
        additional_document_ids=additional_document_ids,
    )

    if not normalized_query_text:
        final_candidates = _with_retrieval_source_defaults(
            _mmr_select(semantic_candidates, query_embedding, top_k, settings.mmr_lambda)
            if semantic_candidates else [],
            "semantic",
        )
        _log_retrieval_summary(
            document_id=document_id,
            semantic_hits=len(semantic_candidates),
            keyword_hits=0,
            deduped_hits=len(semantic_candidates),
            final_hits=len(final_candidates),
            hybrid_enabled=True,
        )
        return _finalize_chunks(final_candidates)

    try:
        keyword_candidates = await retrieve_chunks_keyword(
            db=db,
            document_id=document_id,
            query_text=normalized_query_text,
            top_k=max(top_k, settings.top_n_candidates),
            include_low_signal=include_low_signal,
            section_types=section_types,
            doc_domain=doc_domain,
            source_types=source_types,
            additional_document_ids=additional_document_ids,
        )
    except Exception as exc:
        logger.warning("Keyword retrieval failed; falling back to semantic-only: %s", exc)
        final_candidates = _with_retrieval_source_defaults(
            _mmr_select(semantic_candidates, query_embedding, top_k, settings.mmr_lambda)
            if semantic_candidates else [],
            "semantic",
        )
        _log_retrieval_summary(
            document_id=document_id,
            semantic_hits=len(semantic_candidates),
            keyword_hits=0,
            deduped_hits=len(semantic_candidates),
            final_hits=len(final_candidates),
            hybrid_enabled=True,
        )
        return _finalize_chunks(final_candidates)

    if not keyword_candidates:
        final_candidates = _with_retrieval_source_defaults(
            _mmr_select(semantic_candidates, query_embedding, top_k, settings.mmr_lambda)
            if semantic_candidates else [],
            "semantic",
        )
        _log_retrieval_summary(
            document_id=document_id,
            semantic_hits=len(semantic_candidates),
            keyword_hits=0,
            deduped_hits=len(semantic_candidates),
            final_hits=len(final_candidates),
            hybrid_enabled=True,
        )
        return _finalize_chunks(final_candidates)

    hybrid_candidates = _merge_retrieval_candidates(semantic_candidates, keyword_candidates)
    diversified = _mmr_select(
        hybrid_candidates,
        query_embedding,
        top_k,
        settings.mmr_lambda,
    )
    _log_retrieval_summary(
        document_id=document_id,
        semantic_hits=len(semantic_candidates),
        keyword_hits=len(keyword_candidates),
        deduped_hits=len(hybrid_candidates),
        final_hits=len(diversified),
        hybrid_enabled=True,
    )
    return _finalize_chunks(diversified)


async def retrieve_chunks(
    db: AsyncSession,
    document_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int,
    include_low_signal: bool = False,
    section_types: list[str] | None = None,
    doc_domain: str | None = None,
    source_types: list[str] | None = None,
    additional_document_ids: list[uuid.UUID] | None = None,
    query_text: str | None = None,
) -> list[dict]:
    """
    Shared production retrieval entry point used by ask/retrieve/interview flows.

    The retrieval mode remains controlled by app settings so existing routes keep
    their current behavior. Eval tooling can call `retrieve_chunks_for_mode()`
    directly to compare semantic, keyword, and hybrid strategies.
    """
    return await retrieve_chunks_for_mode(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        top_k=top_k,
        mode=get_default_retrieval_mode(),
        include_low_signal=include_low_signal,
        section_types=section_types,
        doc_domain=doc_domain,
        source_types=source_types,
        additional_document_ids=additional_document_ids,
        query_text=query_text,
    )
