"""Cache helpers for POST /analyze-fit.

Lookup key: user_id + job_description_id + resume_id + normalized question fingerprint
+ aggregate fingerprints of each document's chunks (invalidates when ingestion changes).

Cache hits skip embedding, retrieval, and the LLM call.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_chunk import DocumentChunk
from app.models.fit_analysis import FitAnalysis

DEFAULT_ANALYZE_FIT_QUESTION = (
    "Compare job requirements, qualifications, skills, and responsibilities "
    "to the candidate's resume experience and evidence."
)


def normalize_analyze_fit_question(question: str | None) -> str:
    text = (question or "").strip()
    return text if text else DEFAULT_ANALYZE_FIT_QUESTION


def analyze_fit_query_fingerprint(normalized_question: str) -> str:
    return hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()


def _digest_chunk_rows(rows: Sequence[tuple[int, str | None, str]]) -> str:
    parts: list[str] = []
    for chunk_index, content_hash, content in rows:
        piece = (content_hash or "").strip() or hashlib.sha256(content.encode("utf-8")).hexdigest()
        parts.append(f"{chunk_index}:{piece}")
    joined = "|".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


async def document_chunk_fingerprints(
    db: AsyncSession,
    job_description_id: uuid.UUID,
    resume_id: uuid.UUID,
) -> tuple[str, str]:
    """
    Stable fingerprint per document from ordered chunks (content_hash when set, else SHA-256 of content).
    Empty chunk sets yield a fixed digest of the empty serialization.
    """
    stmt = (
        select(
            DocumentChunk.document_id,
            DocumentChunk.chunk_index,
            DocumentChunk.content_hash,
            DocumentChunk.content,
        )
        .where(DocumentChunk.document_id.in_((job_description_id, resume_id)))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )
    result = await db.execute(stmt)
    jd_rows: list[tuple[int, str | None, str]] = []
    rs_rows: list[tuple[int, str | None, str]] = []
    for doc_id, chunk_index, ch, content in result.all():
        row = (int(chunk_index), ch, str(content or ""))
        if doc_id == job_description_id:
            jd_rows.append(row)
        elif doc_id == resume_id:
            rs_rows.append(row)
    return _digest_chunk_rows(jd_rows), _digest_chunk_rows(rs_rows)


async def fetch_latest_cached_fit_analysis(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    job_description_id: uuid.UUID,
    resume_id: uuid.UUID,
    query_fingerprint: str,
    jd_chunk_fingerprint: str,
    resume_chunk_fingerprint: str,
) -> FitAnalysis | None:
    stmt = (
        select(FitAnalysis)
        .where(
            FitAnalysis.user_id == user_id,
            FitAnalysis.job_description_id == job_description_id,
            FitAnalysis.resume_id == resume_id,
            FitAnalysis.query_fingerprint == query_fingerprint,
            FitAnalysis.jd_chunk_fingerprint == jd_chunk_fingerprint,
            FitAnalysis.resume_chunk_fingerprint == resume_chunk_fingerprint,
        )
        .order_by(FitAnalysis.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    return result.scalar_one_or_none()
