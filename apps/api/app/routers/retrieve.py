"""Retrieve: semantic search over document chunks."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models import Document
from app.services.retrieval import (
    embed_query,
    retrieve_chunks,
    suggest_section_filters,
)

router = APIRouter(prefix="/retrieve", tags=["retrieve"])
logger = logging.getLogger(__name__)


class RetrieveInput(BaseModel):
    user_id: uuid.UUID
    document_id: uuid.UUID
    query: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=8)
    include_low_signal: bool = Field(
        False,
        description="If true, include contact/boilerplate chunks (for queries like 'what is their email?')",
    )
    section_types: list[str] | None = Field(
        None,
        description="Filter to these JD section types (e.g. qualifications, compensation)",
    )
    doc_domain: str | None = Field(
        None,
        description="Filter by doc_domain (e.g. job_description)",
    )


class RetrievedChunk(BaseModel):
    chunk_id: str
    page_number: int
    snippet: str
    score: float
    is_low_signal: bool = False
    section_type: str | None = None


class RetrieveOutput(BaseModel):
    chunks: list[RetrievedChunk]


@router.post("", response_model=RetrieveOutput)
async def retrieve(
    body: RetrieveInput,
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic search over document chunks.
    Validates: top_k <= TOP_K_MAX, doc ownership, status=ready.
    """
    if body.top_k > settings.top_k_max:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "top_k exceeds limit",
                "top_k": body.top_k,
                "max": settings.top_k_max,
            },
        )

    result = await db.execute(
        select(Document).where(
            Document.id == body.document_id,
            Document.user_id == body.user_id,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document must be ready to retrieve; current status: {doc.status}",
        )

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured; set OPENAI_API_KEY",
        )

    try:
        query_embedding = embed_query(body.query)
    except Exception as e:
        logger.exception("embed_query failed")
        raise HTTPException(status_code=503, detail=f"Embedding failed: {str(e)[:200]}")

    section_types = body.section_types
    doc_domain = body.doc_domain
    if section_types is None and doc.jd_extraction_json:
        section_types = suggest_section_filters(body.query)
    if doc_domain is None and doc.jd_extraction_json:
        doc_domain = "job_description"
    try:
        chunks = await retrieve_chunks(
            db=db,
            document_id=body.document_id,
            query_embedding=query_embedding,
            top_k=body.top_k,
            include_low_signal=body.include_low_signal,
            section_types=section_types,
            doc_domain=doc_domain,
        )
    except Exception as e:
        logger.exception("retrieve_chunks failed")
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {str(e)[:200]}")

    return RetrieveOutput(chunks=[RetrievedChunk(**c) for c in chunks])
