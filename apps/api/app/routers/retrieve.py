"""Retrieve: hybrid search over document chunks."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import Document, User
from app.services.retrieval import (
    embed_query,
    retrieve_chunks,
    suggest_section_filters,
)

router = APIRouter(prefix="/retrieve", tags=["retrieve"])
logger = logging.getLogger(__name__)


class RetrieveInput(BaseModel):
    document_id: uuid.UUID
    query: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=8)
    source_types: list[str] | None = Field(
        None,
        description="Filter to these source types (jd, resume, company, notes). Default: all sources.",
    )
    include_low_signal: bool = Field(
        False,
        description="If true, include contact/boilerplate chunks (for queries like 'what is their email?')",
    )
    section_types: list[str] | None = Field(
        None,
        description="Filter to these job description section types (e.g. qualifications, compensation)",
    )
    doc_domain: str | None = Field(
        None,
        description="Filter by doc_domain (e.g. job_description)",
    )


class RetrievedChunk(BaseModel):
    """Metadata-rich citation: text, score, sourceType, sourceTitle, page (optional), chunkId."""

    text: str
    score: float
    source_type: str = Field(..., alias="sourceType")
    source_title: str = Field(..., alias="sourceTitle")
    page: int | None = None
    chunk_id: str = Field(..., alias="chunkId")
    is_low_signal: bool = False
    section_type: str | None = None

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


class RetrieveOutput(BaseModel):
    chunks: list[RetrievedChunk]


@router.post("", response_model=RetrieveOutput)
async def retrieve(
    body: RetrieveInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Hybrid retrieval over document chunks.
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

    result = await db.execute(select(Document).where(Document.id == body.document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    assert_resource_ownership(doc, current_user)

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
    if section_types is None and doc.doc_domain == "job_description":
        section_types = suggest_section_filters(body.query)
    if doc_domain is None and doc.doc_domain:
        doc_domain = doc.doc_domain
    try:
        chunks = await retrieve_chunks(
            db=db,
            document_id=body.document_id,
            query_embedding=query_embedding,
            query_text=body.query,
            top_k=body.top_k,
            include_low_signal=body.include_low_signal,
            section_types=section_types,
            doc_domain=doc_domain,
            source_types=body.source_types,
        )
    except Exception as e:
        logger.exception("retrieve_chunks failed")
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {str(e)[:200]}")

    return RetrieveOutput(
        chunks=[
            RetrievedChunk(
                text=c["text"],
                score=c["score"],
                sourceType=c["sourceType"],
                sourceTitle=c["sourceTitle"],
                page=c.get("page"),
                chunkId=c["chunkId"],
                is_low_signal=c.get("is_low_signal", False),
                section_type=c.get("section_type"),
            )
            for c in chunks
        ]
    )
