"""Grounded Q&A: retrieval + LLM with citation markers."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_user_id_from_bearer
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models import Document
from app.services.qa import generate_grounded_answer
from app.services.retrieval import (
    embed_query,
    retrieve_chunks,
    suggest_section_filters,
)

router = APIRouter(prefix="/ask", tags=["ask"])
logger = logging.getLogger(__name__)

ASK_TOP_K = 6


class AskInput(BaseModel):
    user_id: uuid.UUID | None = None
    document_id: uuid.UUID
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    chunk_id: str
    page_number: int
    snippet: str


class AskOutput(BaseModel):
    answer: str
    citations: list[Citation]


@router.post("", response_model=AskOutput)
async def ask(
    body: AskInput,
    user_id_from_auth: uuid.UUID | None = Depends(get_user_id_from_bearer),
    db: AsyncSession = Depends(get_db),
):
    """
    Grounded Q&A over document chunks.
    Retrieves relevant excerpts, builds a grounded prompt, calls OpenAI chat completion.
    Returns answer with citation markers [pN-cM] and a citations list.
    """
    uid = user_id_from_auth or body.user_id
    if uid is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Validate document
    result = await db.execute(
        select(Document).where(
            Document.id == body.document_id,
            Document.user_id == uid,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document must be ready to answer; current status: {doc.status}",
        )

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured; set OPENAI_API_KEY",
        )

    # Retrieve relevant chunks
    try:
        query_embedding = embed_query(body.question)
    except Exception as e:
        logger.exception("embed_query failed")
        raise HTTPException(status_code=503, detail=f"Embedding failed: {str(e)[:200]}")

    section_types = None
    doc_domain = doc.doc_domain or None
    if doc.doc_domain == "job_description":
        section_types = suggest_section_filters(body.question)

    try:
        chunks = await retrieve_chunks(
            db=db,
            document_id=body.document_id,
            query_embedding=query_embedding,
            query_text=body.question,
            top_k=min(ASK_TOP_K, settings.top_k_max),
            include_low_signal=False,
            section_types=section_types,
            doc_domain=doc_domain,
        )
        # If section filter (e.g. compensation) returned nothing, retry without filter
        if not chunks and section_types:
            chunks = await retrieve_chunks(
                db=db,
                document_id=body.document_id,
                query_embedding=query_embedding,
                query_text=body.question,
                top_k=min(ASK_TOP_K, settings.top_k_max),
                include_low_signal=False,
                section_types=None,
                doc_domain=doc_domain,
            )
    except Exception as e:
        logger.exception("retrieve_chunks failed")
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {str(e)[:200]}")

    # Generate grounded answer (or fallback if no chunks)
    try:
        answer, citations = generate_grounded_answer(
            question=body.question,
            chunks=chunks,
            max_tokens=settings.max_completion_tokens,
        )
    except Exception as e:
        logger.exception("generate_grounded_answer failed")
        raise HTTPException(status_code=503, detail=f"Q&A failed: {str(e)[:200]}")

    return AskOutput(
        answer=answer,
        citations=[Citation(**c) for c in citations],
    )
