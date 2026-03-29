"""Grounded Q&A: retrieval + structured recruiter-style JSON reasoning."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import Document, User
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
    document_id: uuid.UUID
    question: str = Field(..., min_length=1)
    additional_document_ids: list[str] | None = Field(
        default=None,
        description="Optional extra documents (e.g. resume) to search with the primary document.",
    )


class Citation(BaseModel):
    chunk_id: str
    page_number: int
    snippet: str


class AskOutput(BaseModel):
    answer: str = Field(
        ...,
        description="JSON string: key_job_requirements, matches, gaps, fit_score, reasoning.",
    )
    citations: list[Citation]


@router.post("", response_model=AskOutput)
async def ask(
    body: AskInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Fit-oriented Q&A over retrieved chunks (JD vs resume when ``source_type`` is set).
    Returns ``answer`` as strict JSON text and parallel chunk citations for grounding.
    """
    result = await db.execute(select(Document).where(Document.id == body.document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    assert_resource_ownership(doc, current_user)

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

    additional_uuid_list: list[uuid.UUID] = []
    raw_additional = body.additional_document_ids or []
    seen: set[uuid.UUID] = set()
    for raw_id in raw_additional:
        try:
            parsed = uuid.UUID(str(raw_id).strip())
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid additional_document_ids entry: {raw_id!r}",
            ) from None
        if parsed == body.document_id or parsed in seen:
            continue
        seen.add(parsed)
        additional_uuid_list.append(parsed)

    if additional_uuid_list:
        result = await db.execute(
            select(Document).where(Document.id.in_(additional_uuid_list))
        )
        found = {row.id: row for row in result.scalars().all()}
        missing = [str(i) for i in additional_uuid_list if i not in found]
        if missing:
            raise HTTPException(
                status_code=404,
                detail=f"Document(s) not found: {', '.join(missing)}",
            )
        for extra_id in additional_uuid_list:
            extra_doc = found[extra_id]
            assert_resource_ownership(extra_doc, current_user)
            if extra_doc.status != "ready":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Document {extra_id} must be ready to answer; "
                        f"current status: {extra_doc.status}"
                    ),
                )

    additional_for_retrieval = additional_uuid_list or None

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
            additional_document_ids=additional_for_retrieval,
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
                additional_document_ids=additional_for_retrieval,
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
