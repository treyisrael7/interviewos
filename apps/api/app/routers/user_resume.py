"""User-level resume: one resume per account, used across all job descriptions."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.db.session import get_db
from app.models import Document, InterviewSource, User
from app.services.source_ingestion import ingest_resume_pdf
from app.services.storage import get_storage

router = APIRouter(prefix="/user", tags=["user"])

USER_RESUME_DOC_DOMAIN = "user_resume"
USER_RESUME_FILENAME = "Resume.pdf"


def _user_resume_s3_key(user_id: uuid.UUID) -> str:
    return f"users/{user_id}/resume.pdf"


def _validate_pdf_size(file_size_bytes: int):
    from app.core.config import settings
    mb = file_size_bytes / (1024 * 1024)
    if mb > settings.max_pdf_mb:
        raise HTTPException(
            status_code=400,
            detail={"error": "PDF too large", "max_mb": settings.max_pdf_mb, "received_mb": round(mb, 2)},
        )


class PresignInput(BaseModel):
    filename: str = Field(..., min_length=1)
    file_size_bytes: int = Field(..., gt=0)


class PresignOutput(BaseModel):
    s3_key: str
    upload_url: str
    method: str


class ConfirmInput(BaseModel):
    s3_key: str


@router.get("/resume")
async def get_resume_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check if user has an account-level resume."""
    result = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            Document.doc_domain == USER_RESUME_DOC_DOMAIN,
        )
    )
    doc = result.scalar_one_or_none()
    return {"has_resume": doc is not None, "filename": doc.filename if doc else None}


@router.post("/resume/presign", response_model=PresignOutput)
async def presign_user_resume(
    body: PresignInput,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """Get presigned URL for account resume upload."""
    _validate_pdf_size(body.file_size_bytes)

    if not body.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    s3_key = _user_resume_s3_key(current_user.id)
    storage = get_storage()
    upload_url, method = storage.generate_presigned_put(
        key=s3_key,
        content_type="application/pdf",
    )
    if upload_url.startswith("/"):
        base = str(request.base_url).rstrip("/")
        upload_url = f"{base}{upload_url}"

    return PresignOutput(s3_key=s3_key, upload_url=upload_url, method=method)


@router.post("/resume")
async def confirm_user_resume(
    body: ConfirmInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm resume upload and ingest. Creates or replaces user's account resume."""
    expected_key = _user_resume_s3_key(current_user.id)
    if body.s3_key != expected_key:
        raise HTTPException(status_code=400, detail="Invalid s3_key for user resume")

    storage = get_storage()
    if not storage.exists(body.s3_key):
        raise HTTPException(status_code=400, detail="File not found in storage; upload failed")

    result = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            Document.doc_domain == USER_RESUME_DOC_DOMAIN,
        )
    )
    doc = result.scalar_one_or_none()

    if doc:
        await db.execute(delete(InterviewSource).where(InterviewSource.document_id == doc.id))
        await db.commit()
        document_id = doc.id
    else:
        doc = Document(
            user_id=current_user.id,
            filename=USER_RESUME_FILENAME,
            s3_key=body.s3_key,
            status="uploaded",
            doc_domain=USER_RESUME_DOC_DOMAIN,
        )
        db.add(doc)
        await db.flush()
        document_id = doc.id

    try:
        source_id = await ingest_resume_pdf(
            db=db,
            document_id=document_id,
            s3_key=body.s3_key,
            original_filename=USER_RESUME_FILENAME,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc.status = "ready"
    await db.commit()
    return {"source_id": source_id, "status": "ingested", "document_id": str(document_id)}


@router.delete("/resume")
async def delete_user_resume(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete user's account resume."""
    result = await db.execute(
        select(Document).where(
            Document.user_id == current_user.id,
            Document.doc_domain == USER_RESUME_DOC_DOMAIN,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return {"status": "deleted", "message": "No resume found"}

    await db.delete(doc)
    await db.commit()
    return {"status": "deleted"}
