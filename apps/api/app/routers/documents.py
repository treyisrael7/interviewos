import re
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import Document, DocumentChunk, InterviewSource, User
from app.services.gap_analysis import generate_gap_analysis
from app.services.ingestion import run_ingestion
from app.services.storage import get_storage

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPE = "application/pdf"


class RoleProfileOut(BaseModel):
    domain: str
    seniority: str
    roleTitleGuess: str = ""
    focusAreas: list[str] = []
    questionMix: dict = {}


class CompetencyWithCoverage(BaseModel):
    id: str
    label: str
    description: str | None = None
    attempts_count: int = 0
    avg_score: float | None = None


class DocumentSummary(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int | None
    error_message: str | None
    created_at: str
    doc_domain: str = "general"
    role_profile: RoleProfileOut | None = None
    competencies: list[CompetencyWithCoverage] = []
    coverage_practiced: int = 0
    coverage_total: int = 0


def _to_role_profile_out(rp: dict | None) -> RoleProfileOut | None:
    """Convert DB role_profile dict to RoleProfileOut, or None if missing."""
    if not rp or not isinstance(rp, dict):
        return None
    try:
        return RoleProfileOut(
            domain=rp.get("domain", "general_business"),
            seniority=rp.get("seniority", "entry"),
            roleTitleGuess=rp.get("roleTitleGuess", ""),
            focusAreas=rp.get("focusAreas") or [],
            questionMix=rp.get("questionMix") or {},
        )
    except Exception:
        return None


async def _get_coverage_by_document(
    db: AsyncSession, document_ids: list[uuid.UUID]
) -> dict[uuid.UUID, dict[str, dict]]:
    """Returns {doc_id: {competency_id: {attempts_count, avg_score}}} from interview answers."""
    if not document_ids:
        return {}
    stmt = text("""
        SELECT
            s.document_id,
            iq.rubric_json->>'competency_id' AS competency_id,
            COUNT(*)::int AS attempts_count,
            AVG(ia.score)::float AS avg_score
        FROM interview_answers ia
        JOIN interview_questions iq ON ia.question_id = iq.id
        JOIN interview_sessions s ON iq.session_id = s.id
        WHERE s.document_id = ANY(:doc_ids)
          AND iq.rubric_json->>'competency_id' IS NOT NULL
          AND iq.rubric_json->>'competency_id' != ''
        GROUP BY s.document_id, iq.rubric_json->>'competency_id'
    """)
    result = await db.execute(stmt, {"doc_ids": list(document_ids)})
    rows = result.fetchall()
    out: dict[uuid.UUID, dict[str, dict]] = {did: {} for did in document_ids}
    for r in rows:
        doc_id, cid, cnt, avg = r[0], r[1], r[2], r[3]
        if doc_id and cid:
            out.setdefault(doc_id, {})[str(cid)] = {
                "attempts_count": cnt or 0,
                "avg_score": float(avg) if avg is not None else None,
            }
    return out


def _build_competencies_with_coverage(
    competencies: list | None,
    coverage: dict[str, dict],
    max_chips: int = 8,
) -> tuple[list[CompetencyWithCoverage], int, int]:
    """Build competency list with coverage; return (comps for display, practiced, total)."""
    comps = competencies if isinstance(competencies, list) else []
    total = sum(1 for c in comps if isinstance(c, dict) and str(c.get("label", "")).strip())
    practiced = sum(
        1 for c in comps
        if isinstance(c, dict) and coverage.get(str(c.get("id", "")), {}).get("attempts_count", 0) > 0
    )
    if not comps:
        return [], 0, 0

    result = []
    for c in comps[:max_chips]:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id", ""))
        label = str(c.get("label", "")).strip()
        if not label:
            continue
        stats = coverage.get(cid, {})
        attempts = stats.get("attempts_count", 0) or 0
        avg = stats.get("avg_score")
        result.append(
            CompetencyWithCoverage(
                id=cid or f"comp-{len(result)}",
                label=label,
                description=c.get("description"),
                attempts_count=attempts,
                avg_score=round(avg, 1) if avg is not None else None,
            )
        )

    return result, practiced, total


def _to_document_summary(d: Document, coverage: dict[str, dict]) -> DocumentSummary:
    competencies_raw = getattr(d, "competencies", None)
    comps, practiced, total = _build_competencies_with_coverage(
        competencies_raw, coverage or {}, max_chips=8
    )
    return DocumentSummary(
        id=str(d.id),
        filename=d.filename,
        status=d.status,
        page_count=d.page_count,
        error_message=d.error_message,
        created_at=d.created_at.isoformat() if d.created_at else "",
        doc_domain=getattr(d, "doc_domain", None) or "general",
        role_profile=_to_role_profile_out(getattr(d, "role_profile", None)),
        competencies=comps,
        coverage_practiced=practiced,
        coverage_total=total,
    )


async def _get_document_for_user(
    db: AsyncSession,
    document_id: uuid.UUID,
    current_user: User,
) -> Document:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    assert_resource_ownership(doc, current_user)
    return doc


@router.get("", response_model=list[DocumentSummary])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List documents for a user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    doc_ids = [d.id for d in docs]
    coverage_map = await _get_coverage_by_document(db, doc_ids)
    return [
        _to_document_summary(d, coverage_map.get(d.id, {}))
        for d in docs
    ]


@router.delete("", response_model=dict)
async def delete_all_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all documents for the current user. Removes files from storage."""
    result = await db.execute(
        select(Document).where(Document.user_id == current_user.id)
    )
    docs = result.scalars().all()
    storage = get_storage()
    for doc in docs:
        if doc.s3_key:
            try:
                storage.delete(doc.s3_key)
            except Exception:
                pass
        await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "count": len(docs)}


@router.delete("/{document_id}")
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a document and its chunks. Removes file from storage if present.
    Interview sessions linked to this document are cascade-deleted.
    """
    doc = await _get_document_for_user(db, document_id, current_user)

    s3_key = doc.s3_key
    await db.delete(doc)
    await db.commit()

    if s3_key:
        try:
            storage = get_storage()
            storage.delete(s3_key)
        except Exception:
            pass

    return {"status": "deleted", "document_id": str(document_id)}


@router.get("/{document_id}", response_model=DocumentSummary)
async def get_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get single document for status polling."""
    doc = await _get_document_for_user(db, document_id, current_user)
    coverage_map = await _get_coverage_by_document(db, [doc.id])
    return _to_document_summary(doc, coverage_map.get(doc.id, {}))


class PresignInput(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str = Field(..., pattern=r"^application/pdf$")
    file_size_bytes: int = Field(..., gt=0)


class PresignOutput(BaseModel):
    document_id: uuid.UUID
    s3_key: str
    upload_url: str
    method: str = "PUT"


class ConfirmInput(BaseModel):
    document_id: uuid.UUID
    s3_key: str = Field(..., min_length=1)


def _mb_from_bytes(b: int) -> float:
    return b / (1024 * 1024)


def _validate_pdf_size(file_size_bytes: int) -> None:
    mb = _mb_from_bytes(file_size_bytes)
    if mb > settings.max_pdf_mb:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "PDF too large",
                "max_mb": settings.max_pdf_mb,
                "received_mb": round(mb, 2),
            },
        )


def _make_s3_key(user_id: uuid.UUID, document_id: uuid.UUID, filename: str) -> str:
    safe_name = re.sub(r"[^\w\.\-]", "_", filename)
    return f"documents/{user_id}/{document_id}/{safe_name}"


@router.post("/presign", response_model=PresignOutput)
async def presign(
    body: PresignInput,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get presigned PUT URL for PDF upload. Rate limit: 10/day."""
    _validate_pdf_size(body.file_size_bytes)

    doc = Document(
        user_id=current_user.id,
        filename=body.filename,
        s3_key="",  # Set below
        status="pending",
    )
    db.add(doc)
    await db.flush()

    s3_key = _make_s3_key(current_user.id, doc.id, body.filename)
    doc.s3_key = s3_key

    storage = get_storage()
    upload_url, method = storage.generate_presigned_put(
        key=s3_key,
        content_type=body.content_type,
    )

    # For local storage, upload_url is relative; prepend base URL
    if upload_url.startswith("/"):
        base = str(request.base_url).rstrip("/")
        upload_url = f"{base}{upload_url}"

    return PresignOutput(
        document_id=doc.id,
        s3_key=s3_key,
        upload_url=upload_url,
        method=method,
    )


@router.post("/confirm")
async def confirm(
    body: ConfirmInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Verify document exists in storage and set status=uploaded."""
    doc = await _get_document_for_user(db, body.document_id, current_user)
    if doc.s3_key != body.s3_key:
        raise HTTPException(status_code=400, detail="Invalid s3_key for document")

    storage = get_storage()
    if not storage.exists(body.s3_key):
        raise HTTPException(
            status_code=400,
            detail="File not found in storage; upload may have failed",
        )

    doc.status = "uploaded"
    return {"status": "uploaded", "document_id": str(doc.id)}


@router.put("/upload-local")
async def upload_local(key: str, request: Request):
    """Local dev only: receive PUT file. Used when S3 is not configured."""
    from pathlib import Path

    from app.services.storage import get_storage, LocalStorage

    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        raise HTTPException(status_code=400, detail="Local upload not available; use S3")

    path = Path(storage.get_path(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    body = await request.body()
    path.write_bytes(body)
    return PlainTextResponse("OK", status_code=200)


class IngestInput(BaseModel):
    pass


@router.post("/{document_id}/ingest")
async def ingest(
    document_id: uuid.UUID,
    body: IngestInput,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Start document ingestion. Rate limit: 3/day.
    Checks: doc ownership, status must be uploaded.
    Sets status=processing and runs ingestion in background.
    """
    doc = await _get_document_for_user(db, document_id, current_user)

    if doc.status != "uploaded":
        raise HTTPException(
            status_code=400,
            detail=f"Document must be uploaded to ingest; current status: {doc.status}",
        )

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured; set OPENAI_API_KEY",
        )

    doc.status = "processing"
    doc.error_message = None
    await db.commit()

    background_tasks.add_task(run_ingestion, document_id)

    return {"status": "processing", "document_id": str(document_id)}


@router.post("/{document_id}/reingest")
async def reingest(
    document_id: uuid.UUID,
    body: IngestInput,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Re-run ingestion for an existing document (dev utility).
    Deletes existing chunks, resets status, runs ingestion.
    Doc must be uploaded or ready; file must exist in storage.
    """
    doc = await _get_document_for_user(db, document_id, current_user)

    if doc.status not in ("uploaded", "ready"):
        raise HTTPException(
            status_code=400,
            detail=f"Document must be uploaded or ready to reingest; current: {doc.status}",
        )

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured; set OPENAI_API_KEY",
        )

    await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    doc.status = "processing"
    doc.error_message = None
    doc.page_count = None
    await db.commit()

    background_tasks.add_task(run_ingestion, document_id)

    return {"status": "processing", "document_id": str(document_id)}


@router.get("/{document_id}/chunk-stats")
async def chunk_stats(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Dev: ingestion stats for a document.
    Returns total_chunks, low_signal_chunks, embedded_chunks, pages_covered,
    avg/min/max chunk length, sample_previews (first 120 chars per chunk).
    """
    await _get_document_for_user(db, document_id, current_user)

    r = await db.execute(
        text("""
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN is_low_signal THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END), 0),
                COUNT(DISTINCT page_number),
                COALESCE(ROUND(AVG(LENGTH(content)))::int, 0),
                COALESCE(MIN(LENGTH(content)), 0),
                COALESCE(MAX(LENGTH(content)), 0)
            FROM document_chunks WHERE document_id = :doc_id
        """),
        {"doc_id": document_id},
    )
    agg_row = r.fetchone()
    # Section-type breakdown for job description documents
    section_breakdown: dict[str, int] = {}
    try:
        section_r = await db.execute(
            text("""
                SELECT section_type, COUNT(*) FROM document_chunks
                WHERE document_id = :doc_id AND section_type IS NOT NULL
                GROUP BY section_type
            """),
            {"doc_id": document_id},
        )
        for sec_row in section_r:
            section_breakdown[str(sec_row[0])] = sec_row[1]
    except Exception:
        pass

    previews_r = await db.execute(
        select(
            DocumentChunk.chunk_index,
            DocumentChunk.page_number,
            DocumentChunk.content,
            DocumentChunk.is_low_signal,
            DocumentChunk.section_type,
        )
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    preview_rows = previews_r.all()
    sample_previews = [
        {
            "chunk_index": r.chunk_index,
            "page_number": r.page_number,
            "preview": (r.content or "")[:120],
            "is_low_signal": r.is_low_signal,
            "section_type": getattr(r, "section_type", None),
        }
        for r in preview_rows
    ]

    if not agg_row or agg_row[0] == 0:
        return {
            "total_chunks": 0,
            "low_signal_chunks": 0,
            "embedded_chunks": 0,
            "pages_covered": 0,
            "avg_chunk_length": 0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
            "sample_previews": sample_previews,
            "section_type_breakdown": section_breakdown,
        }
    return {
        "total_chunks": agg_row[0],
        "low_signal_chunks": agg_row[1],
        "embedded_chunks": agg_row[2],
        "pages_covered": agg_row[3],
        "avg_chunk_length": agg_row[4],
        "min_chunk_length": agg_row[5],
        "max_chunk_length": agg_row[6],
        "sample_previews": sample_previews,
        "section_type_breakdown": section_breakdown,
    }


# --- Interview Kit Sources (optional resume, company, notes) ---

class AddTextSourceInput(BaseModel):
    source_type: str = Field(..., pattern=r"^(resume|company|notes)$")
    title: str = Field("", max_length=200)
    content: str = Field(..., min_length=1)


class AddTextSourceOutput(BaseModel):
    source_id: str
    status: str = "ingested"


class PresignResumeInput(BaseModel):
    filename: str = Field(..., min_length=1)
    file_size_bytes: int = Field(..., gt=0)


class PresignResumeOutput(BaseModel):
    s3_key: str
    upload_url: str
    method: str = "PUT"


class ConfirmResumeInput(BaseModel):
    s3_key: str = Field(..., min_length=1)


class IngestResumeInput(BaseModel):
    s3_key: str = Field(..., min_length=1)


class AddFromUrlInput(BaseModel):
    url: str = Field(..., min_length=5)
    title: str = Field("Company / About", max_length=200)


class SourceSummary(BaseModel):
    id: str
    source_type: str
    title: str


class GapAnalysisInput(BaseModel):
    pass


class GapCitation(BaseModel):
    chunkId: str
    page: int | None = None
    sourceTitle: str = ""
    sourceType: str = "jd"


class GapEvidenceItem(BaseModel):
    chunkId: str
    page: int | None = None
    snippet: str = ""
    sourceTitle: str = ""
    sourceType: str = "jd"
    retrieval_source: str | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None
    final_score: float | None = None


class GapRequirementItem(BaseModel):
    id: str
    type: str
    label: str
    importance: str
    description: str | None = None
    status: str
    reason: str
    confidence: str
    jd_evidence: list[GapEvidenceItem] = []
    resume_evidence: list[GapEvidenceItem] = []


class GapCitedItem(BaseModel):
    text: str
    citations: list[GapCitation] = []


class ResumeSourceUsed(BaseModel):
    sourceId: str
    title: str
    documentId: str


class GapAnalysisOutput(BaseModel):
    summary: str
    overall_alignment_score: int
    matched_requirements: list[GapRequirementItem] = []
    partial_requirements: list[GapRequirementItem] = []
    gap_requirements: list[GapRequirementItem] = []
    strengths_cited: list[GapCitedItem] = []
    gaps_cited: list[GapCitedItem] = []
    resume_sources_considered: list[ResumeSourceUsed] = []


def _make_source_s3_key(document_id: uuid.UUID, filename: str) -> str:
    safe = re.sub(r"[^\w\.\-]", "_", filename)
    return f"sources/{document_id}/{safe}"


@router.post("/{document_id}/sources/add-text", response_model=AddTextSourceOutput)
async def add_text_source(
    document_id: uuid.UUID,
    body: AddTextSourceInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add a pasted text source (resume, company, notes). Chunks and embeds."""
    doc = await _get_document_for_user(db, document_id, current_user)
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document must be ready (ingested)")

    from app.services.source_ingestion import ingest_text_source

    try:
        source_id = await ingest_text_source(
            db=db,
            document_id=document_id,
            source_type=body.source_type,
            title=body.title or body.source_type.replace("_", " ").title(),
            content=body.content,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AddTextSourceOutput(source_id=source_id)


@router.post("/{document_id}/sources/presign-resume", response_model=PresignResumeOutput)
async def presign_resume(
    document_id: uuid.UUID,
    body: PresignResumeInput,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get presigned URL for resume PDF upload."""
    _validate_pdf_size(body.file_size_bytes)
    doc = await _get_document_for_user(db, document_id, current_user)
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document must be ready")

    s3_key = _make_source_s3_key(document_id, body.filename)
    storage = get_storage()
    upload_url, method = storage.generate_presigned_put(
        key=s3_key,
        content_type="application/pdf",
    )
    if upload_url.startswith("/"):
        base = str(request.base_url).rstrip("/")
        upload_url = f"{base}{upload_url}"

    return PresignResumeOutput(s3_key=s3_key, upload_url=upload_url, method=method)


@router.post("/{document_id}/sources/ingest-resume")
async def ingest_resume(
    document_id: uuid.UUID,
    body: IngestResumeInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Confirm resume upload and ingest (extract, chunk, embed)."""
    await _get_document_for_user(db, document_id, current_user)

    storage = get_storage()
    if not storage.exists(body.s3_key):
        raise HTTPException(status_code=400, detail="File not found in storage; upload failed")

    from pathlib import Path
    from app.services.source_ingestion import ingest_resume_pdf

    original_filename = Path(body.s3_key).name
    try:
        source_id = await ingest_resume_pdf(
            db=db,
            document_id=document_id,
            s3_key=body.s3_key,
            original_filename=original_filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"source_id": source_id, "status": "ingested"}


@router.post("/{document_id}/sources/add-from-url", response_model=AddTextSourceOutput)
async def add_from_url(
    document_id: uuid.UUID,
    body: AddFromUrlInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch URL content and add as company source."""
    doc = await _get_document_for_user(db, document_id, current_user)
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document must be ready")

    from app.services.source_ingestion import fetch_url_content, ingest_text_source

    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        content = fetch_url_content(body.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)[:100]}")
    if not content or len(content) < 50:
        raise HTTPException(status_code=400, detail="URL returned insufficient text content")

    try:
        source_id = await ingest_text_source(
            db=db,
            document_id=document_id,
            source_type="company",
            title=body.title or "Company / About",
            content=content,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AddTextSourceOutput(source_id=source_id)


@router.get("/{document_id}/sources", response_model=list[SourceSummary])
async def list_sources(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List Interview Kit sources for a document (excluding JD)."""
    await _get_document_for_user(db, document_id, current_user)

    src_result = await db.execute(
        select(InterviewSource)
        .where(
            InterviewSource.document_id == document_id,
            InterviewSource.source_type != "jd",
        )
        .order_by(InterviewSource.created_at)
    )
    sources = src_result.scalars().all()
    return [SourceSummary(id=str(s.id), source_type=s.source_type, title=s.title) for s in sources]


@router.post("/{document_id}/gap-analysis", response_model=GapAnalysisOutput)
async def gap_analysis(
    document_id: uuid.UUID,
    body: GapAnalysisInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a retrieval-backed resume gap analysis for a job description."""
    doc = await _get_document_for_user(db, document_id, current_user)
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document must be ready")
    if doc.doc_domain != "job_description":
        raise HTTPException(status_code=400, detail="Gap analysis requires a job description document")

    analysis = await generate_gap_analysis(db, document=doc, user_id=current_user.id)
    if not analysis.get("resume_sources_considered"):
        raise HTTPException(
            status_code=400,
            detail="No resume source found. Upload an account-level resume or attach a resume source first.",
        )
    return GapAnalysisOutput(**analysis)
