"""GET /fit-history: past fit analyses and score progression."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.db.session import get_db
from app.models import FitAnalysis, User
from app.services.fit_progression import build_fit_progression

router = APIRouter(prefix="/fit-history", tags=["fit-history"])

_PROGRESSION_ROW_CAP = 3000


class FitHistoryAnalysisItem(BaseModel):
    id: str
    job_description_id: str
    resume_id: str
    fit_score: int
    summary: str
    matches: list[dict] = Field(default_factory=list)
    gaps: list[dict] = Field(default_factory=list)
    recommendations: list[dict] = Field(default_factory=list)
    created_at: str


class FitProgressionItem(BaseModel):
    job_description_id: str
    resume_id: str
    first_score: int
    latest_score: int
    delta: int
    run_count: int
    improved: bool | None = Field(
        description="True if latest > first over 2+ runs; null if only one run.",
    )
    message: str


class FitHistoryResponse(BaseModel):
    analyses: list[FitHistoryAnalysisItem]
    progression: list[FitProgressionItem]


@router.get("", response_model=FitHistoryResponse)
async def fit_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    job_description_id: str | None = Query(
        default=None,
        description="Only include analyses for this job description document.",
    ),
    resume_id: str | None = Query(
        default=None,
        description="Only include analyses for this resume document.",
    ),
):
    """
    Return recent stored analyze-fit results for the current user, plus per (JD, resume)
    progression for messaging such as "Your fit improved from X → Y".
    """
    jd_filter: uuid.UUID | None = None
    rs_filter: uuid.UUID | None = None
    if job_description_id and str(job_description_id).strip():
        try:
            jd_filter = uuid.UUID(str(job_description_id).strip())
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail="Invalid job_description_id",
            ) from e
    if resume_id and str(resume_id).strip():
        try:
            rs_filter = uuid.UUID(str(resume_id).strip())
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail="Invalid resume_id",
            ) from e

    list_stmt = select(FitAnalysis).where(FitAnalysis.user_id == current_user.id)
    prog_stmt = select(FitAnalysis).where(FitAnalysis.user_id == current_user.id)
    if jd_filter is not None:
        list_stmt = list_stmt.where(FitAnalysis.job_description_id == jd_filter)
        prog_stmt = prog_stmt.where(FitAnalysis.job_description_id == jd_filter)
    if rs_filter is not None:
        list_stmt = list_stmt.where(FitAnalysis.resume_id == rs_filter)
        prog_stmt = prog_stmt.where(FitAnalysis.resume_id == rs_filter)

    list_stmt = list_stmt.order_by(FitAnalysis.created_at.desc()).limit(limit).offset(offset)
    prog_stmt = prog_stmt.order_by(FitAnalysis.created_at.asc()).limit(_PROGRESSION_ROW_CAP)

    list_result = await db.execute(list_stmt)
    list_rows = list_result.scalars().all()
    prog_result = await db.execute(prog_stmt)
    prog_rows = prog_result.scalars().all()

    analyses = [
        FitHistoryAnalysisItem(
            id=str(r.id),
            job_description_id=str(r.job_description_id),
            resume_id=str(r.resume_id),
            fit_score=int(r.fit_score),
            summary=r.summary or "",
            matches=list(r.matches_json) if isinstance(r.matches_json, list) else [],
            gaps=list(r.gaps_json) if isinstance(r.gaps_json, list) else [],
            recommendations=list(r.recommendations_json)
            if isinstance(r.recommendations_json, list)
            else [],
            created_at=r.created_at.isoformat(),
        )
        for r in list_rows
    ]

    progression_raw = build_fit_progression(list(prog_rows))
    progression = [FitProgressionItem(**p) for p in progression_raw]

    return FitHistoryResponse(analyses=analyses, progression=progression)
