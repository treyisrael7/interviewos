"""Interview analytics endpoints (user overview + per-session)."""

import uuid
from collections import defaultdict

from fastapi import Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.db.session import get_db
from app.models import InterviewAnswer, InterviewQuestion, InterviewSession, User
from app.routers.interview.helpers import competency_key_for_question
from app.routers.interview.router import router
from app.routers.interview.schemas import (
    CompetencyStats,
    GlobalScoreTrendPoint,
    ImprovementSummary,
    InterviewAnalyticsOverview,
    InterviewSessionAnalytics,
    RecentSessionAnalyticsRow,
    ScoreTrendPoint,
)


@router.get("/analytics/overview", response_model=InterviewAnalyticsOverview)
async def get_interview_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    User-wide interview stats: score trend, competency strengths/weaknesses, recent sessions.
    """
    session_count_row = await db.execute(
        select(func.count())
        .select_from(InterviewSession)
        .where(InterviewSession.user_id == current_user.id)
    )
    total_session_count = int(session_count_row.scalar_one() or 0)

    ar = await db.execute(
        select(InterviewAnswer, InterviewQuestion, InterviewSession)
        .join(InterviewQuestion, InterviewAnswer.question_id == InterviewQuestion.id)
        .join(InterviewSession, InterviewQuestion.session_id == InterviewSession.id)
        .where(InterviewSession.user_id == current_user.id)
        .order_by(InterviewAnswer.created_at.asc())
    )
    rows = ar.all()

    trend: list[GlobalScoreTrendPoint] = []
    scores: list[float] = []
    by_label: dict[str, dict] = defaultdict(lambda: {"scores": [], "competency_id": None})
    session_scores: dict[uuid.UUID, list[float]] = defaultdict(list)
    session_meta: dict[uuid.UUID, object] = {}

    for ans, q, sess in rows:
        s = float(ans.score)
        scores.append(s)
        session_scores[sess.id].append(s)
        session_meta[sess.id] = sess.created_at
        trend.append(
            GlobalScoreTrendPoint(
                at=ans.created_at.isoformat() if ans.created_at else "",
                score=s,
                session_id=sess.id,
                question_id=q.id,
            )
        )
        cid, label = competency_key_for_question(q)
        by_label[label]["scores"].append(s)
        if cid is not None:
            by_label[label]["competency_id"] = cid

    n = len(scores)
    overall_avg = round(sum(scores) / n, 2) if n else None

    comp_rows: list[CompetencyStats] = []
    for label, data in by_label.items():
        sc = data["scores"]
        if not sc:
            continue
        comp_rows.append(
            CompetencyStats(
                competency_id=data.get("competency_id"),
                competency_label=label,
                average_score=round(sum(sc) / len(sc), 2),
                answer_count=len(sc),
            )
        )
    top_n = 5
    strongest = sorted(comp_rows, key=lambda x: x.average_score, reverse=True)[:top_n]
    weakest = sorted(comp_rows, key=lambda x: x.average_score)[:top_n]

    sr = await db.execute(
        select(
            InterviewSession,
            func.count(InterviewQuestion.id).label("question_count"),
        )
        .outerjoin(InterviewQuestion, InterviewQuestion.session_id == InterviewSession.id)
        .where(InterviewSession.user_id == current_user.id)
        .group_by(InterviewSession.id)
        .order_by(InterviewSession.created_at.desc())
        .limit(12)
    )
    sess_rows = sr.all()

    recent: list[RecentSessionAnalyticsRow] = []
    for row in sess_rows:
        s = row[0]
        qcount = int(row[1] or 0)
        sc_list = session_scores.get(s.id, [])
        ac = len(sc_list)
        avg_s = round(sum(sc_list) / len(sc_list), 2) if sc_list else None
        recent.append(
            RecentSessionAnalyticsRow(
                id=s.id,
                document_id=s.document_id,
                created_at=s.created_at.isoformat() if s.created_at else "",
                difficulty=s.difficulty,
                question_count=qcount,
                answer_count=ac,
                average_score=avg_s,
            )
        )

    def _session_sort_key(sid: uuid.UUID) -> float:
        ct = session_meta.get(sid)
        if ct is None:
            return 0.0
        try:
            return ct.timestamp()  # type: ignore[union-attr]
        except (AttributeError, OSError):
            return 0.0

    ordered_with_answers = sorted(
        [sid for sid in session_scores if session_scores[sid]],
        key=_session_sort_key,
        reverse=True,
    )
    pct_change: float | None = None
    if len(ordered_with_answers) >= 2:
        last_avg = sum(session_scores[ordered_with_answers[0]]) / len(
            session_scores[ordered_with_answers[0]]
        )
        prior_avg = sum(session_scores[ordered_with_answers[1]]) / len(
            session_scores[ordered_with_answers[1]]
        )
        if prior_avg > 0:
            pct_change = round((last_avg - prior_avg) / prior_avg * 100, 1)
        elif last_avg > 0:
            pct_change = 100.0
        else:
            pct_change = 0.0

    focus_hint: str | None = None
    if weakest and weakest[0].answer_count >= 1:
        focus_hint = weakest[0].competency_label

    return InterviewAnalyticsOverview(
        total_session_count=total_session_count,
        total_answer_count=n,
        overall_average_score=overall_avg,
        score_trend=trend,
        strongest_competencies=strongest,
        weakest_competencies=weakest,
        recent_sessions=recent,
        last_session_vs_prior_percent_change=pct_change,
        focus_area_hint=focus_hint,
    )


@router.get("/{session_id}/analytics", response_model=InterviewSessionAnalytics)
async def get_session_analytics(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregated scores and competency trends for a session (chronological answers).
    """
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    assert_resource_ownership(session, current_user)

    ar = await db.execute(
        select(InterviewAnswer, InterviewQuestion)
        .join(InterviewQuestion, InterviewAnswer.question_id == InterviewQuestion.id)
        .where(InterviewQuestion.session_id == session_id)
        .order_by(InterviewAnswer.created_at.asc())
    )
    rows = ar.all()

    trend: list[ScoreTrendPoint] = []
    scores: list[float] = []
    by_label: dict[str, dict] = defaultdict(lambda: {"scores": [], "competency_id": None})

    for ans, q in rows:
        scores.append(float(ans.score))
        trend.append(
            ScoreTrendPoint(
                at=ans.created_at.isoformat() if ans.created_at else "",
                score=float(ans.score),
                question_id=q.id,
            )
        )
        cid, label = competency_key_for_question(q)
        by_label[label]["scores"].append(float(ans.score))
        if cid is not None:
            by_label[label]["competency_id"] = cid

    n = len(scores)
    avg = sum(scores) / n if n else None

    comp_rows: list[CompetencyStats] = []
    for label, data in by_label.items():
        sc = data["scores"]
        if not sc:
            continue
        comp_rows.append(
            CompetencyStats(
                competency_id=data.get("competency_id"),
                competency_label=label,
                average_score=round(sum(sc) / len(sc), 2),
                answer_count=len(sc),
            )
        )
    top_n = 5
    strongest = sorted(comp_rows, key=lambda x: x.average_score, reverse=True)[:top_n]
    weakest = sorted(comp_rows, key=lambda x: x.average_score)[:top_n]

    first_half_avg: float | None = None
    second_half_avg: float | None = None
    improvement_delta: float | None = None
    if n >= 2:
        mid = n // 2
        first_part = scores[:mid]
        second_part = scores[mid:]
        if first_part:
            first_half_avg = round(sum(first_part) / len(first_part), 2)
        if second_part:
            second_half_avg = round(sum(second_part) / len(second_part), 2)
        if first_half_avg is not None and second_half_avg is not None:
            improvement_delta = round(second_half_avg - first_half_avg, 2)

    return InterviewSessionAnalytics(
        session_id=session_id,
        answer_count=n,
        average_score=round(avg, 2) if avg is not None else None,
        score_trend=trend,
        strongest_competencies=strongest,
        weakest_competencies=weakest,
        improvement=ImprovementSummary(
            answer_count=n,
            first_half_average=first_half_avg,
            second_half_average=second_half_avg,
            improvement_delta=improvement_delta,
        ),
    )
