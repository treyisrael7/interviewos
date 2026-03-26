"""GET interview session and question endpoints."""

import uuid

from fastapi import Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.db.session import get_db
from app.models import InterviewQuestion, InterviewSession, User
from app.routers.interview.helpers import from_rubric, norm_question_type_for_api, question_to_output, to_role_profile_out
from app.routers.interview.router import router
from app.routers.interview.schemas import EvidenceItem, QuestionDetail, SessionDetail, SessionSummary


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List interview sessions for a user."""
    result = await db.execute(
        select(
            InterviewSession,
            func.count(InterviewQuestion.id).label("question_count"),
        )
        .outerjoin(InterviewQuestion, InterviewQuestion.session_id == InterviewSession.id)
        .where(InterviewSession.user_id == current_user.id)
        .group_by(InterviewSession.id)
        .order_by(InterviewSession.created_at.desc())
    )
    rows = result.all()

    return [
        SessionSummary(
            id=row[0].id,
            document_id=row[0].document_id,
            mode=row[0].mode,
            difficulty=row[0].difficulty,
            created_at=row[0].created_at.isoformat() if row[0].created_at else "",
            question_count=row[1] or 0,
            role_profile=to_role_profile_out(getattr(row[0], "role_profile_json", None)),
        )
        for row in rows
    ]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a session with its questions. Validates user ownership."""
    result = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    assert_resource_ownership(session, current_user)

    q_result = await db.execute(
        select(InterviewQuestion)
        .where(InterviewQuestion.session_id == session_id)
        .order_by(InterviewQuestion.created_at)
    )
    questions = q_result.scalars().all()

    return SessionDetail(
        id=session.id,
        user_id=session.user_id,
        document_id=session.document_id,
        mode=session.mode,
        difficulty=session.difficulty,
        created_at=session.created_at.isoformat() if session.created_at else "",
        role_profile=to_role_profile_out(getattr(session, "role_profile_json", None)),
        questions=[question_to_output(q) for q in questions],
    )


@router.get("/questions/{question_id}", response_model=QuestionDetail)
async def get_question(
    question_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a question by ID. Validates user ownership via session."""
    result = await db.execute(
        select(InterviewQuestion, InterviewSession)
        .join(InterviewSession, InterviewQuestion.session_id == InterviewSession.id)
        .where(
            InterviewQuestion.id == question_id,
        )
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")

    question = row[0]
    assert_resource_ownership(row[1], current_user)
    bullets, evidence, key_topics, focus_area, _, comp_id, comp_label, _ = from_rubric(question.rubric_json)

    return QuestionDetail(
        id=question.id,
        session_id=question.session_id,
        type=norm_question_type_for_api(question.type),
        focus_area=focus_area,
        competency_id=comp_id,
        competency_label=comp_label,
        question=question.question,
        key_topics=key_topics,
        evidence=[EvidenceItem(**e) for e in evidence],
        rubric_bullets=bullets,
        created_at=question.created_at.isoformat() if question.created_at else "",
    )
