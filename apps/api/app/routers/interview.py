"""Interview Prep: sessions, questions, generate, evaluate."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import Document, InterviewAnswer, InterviewQuestion, InterviewSession, User
from app.services.interview import (
    evaluate_answer_with_retrieval,
    generate_questions,
)
from app.services.role_intelligence import VALID_DOMAINS, VALID_SENIORITIES

router = APIRouter(prefix="/interview", tags=["interview"])
logger = logging.getLogger(__name__)


# --- Schemas ---


# Question mix presets (behavioral, roleSpecific, scenario - must sum to 100)
QUESTION_MIX_PRESETS = {
    "balanced": {"behavioral": 40, "roleSpecific": 30, "scenario": 30},
    "behavioral_heavy": {"behavioral": 60, "roleSpecific": 25, "scenario": 15},
    "scenario_heavy": {"behavioral": 25, "roleSpecific": 25, "scenario": 50},
}

class InterviewGenerateInput(BaseModel):
    document_id: uuid.UUID
    difficulty: str = Field("junior", pattern="^(junior|mid|senior)$")
    num_questions: int = Field(8, ge=1, le=10)
    domain_override: str | None = None
    seniority_override: str | None = None
    question_mix_preset: str | None = None  # "balanced" | "behavioral_heavy" | "scenario_heavy"


class EvidenceItem(BaseModel):
    chunk_id: str
    page_number: int
    snippet: str


class InterviewQuestionOutput(BaseModel):
    id: uuid.UUID
    type: str
    focus_area: str = ""
    competency_id: str | None = None
    competency_label: str | None = None
    question: str
    key_topics: list[str] = []
    evidence: list[EvidenceItem]
    rubric_bullets: list[str]


class InterviewGenerateOutput(BaseModel):
    session_id: uuid.UUID
    questions: list[InterviewQuestionOutput]


class InterviewEvaluateInput(BaseModel):
    document_id: uuid.UUID
    question_id: uuid.UUID
    answer_text: str = Field(..., min_length=1)


class EvidenceUsedItem(BaseModel):
    quote: str
    sourceId: str
    sourceType: str | None = None
    sourceTitle: str | None = None
    page: int | None = None
    chunkId: str | None = None


class CitationItem(BaseModel):
    chunkId: str
    page: int | None = None
    sourceTitle: str = ""
    sourceType: str = "jd"


class CitedItem(BaseModel):
    text: str
    citations: list[CitationItem] = []


class InterviewEvaluateOutput(BaseModel):
    answer_id: uuid.UUID
    score: float
    strengths: list[str]
    gaps: list[str]
    strengths_cited: list[CitedItem] = []
    gaps_cited: list[CitedItem] = []
    improved_answer: str
    follow_up_questions: list[str]
    suggested_followup: str | None = None
    evidence_used: list[EvidenceUsedItem]


class RoleProfileOut(BaseModel):
    domain: str
    seniority: str
    roleTitleGuess: str = ""
    focusAreas: list[str] = []
    questionMix: dict = {}


# Helper to convert role_profile from dict
def _to_role_profile_out(rp: dict | None) -> RoleProfileOut | None:
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


# Helper to extract from rubric_json
def _from_rubric(rubric: dict) -> tuple[list[str], list[dict], list[str], str, list[str], str | None, str | None, list[str]]:
    """Returns (bullets, evidence, key_topics, focus_area, must_mention, competency_id, competency_label, evidence_chunk_ids)."""
    if not rubric:
        return [], [], [], "", [], None, None, []
    bullets = rubric.get("bullets") or []
    evidence = rubric.get("evidence") or []
    key_topics = rubric.get("key_topics") or []
    focus_area = str(rubric.get("focus_area") or rubric.get("competency_label") or "").strip()
    competency_id = rubric.get("competency_id")
    competency_label = rubric.get("competency_label")
    must_mention = rubric.get("must_mention") or []
    evidence_chunk_ids = rubric.get("evidence_chunk_ids") or []
    if isinstance(evidence_chunk_ids, list):
        evidence_chunk_ids = [str(x) for x in evidence_chunk_ids if x]
    else:
        evidence_chunk_ids = []
    return (
        bullets if isinstance(bullets, list) else [],
        evidence if isinstance(evidence, list) else [],
        key_topics if isinstance(key_topics, list) else [],
        focus_area,
        must_mention if isinstance(must_mention, list) else [],
        competency_id,
        competency_label,
        evidence_chunk_ids,
    )


@router.post("/generate", response_model=InterviewGenerateOutput)
async def generate(
    body: InterviewGenerateInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate interview questions from a job description document.
    Creates a new session and attaches questions.
    Requires doc_domain=job_description and status=ready.
    """
    result = await db.execute(select(Document).where(Document.id == body.document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    assert_resource_ownership(doc, current_user)

    if doc.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Document must be ingested and ready; current status: {doc.status}",
        )

    if doc.doc_domain != "job_description":
        raise HTTPException(
            status_code=400,
            detail="Interview generation requires a job description document (doc_domain=job_description)",
        )

    if body.domain_override and body.domain_override not in VALID_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Invalid domain_override: {body.domain_override}")
    if body.seniority_override and body.seniority_override not in VALID_SENIORITIES:
        raise HTTPException(status_code=400, detail=f"Invalid seniority_override: {body.seniority_override}")
    if body.question_mix_preset and body.question_mix_preset not in QUESTION_MIX_PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid question_mix_preset: {body.question_mix_preset}")

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured; set OPENAI_API_KEY",
        )

    # Role profile from document, merge with user overrides
    role_profile = getattr(doc, "role_profile", None) if doc else None
    if not role_profile or not isinstance(role_profile, dict):
        from app.services.interview import DEFAULT_ROLE_PROFILE
        role_profile = DEFAULT_ROLE_PROFILE.copy()
    else:
        role_profile = dict(role_profile)

    # Apply overrides from Setup Advanced
    if body.domain_override:
        role_profile["domain"] = body.domain_override
    if body.seniority_override:
        role_profile["seniority"] = body.seniority_override
    if body.question_mix_preset and body.question_mix_preset in QUESTION_MIX_PRESETS:
        role_profile["questionMix"] = QUESTION_MIX_PRESETS[body.question_mix_preset].copy()

    # Competencies from document (fallback to focusAreas if none)
    competencies = getattr(doc, "competencies", None)
    if not competencies or not isinstance(competencies, list):
        competencies = []

    # Generate questions (competency-tied, RAG per question)
    try:
        raw_questions = await generate_questions(
            db=db,
            document_id=body.document_id,
            num_questions=body.num_questions,
            role_profile=role_profile,
            competencies=competencies,
        )
    except Exception as e:
        logger.exception("generate_questions failed")
        raise HTTPException(status_code=503, detail=f"Question generation failed: {str(e)[:200]}")

    if not raw_questions:
        raise HTTPException(
            status_code=400,
            detail="No questions generated. Ensure document has competencies or sections (e.g. responsibilities, qualifications).",
        )

    # Create session and questions (store effective profile + overrides)
    session = InterviewSession(
        user_id=current_user.id,
        document_id=body.document_id,
        mode="role_driven",
        difficulty=body.difficulty,
        role_profile_json=role_profile,
        domain_override=body.domain_override,
        seniority_override=body.seniority_override,
        question_mix_override=(
            QUESTION_MIX_PRESETS.get(body.question_mix_preset) if body.question_mix_preset else None
        ),
    )
    db.add(session)
    await db.flush()

    for rq in raw_questions:
        rubric_json = {
            "bullets": rq.get("whatGoodLooksLike") or [],
            "must_mention": rq.get("mustMention") or [],
            "focus_area": rq.get("competencyLabel") or rq.get("focusArea") or "",
            "competency_id": rq.get("competencyId"),
            "competency_label": rq.get("competencyLabel"),
            "evidence_chunk_ids": rq.get("evidenceChunkIds") or [],
            "evidence": rq.get("evidence") or [],
            "key_topics": [rq.get("competencyLabel") or rq.get("focusArea")] if (rq.get("competencyLabel") or rq.get("focusArea")) else [],
        }
        q_type = rq.get("type", "behavioral")
        q = InterviewQuestion(
            session_id=session.id,
            type=q_type,
            question=rq.get("questionText") or rq.get("question", ""),
            rubric_json=rubric_json,
        )
        db.add(q)

    await db.commit()
    await db.refresh(session)

    q_result = await db.execute(
        select(InterviewQuestion)
        .where(InterviewQuestion.session_id == session.id)
        .order_by(InterviewQuestion.created_at)
    )
    questions = q_result.scalars().all()

    def _norm_type(t: str) -> str:
        """Normalize legacy 'technical' to 'role_specific' for API consistency."""
        if t == "technical":
            return "role_specific"
        return t

    def _q_out(q) -> InterviewQuestionOutput:
        bullets, evidence, key_topics, focus_area, _, comp_id, comp_label, _ = _from_rubric(q.rubric_json)
        return InterviewQuestionOutput(
            id=q.id,
            type=_norm_type(q.type),
            focus_area=focus_area,
            competency_id=comp_id,
            competency_label=comp_label,
            question=q.question,
            key_topics=key_topics,
            evidence=[EvidenceItem(**e) for e in evidence],
            rubric_bullets=bullets,
        )

    return InterviewGenerateOutput(
        session_id=session.id,
        questions=[_q_out(q) for q in questions],
    )


@router.post("/evaluate", response_model=InterviewEvaluateOutput)
async def evaluate(
    body: InterviewEvaluateInput,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Evaluate a candidate's answer against the stored question rubric and evidence.
    Saves answer + score + feedback. Requires question to belong to user's document.
    """
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API not configured; set OPENAI_API_KEY",
        )

    # Load question and session (for role_profile)
    result = await db.execute(
        select(InterviewQuestion, InterviewSession)
        .join(InterviewSession, InterviewQuestion.session_id == InterviewSession.id)
        .where(
            InterviewQuestion.id == body.question_id,
            InterviewSession.document_id == body.document_id,
        )
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")

    question = row[0]
    session = row[1]
    assert_resource_ownership(session, current_user)
    bullets, evidence, _, focus_area, must_mention, comp_id, comp_label, evidence_chunk_ids = _from_rubric(question.rubric_json)
    role_profile = _to_role_profile_out(getattr(session, "role_profile_json", None))
    rp_dict = (
        {
            "domain": role_profile.domain,
            "seniority": role_profile.seniority,
            "focusAreas": role_profile.focusAreas,
            "questionMix": role_profile.questionMix,
        }
        if role_profile
        else {}
    )

    try:
        evaluation = await evaluate_answer_with_retrieval(
            db=db,
            document_id=session.document_id,
            user_id=current_user.id,
            question=question.question,
            question_type=question.type if question.type != "technical" else "role_specific",
            focus_area=focus_area,
            competency_id=comp_id,
            competency_label=comp_label,
            evidence_chunk_ids=evidence_chunk_ids,
            what_good_looks_like=bullets,
            must_mention=must_mention,
            role_profile=rp_dict,
            user_answer=body.answer_text,
            evidence=evidence,
        )
    except Exception as e:
        logger.exception("evaluate_answer_with_retrieval failed")
        raise HTTPException(status_code=503, detail=f"Evaluation failed: {str(e)[:200]}")

    follow_ups = evaluation.get("follow_up_questions") or []
    if evaluation.get("suggested_followup") and not follow_ups:
        follow_ups = [evaluation["suggested_followup"]]

    feedback_json = {
        "strengths": evaluation["strengths"],
        "gaps": evaluation["gaps"],
        "strengths_cited": evaluation.get("strengths_cited", []),
        "gaps_cited": evaluation.get("gaps_cited", []),
        "improved_answer": evaluation["improved_answer"],
        "follow_up_questions": follow_ups,
        "suggested_followup": evaluation.get("suggested_followup"),
        "evidence_used": evaluation["evidence_used"],
    }

    answer = InterviewAnswer(
        question_id=body.question_id,
        answer_text=body.answer_text,
        score=evaluation["score"],
        feedback_json=feedback_json,
    )
    db.add(answer)
    await db.commit()
    await db.refresh(answer)

    evidence_used = evaluation.get("evidence_used", [])
    strengths_cited = evaluation.get("strengths_cited", [])
    gaps_cited = evaluation.get("gaps_cited", [])

    def _to_cited(item: dict) -> "CitedItem":
        cites = []
        for c in item.get("citations") or []:
            if isinstance(c, dict) and c.get("chunkId"):
                cites.append(CitationItem(
                    chunkId=str(c["chunkId"]),
                    page=c.get("page"),
                    sourceTitle=str(c.get("sourceTitle", "")),
                    sourceType=str(c.get("sourceType", "jd")),
                ))
        return CitedItem(text=str(item.get("text", "")), citations=cites)

    return InterviewEvaluateOutput(
        answer_id=answer.id,
        score=evaluation["score"],
        strengths=evaluation["strengths"],
        gaps=evaluation["gaps"],
        strengths_cited=[_to_cited(s) for s in strengths_cited if isinstance(s, dict)],
        gaps_cited=[_to_cited(g) for g in gaps_cited if isinstance(g, dict)],
        improved_answer=evaluation["improved_answer"],
        follow_up_questions=follow_ups,
        suggested_followup=evaluation.get("suggested_followup"),
        evidence_used=[EvidenceUsedItem(**e) for e in evidence_used],
    )


# --- GET endpoints ---


class SessionSummary(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    mode: str
    difficulty: str
    created_at: str
    question_count: int
    role_profile: RoleProfileOut | None = None


class SessionDetail(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    document_id: uuid.UUID
    mode: str
    difficulty: str
    created_at: str
    questions: list[InterviewQuestionOutput]
    role_profile: RoleProfileOut | None = None


class QuestionDetail(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    type: str
    focus_area: str = ""
    competency_id: str | None = None
    competency_label: str | None = None
    question: str
    key_topics: list[str]
    evidence: list[EvidenceItem]
    rubric_bullets: list[str]
    created_at: str


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List interview sessions for a user."""
    from sqlalchemy import func

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
            role_profile=_to_role_profile_out(getattr(row[0], "role_profile_json", None)),
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
        role_profile=_to_role_profile_out(getattr(session, "role_profile_json", None)),
        questions=[
            InterviewQuestionOutput(
                id=q.id,
                type=q.type if q.type != "technical" else "role_specific",
                focus_area=_from_rubric(q.rubric_json)[3],
                competency_id=_from_rubric(q.rubric_json)[5],
                competency_label=_from_rubric(q.rubric_json)[6],
                question=q.question,
                key_topics=_from_rubric(q.rubric_json)[2],
                evidence=[EvidenceItem(**e) for e in _from_rubric(q.rubric_json)[1]],
                rubric_bullets=_from_rubric(q.rubric_json)[0],
            )
            for q in questions
        ],
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
    bullets, evidence, key_topics, focus_area, _, comp_id, comp_label, _ = _from_rubric(question.rubric_json)

    return QuestionDetail(
        id=question.id,
        session_id=question.session_id,
        type=question.type if question.type != "technical" else "role_specific",
        focus_area=focus_area,
        competency_id=comp_id,
        competency_label=comp_label,
        question=question.question,
        key_topics=key_topics,
        evidence=[EvidenceItem(**e) for e in evidence],
        rubric_bullets=bullets,
        created_at=question.created_at.isoformat() if question.created_at else "",
    )
