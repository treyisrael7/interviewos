"""POST /interview/generate and POST /interview/evaluate."""

import logging

from fastapi import Depends, HTTPException

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import assert_resource_ownership, get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models import Document, InterviewAnswer, InterviewQuestion, InterviewSession, User
from app.routers.interview.helpers import from_rubric, question_to_output, to_role_profile_out
from app.routers.interview.router import router
from app.routers.interview import runtime as interview_runtime
from app.routers.interview.runtime import evaluate_answer_with_retrieval
from app.routers.interview.schemas import (
    CitationItem,
    CitedItem,
    EvaluationCitationOut,
    EvidenceUsedItem,
    GapEvalItem,
    EvaluationUsageOut,
    InterviewEvaluateInput,
    InterviewEvaluateOutput,
    InterviewGenerateInput,
    InterviewGenerateOutput,
    QUESTION_MIX_PRESETS,
    RubricScoreItem,
    ScoreBreakdownOut,
    StrengthEvalItem,
)
from app.services.evaluation_usage import consume_evaluation_quota
from app.services.interview import normalize_rubric_scores_output
from app.services.performance_profile import compute_performance_profile, profile_answer_from_feedback
from app.services.interview_scoring import (
    build_feedback_summary,
    compute_score_breakdown,
    score_from_rubric_dimension_mean,
)
from app.services.role_intelligence import VALID_DOMAINS, VALID_SENIORITIES

logger = logging.getLogger(__name__)


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

    role_profile = getattr(doc, "role_profile", None) if doc else None
    if not role_profile or not isinstance(role_profile, dict):
        from app.services.interview import DEFAULT_ROLE_PROFILE

        role_profile = DEFAULT_ROLE_PROFILE.copy()
    else:
        role_profile = dict(role_profile)

    if body.domain_override:
        role_profile["domain"] = body.domain_override
    if body.seniority_override:
        role_profile["seniority"] = body.seniority_override
    if body.question_mix_preset and body.question_mix_preset in QUESTION_MIX_PRESETS:
        role_profile["questionMix"] = QUESTION_MIX_PRESETS[body.question_mix_preset].copy()

    competencies = getattr(doc, "competencies", None)
    if not competencies or not isinstance(competencies, list):
        competencies = []

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

    try:
        raw_questions = await interview_runtime.generate_questions(
            db=db,
            document_id=body.document_id,
            num_questions=body.num_questions,
            role_profile=role_profile,
            competencies=competencies,
            session_id=session.id,
        )
    except Exception as e:
        logger.exception("generate_questions failed")
        raise HTTPException(status_code=503, detail=f"Question generation failed: {str(e)[:200]}")

    if not raw_questions:
        raise HTTPException(
            status_code=400,
            detail="No questions generated. Ensure document has competencies or sections (e.g. responsibilities, qualifications).",
        )

    for rq in raw_questions:
        rubric_json = {
            "bullets": rq.get("whatGoodLooksLike") or [],
            "must_mention": rq.get("mustMention") or [],
            "focus_area": rq.get("competencyLabel") or rq.get("focusArea") or "",
            "competency_id": rq.get("competencyId"),
            "competency_label": rq.get("competencyLabel"),
            "evidence_chunk_ids": rq.get("evidenceChunkIds") or [],
            "evidence": rq.get("evidence") or [],
            "key_topics": [rq.get("competencyLabel") or rq.get("focusArea")]
            if (rq.get("competencyLabel") or rq.get("focusArea"))
            else [],
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

    return InterviewGenerateOutput(
        session_id=session.id,
        questions=[question_to_output(q) for q in questions],
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

    used, eval_limit, plan_key = await consume_evaluation_quota(db, current_user.id)

    bullets, evidence, _, focus_area, must_mention, comp_id, comp_label, evidence_chunk_ids = from_rubric(question.rubric_json)
    role_profile = to_role_profile_out(getattr(session, "role_profile_json", None))
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
            evaluation_mode=body.mode,
            session_id=session.id,
        )
    except Exception as e:
        logger.exception("evaluate_answer_with_retrieval failed")
        raise HTTPException(status_code=503, detail=f"Evaluation failed: {str(e)[:200]}")

    ev_for_scoring = evaluation.pop("evidence_for_scoring", []) or []
    rubric_scores_payload = normalize_rubric_scores_output(evaluation.get("rubric_scores"))
    mean_0_10, rubric_breakdown = score_from_rubric_dimension_mean(rubric_scores_payload)
    if rubric_breakdown is not None:
        score_breakdown = rubric_breakdown
        llm_score = float(mean_0_10)
    else:
        score_breakdown = compute_score_breakdown(
            user_answer=body.answer_text,
            evidence=ev_for_scoring,
            what_good_looks_like=bullets,
            must_mention=must_mention,
            role_profile=rp_dict,
            competency_label=comp_label,
        )
        llm_score = float(evaluation.get("score") or 0.0)
    feedback_summary = build_feedback_summary(score_breakdown)
    final_score = float(score_breakdown["overall"])

    follow_ups = evaluation.get("follow_up_questions") or []
    if evaluation.get("suggested_followup") and not follow_ups:
        follow_ups = [evaluation["suggested_followup"]]
    eval_summary = str(evaluation.get("summary") or "").strip()
    score_reasoning = str(evaluation.get("score_reasoning") or "").strip()
    strengths_raw = list(evaluation.get("strengths") or [])
    gaps_raw = list(evaluation.get("gaps") or [])
    citations_raw = list(evaluation.get("citations") or [])

    def _strength_items(raw: list) -> list[dict]:
        out: list[dict] = []
        for x in raw:
            if isinstance(x, dict):
                out.append(
                    {
                        "text": str(x.get("text", "")).strip(),
                        "evidence": str(x.get("evidence", "")).strip(),
                        "highlight": str(x.get("highlight", "")).strip(),
                        "impact": str(x.get("impact", "")).strip(),
                    }
                )
            elif isinstance(x, str) and x.strip():
                out.append({"text": x.strip(), "evidence": "", "highlight": "", "impact": ""})
        return out

    def _gap_items(raw: list) -> list[dict]:
        out: list[dict] = []
        for x in raw:
            if isinstance(x, dict):
                out.append(
                    {
                        "text": str(x.get("text", "")).strip(),
                        "missing": str(x.get("missing", "")).strip(),
                        "expected": str(x.get("expected", "")).strip(),
                        "jd_alignment": str(x.get("jd_alignment", "")).strip(),
                        "improvement": str(x.get("improvement", "")).strip(),
                    }
                )
            elif isinstance(x, str) and x.strip():
                out.append(
                    {
                        "text": x.strip(),
                        "missing": "",
                        "expected": "",
                        "jd_alignment": "",
                        "improvement": "",
                    }
                )
        return out

    strengths_list = _strength_items(strengths_raw)
    gaps_list = _gap_items(gaps_raw)

    rubric_score_models = [RubricScoreItem(**x) for x in rubric_scores_payload]

    evaluation_json_payload = {
        "evaluation_mode": body.mode,
        "score": llm_score,
        "summary": eval_summary,
        "score_reasoning": score_reasoning,
        "strengths": strengths_list,
        "gaps": gaps_list,
        "citations": citations_raw,
        "improved_answer": str(evaluation.get("improved_answer") or "").strip(),
        "rubric_scores": rubric_scores_payload,
    }

    feedback_json = {
        "score": llm_score,
        "summary": eval_summary,
        "score_reasoning": score_reasoning,
        "strengths": strengths_list,
        "gaps": gaps_list,
        "citations": citations_raw,
        "strengths_cited": evaluation.get("strengths_cited", []),
        "gaps_cited": evaluation.get("gaps_cited", []),
        "improved_answer": str(evaluation.get("improved_answer") or ""),
        "follow_up_questions": follow_ups,
        "suggested_followup": evaluation.get("suggested_followup"),
        "evidence_used": evaluation.get("evidence_used", []),
        "score_breakdown": score_breakdown,
        "llm_score_0_10": llm_score,
        "rubric_scores": rubric_scores_payload,
    }

    answer = InterviewAnswer(
        question_id=body.question_id,
        answer_text=body.answer_text,
        score=final_score,
        feedback_summary=feedback_summary,
        strengths=strengths_list,
        weaknesses=gaps_list,
        feedback_json=feedback_json,
        evaluation_json=evaluation_json_payload,
    )
    db.add(answer)
    await db.flush()

    ar_profile = await db.execute(
        select(InterviewAnswer, InterviewQuestion)
        .join(InterviewQuestion, InterviewAnswer.question_id == InterviewQuestion.id)
        .where(InterviewQuestion.session_id == session.id)
        .order_by(InterviewAnswer.created_at.asc())
    )
    profile_inputs = [
        profile_answer_from_feedback(q.type, a.feedback_json) for a, q in ar_profile.all()
    ]
    session.performance_profile = compute_performance_profile(profile_inputs)

    await db.commit()
    await db.refresh(answer)

    evidence_used = evaluation.get("evidence_used", [])
    strengths_cited = evaluation.get("strengths_cited", [])
    gaps_cited = evaluation.get("gaps_cited", [])

    def _to_cited(item: dict) -> CitedItem:
        cites = []
        for c in item.get("citations") or []:
            if isinstance(c, dict) and c.get("chunkId"):
                cites.append(
                    CitationItem(
                        chunkId=str(c["chunkId"]),
                        page=c.get("page"),
                        sourceTitle=str(c.get("sourceTitle", "")),
                        sourceType=str(c.get("sourceType", "jd")),
                    )
                )
        return CitedItem(text=str(item.get("text", "")), citations=cites)

    citation_models = [
        EvaluationCitationOut(
            chunk_id=str(c.get("chunk_id") or c.get("chunkId") or ""),
            page_number=int(c.get("page_number", 0) or 0),
            text=str(c.get("text", "")),
        )
        for c in citations_raw
        if isinstance(c, dict) and (c.get("chunk_id") or c.get("chunkId"))
    ]

    return InterviewEvaluateOutput(
        answer_id=answer.id,
        evaluation_mode=body.mode,
        usage=EvaluationUsageOut(
            plan=plan_key,
            evaluations_used_this_month=used,
            evaluation_limit=eval_limit,
        ),
        score=final_score,
        llm_score=llm_score,
        summary=eval_summary,
        score_reasoning=score_reasoning,
        score_breakdown=ScoreBreakdownOut(
            relevance_to_context=score_breakdown["relevance_to_context"],
            completeness=score_breakdown["completeness"],
            clarity=score_breakdown["clarity"],
            jd_alignment=score_breakdown["jd_alignment"],
            overall=score_breakdown["overall"],
        ),
        feedback_summary=feedback_summary,
        strengths=[StrengthEvalItem(**s) for s in strengths_list],
        gaps=[GapEvalItem(**g) for g in gaps_list],
        citations=citation_models,
        strengths_cited=[_to_cited(s) for s in strengths_cited if isinstance(s, dict)],
        gaps_cited=[_to_cited(g) for g in gaps_cited if isinstance(g, dict)],
        improved_answer=str(evaluation.get("improved_answer") or ""),
        follow_up_questions=follow_ups,
        suggested_followup=evaluation.get("suggested_followup"),
        evidence_used=[EvidenceUsedItem(**e) for e in evidence_used],
        rubric_scores=rubric_score_models,
        evaluation_json=evaluation_json_payload,
    )
