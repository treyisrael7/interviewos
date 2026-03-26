"""Shared parsing helpers for interview rubrics and API output."""

from app.models import InterviewQuestion
from app.routers.interview.schemas import EvidenceItem, InterviewQuestionOutput, RoleProfileOut


def to_role_profile_out(rp: dict | None) -> RoleProfileOut | None:
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


def from_rubric(rubric: dict) -> tuple[list[str], list[dict], list[str], str, list[str], str | None, str | None, list[str]]:
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


def competency_key_for_question(q: InterviewQuestion) -> tuple[str | None, str]:
    _, _, _, focus_area, _, comp_id, comp_label, _ = from_rubric(q.rubric_json)
    label = (comp_label or focus_area or "").strip() or "General"
    cid = str(comp_id) if comp_id is not None else None
    return cid, label


def norm_question_type_for_api(t: str) -> str:
    if t == "technical":
        return "role_specific"
    return t


def question_to_output(q: InterviewQuestion) -> InterviewQuestionOutput:
    bullets, evidence, key_topics, focus_area, _, comp_id, comp_label, _ = from_rubric(q.rubric_json)
    return InterviewQuestionOutput(
        id=q.id,
        type=norm_question_type_for_api(q.type),
        focus_area=focus_area,
        competency_id=comp_id,
        competency_label=comp_label,
        question=q.question,
        key_topics=key_topics,
        evidence=[EvidenceItem(**e) for e in evidence],
        rubric_bullets=bullets,
    )
