"""Interview Prep: evidence retrieval + domain-aware question generation from job descriptions."""

import asyncio
import copy
import difflib
import json
import logging
import re
import uuid

try:
    from json_repair import loads as json_repair_loads
except ImportError:
    json_repair_loads = None  # type: ignore[misc, assignment]

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.llm_cache import (
    cache_get_json,
    cache_set_json,
    evaluation_cache_key,
    retrieval_cache_key,
    session_pool_get,
    session_pool_set,
)
from app.models import Document, InterviewAnswer, InterviewQuestion, InterviewSession, InterviewSource
from app.services.adaptive_engine import select_next_question_type
from app.services.jd_sections import normalize_jd_text
from app.services.retrieval import embed_query, retrieve_chunks

logger = logging.getLogger(__name__)

# Domain-specific guidance for role_specific questions
DOMAIN_ROLE_SPECIFIC_GUIDANCE: dict[str, str] = {
    "technical": "Tools, architecture, systems design, technologies, technical problem-solving",
    "finance": "Analysis, risk assessment, markets, accounting concepts, decision-making under uncertainty",
    "healthcare_social_work": "Ethics, boundaries, documentation, crisis handling, advocacy, client-centered care",
    "sales_marketing": "Funnel, experimentation, metrics, messaging, conversion, campaigns",
    "operations": "Process improvement, prioritization, stakeholders, KPIs, efficiency",
    "education": "Classroom management, differentiated instruction, assessment, student support, curriculum",
    "general_business": "Core competencies, collaboration, decision-making, communication",
}

# Default role profile when missing
DEFAULT_ROLE_PROFILE = {
    "domain": "general_business",
    "seniority": "entry",
    "focusAreas": ["communication", "problem solving"],
    "questionMix": {"behavioral": 40, "roleSpecific": 30, "scenario": 30},
}

INTERVIEW_EVIDENCE_TOP_K = 18
COMPETENCY_EVIDENCE_TOP_K = 6
# Top JD chunks merged into evaluation context (after rubric/auxiliary evidence).
EVALUATION_QUERY_TOP_K = 12
# Broad JD pool per interview session; per-question picks top_k via lexical re-rank (no extra vector query).
SESSION_JD_POOL_TOP_K = 32
VALID_QUESTION_TYPES = frozenset({"behavioral", "role_specific", "scenario"})
USER_RESUME_DOC_DOMAIN = "user_resume"


async def get_user_resume_document_id(db: AsyncSession, user_id: uuid.UUID) -> uuid.UUID | None:
    """Return document_id of user's account-level resume, or None if none."""
    r = await db.execute(
        select(Document.id).where(
            Document.user_id == user_id,
            Document.doc_domain == USER_RESUME_DOC_DOMAIN,
        )
    )
    row = r.scalar_one_or_none()
    return row if row is not None else None


async def retrieve_interview_evidence(
    db: AsyncSession,
    document_id: uuid.UUID,
    role_profile: dict | None = None,
    mode: str | None = None,
    source_types: list[str] | None = None,
) -> list[dict]:
    """
    Retrieve evidence chunks from job description. Uses role_profile.focusAreas when available,
    else falls back to mode-based retrieval for backward compat.
    When source_types is None, retrieves from all sources (default).
    Returns list of {chunk_id, page_number, snippet, sourceType, sourceTitle} for citations.
    """
    if role_profile:
        focus_areas = role_profile.get("focusAreas") or DEFAULT_ROLE_PROFILE["focusAreas"]
        query = " ".join(focus_areas) + " responsibilities qualifications role requirements"
        section_types = ["responsibilities", "qualifications", "tools", "about"]
    elif mode:
        config = _MODE_CONFIG.get(mode)
        if not config:
            raise ValueError(f"Invalid mode: {mode}")
        section_types, query = config
    else:
        section_types = ["responsibilities", "qualifications", "tools", "about"]
        query = "job responsibilities qualifications tools technologies about company role"

    query_embedding = embed_query(query)
    chunks = await retrieve_chunks(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        query_text=query,
        top_k=min(INTERVIEW_EVIDENCE_TOP_K, settings.top_k_max * 2),
        include_low_signal=False,
        section_types=section_types,
        doc_domain="job_description",
        source_types=source_types,
    )

    # Fallback: if no chunks match section filter, retry without section filter
    if not chunks:
        chunks = await retrieve_chunks(
            db=db,
            document_id=document_id,
            query_embedding=query_embedding,
            query_text=query,
            top_k=min(INTERVIEW_EVIDENCE_TOP_K, settings.top_k_max * 2),
            include_low_signal=False,
            section_types=None,
            doc_domain="job_description",
            source_types=source_types,
        )

    return [
        {
            "chunk_id": c["chunk_id"],
            "page_number": c["page_number"],
            "snippet": c["snippet"],
            "sourceType": c.get("sourceType", "jd"),
            "sourceTitle": c.get("sourceTitle", ""),
            "retrieval_source": c.get("retrieval_source"),
            "semantic_score": c.get("semantic_score"),
            "keyword_score": c.get("keyword_score"),
            "final_score": c.get("final_score"),
        }
        for c in chunks
    ]


_MODE_CONFIG: dict[str, tuple[list[str], str]] = {
    "technical": (
        ["responsibilities", "qualifications", "tools"],
        "key responsibilities qualifications required skills tools technologies",
    ),
    "behavioral": (
        ["responsibilities", "about"],
        "responsibilities role description about company culture",
    ),
    "mixed": (
        ["responsibilities", "qualifications", "tools", "about"],
        "job responsibilities qualifications tools technologies about company role",
    ),
    "role_driven": (
        ["responsibilities", "qualifications", "tools", "about"],
        "job responsibilities qualifications tools technologies about company role",
    ),
}


async def _retrieve_evidence_for_competency(
    db: AsyncSession,
    document_id: uuid.UUID,
    competency_label: str,
) -> list[dict]:
    """Retrieve evidence for a competency label. sourceTypes=['jd'], topK=6."""
    query_embedding = embed_query(competency_label)
    chunks = await retrieve_chunks(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        query_text=competency_label,
        top_k=COMPETENCY_EVIDENCE_TOP_K,
        include_low_signal=False,
        section_types=None,
        doc_domain="job_description",
        source_types=["jd"],
    )
    return [_retrieval_dict_to_evidence_item(c) for c in chunks]


def _retrieval_dict_to_evidence_item(c: dict) -> dict:
    """Map a retrieve_chunks row into an interview evidence dict (snippet + section_type + text)."""
    cid = c.get("chunk_id") or c.get("chunkId") or ""
    raw = (c.get("text") or c.get("snippet") or "").strip()
    st = c.get("section_type")
    section_type = str(st).strip() if st not in (None, "") else "general"
    pg = c.get("page_number") if c.get("page_number") is not None else c.get("page")
    try:
        page_num = int(pg) if pg is not None else 0
    except (TypeError, ValueError):
        page_num = 0
    return {
        "chunk_id": cid,
        "chunkId": cid,
        "page_number": page_num,
        "page": page_num,
        "text": raw,
        "snippet": raw,
        "section_type": section_type,
        "sourceType": c.get("sourceType", "jd"),
        "sourceTitle": c.get("sourceTitle", ""),
        "retrieval_source": c.get("retrieval_source"),
        "semantic_score": c.get("semantic_score"),
        "keyword_score": c.get("keyword_score"),
        "final_score": c.get("final_score"),
    }


def _session_pool_query_text(role_profile: dict | None) -> str:
    """Stable, broad query for one vector retrieval per session (JD pool)."""
    rp = role_profile or {}
    focus = rp.get("focusAreas") or []
    parts = [
        " ".join(str(x) for x in focus if x),
        str(rp.get("domain") or ""),
        str(rp.get("seniority") or ""),
        "job responsibilities qualifications requirements skills experience role",
    ]
    return " ".join(p.strip() for p in parts if p and str(p).strip()).strip() or (
        "job description responsibilities qualifications requirements"
    )


def _token_overlap_score(query: str, doc_text: str) -> float:
    qt = set(re.findall(r"[a-zA-Z0-9]+", (query or "").lower()))
    ct = set(re.findall(r"[a-zA-Z0-9]+", (doc_text or "").lower()))
    if not qt or not ct:
        return 0.0
    inter = len(qt & ct)
    union = len(qt | ct) or 1
    return inter / union


def _rank_pool_for_query(pool: list[dict], query_text: str, top_k: int) -> list[dict]:
    """
    Pick top_k chunks from a session pool by lexical overlap with query_text, blended with
    retrieval final_score from the pool build (no extra vector DB query).
    """
    if not pool or top_k <= 0:
        return []
    scored: list[tuple[float, int, dict]] = []
    for i, item in enumerate(pool):
        if not isinstance(item, dict):
            continue
        text = (item.get("text") or item.get("snippet") or "").strip()
        lex = _token_overlap_score(query_text, text)
        fs = item.get("final_score")
        try:
            fs_f = float(fs) if fs is not None else 0.0
        except (TypeError, ValueError):
            fs_f = 0.0
        fs_f = max(0.0, min(1.0, fs_f))
        combined = 0.62 * lex + 0.38 * fs_f
        scored.append((combined, -i, item))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [dict(t[2]) for t in scored[:top_k]]


async def _get_or_create_session_jd_pool(
    db: AsyncSession,
    document_id: uuid.UUID,
    session_id: uuid.UUID,
    role_profile: dict | None,
) -> list[dict]:
    """
    Cached top-K JD chunks for the session (one embed + retrieve per session until TTL expires).
    """
    cached = await asyncio.to_thread(session_pool_get, session_id)
    if isinstance(cached, list) and cached:
        return cached

    query_text = _session_pool_query_text(role_profile)
    query_embedding = embed_query(query_text)
    top_k = min(SESSION_JD_POOL_TOP_K, settings.top_k_max * 4)
    chunks = await retrieve_chunks(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=top_k,
        include_low_signal=False,
        section_types=None,
        doc_domain="job_description",
        source_types=["jd"],
    )
    result = [_retrieval_dict_to_evidence_item(c) for c in chunks]
    if result:
        await asyncio.to_thread(session_pool_set, session_id, result)
    return result


def _normalize_evaluation_chunk(e: dict) -> dict:
    """
    Canonical shape for evaluate_answer / citation generation:
    chunk_id, text, page_number, section_type; snippet mirrors text for scoring helpers.
    """
    cid = str(e.get("chunk_id") or e.get("chunkId") or "").strip()
    raw_text = (e.get("text") or e.get("snippet") or "").strip()
    pg = e.get("page_number") if e.get("page_number") is not None else e.get("page")
    try:
        page_number = int(pg) if pg is not None else 0
    except (TypeError, ValueError):
        page_number = 0
    st = e.get("section_type")
    section_type = str(st).strip() if st not in (None, "") else "general"
    return {
        "chunk_id": cid,
        "chunkId": cid,
        "text": raw_text,
        "snippet": raw_text,
        "page_number": page_number,
        "page": page_number,
        "section_type": section_type,
        "sourceType": str(e.get("sourceType", "jd")),
        "sourceTitle": str(e.get("sourceTitle", "")),
        **(
            {k: e[k] for k in ("retrieval_source", "semantic_score", "keyword_score", "final_score") if k in e}
        ),
    }


def normalize_evaluation_evidence(evidence: list[dict]) -> list[dict]:
    """Normalize rubric/retrieved chunks before the evaluation LLM and citation parsing."""
    out: list[dict] = []
    for e in evidence or []:
        if not isinstance(e, dict):
            continue
        n = _normalize_evaluation_chunk(e)
        if not n["text"].strip():
            continue
        out.append(n)
    return out


_ADAPTIVE_TO_CANONICAL_TYPE: dict[str, str] = {
    "technical": "role_specific",
    "behavioral": "behavioral",
    "behavioral_followup": "behavioral",
    "hard": "scenario",
}


def _prompt_question_type(canonical_type: str, adaptive_target: str | None) -> str:
    """
    User-facing question type label injected into the generation prompt.
    Matches adaptive_engine outputs; maps mix-based canonical types when not adaptive.
    """
    if adaptive_target in ("technical", "behavioral", "behavioral_followup", "hard"):
        return adaptive_target
    if canonical_type == "behavioral":
        return "behavioral"
    if canonical_type == "role_specific":
        return "technical"
    if canonical_type == "scenario":
        return "hard"
    return "behavioral"


def _feedback_line_from_item(x: object) -> str:
    """Normalize one strength or gap entry (string or {{text, ...}}) to a line of text."""
    if isinstance(x, dict):
        t = str(x.get("text", "")).strip()
        if t:
            return t
        return ""
    return str(x).strip() if x else ""


def _feedback_from_answer_row(answer: InterviewAnswer) -> dict[str, list[str]]:
    """Normalize strengths/gaps from stored answer columns and feedback_json."""
    strengths: list[str] = []
    gaps: list[str] = []
    if isinstance(answer.strengths, list):
        strengths = [_feedback_line_from_item(x) for x in answer.strengths if _feedback_line_from_item(x)]
    if isinstance(answer.weaknesses, list):
        gaps = [_feedback_line_from_item(x) for x in answer.weaknesses if _feedback_line_from_item(x)]
    fj = answer.feedback_json if isinstance(answer.feedback_json, dict) else {}
    if not strengths:
        s = fj.get("strengths")
        if isinstance(s, list):
            strengths = [_feedback_line_from_item(x) for x in s if _feedback_line_from_item(x)]
    if not gaps:
        g = fj.get("gaps")
        if isinstance(g, list):
            gaps = [_feedback_line_from_item(x) for x in g if _feedback_line_from_item(x)]
    return {"strengths": strengths, "gaps": gaps}


def _format_feedback_lines(items: list[str]) -> str:
    if not items:
        return "(none recorded)"
    return "; ".join(items)


def _generate_single_question(
    competency_id: str,
    competency_label: str,
    canonical_question_type: str,
    evidence: list[dict],
    role_profile: dict,
    adaptive_target: str | None = None,
    last_answer_feedback: dict[str, list[str]] | None = None,
) -> dict | None:
    """Generate one question for a competency using its evidence. Returns None if no evidence.

    ``canonical_question_type`` is stored on the question row (behavioral | role_specific | scenario).
    The prompt injects ``question_type`` (technical | behavioral | behavioral_followup | hard) via
    :func:`_prompt_question_type`.

    For ``behavioral_followup``, ``last_answer_feedback`` may contain ``strengths`` and ``gaps`` from
    the most recent evaluated answer in the session.
    """
    if not evidence or not settings.openai_api_key:
        return None

    from app.services.jd_sections import normalize_jd_text

    excerpt_lines = []
    for i, e in enumerate(evidence):
        snippet = normalize_jd_text(e.get("snippet", "")).strip()
        excerpt_lines.append(f"[{i}] {snippet}")

    excerpts_text = "\n\n".join(excerpt_lines)
    domain = role_profile.get("domain", "general_business")
    seniority = role_profile.get("seniority", "entry")
    seniority_hint = "entry-level" if seniority == "entry" else "mid-level" if seniority == "mid" else "senior-level"

    question_type = _prompt_question_type(canonical_question_type, adaptive_target)

    followup_context = ""
    user_followup = ""
    if question_type == "behavioral_followup":
        fb = last_answer_feedback or {}
        strengths_items = [str(x).strip() for x in (fb.get("strengths") or []) if str(x).strip()]
        gaps_items = [str(x).strip() for x in (fb.get("gaps") or []) if str(x).strip()]
        strengths_txt = _format_feedback_lines(strengths_items)
        gaps_txt = _format_feedback_lines(gaps_items)
        followup_context = f"""

PRIOR ANSWER FEEDBACK (target the follow-up):
Strengths noted: {strengths_txt}

The candidate previously showed weaknesses in: {gaps_txt}

Generate a follow-up question that specifically probes this weakness."""
        user_followup = """

The follow-up must directly address the weaknesses above; if none were recorded, infer a plausible gap from the competency and JD and still probe deeply."""

    system_prompt = f"""You are generating an interview question.

Question type: {question_type}

- If technical: focus on role-specific technical skills
- If behavioral: ask STAR-style questions
- If behavioral_followup: ask a deeper probing follow-up question based on previous answer weaknesses
- If hard: increase difficulty, ambiguity, or depth

Ground all questions in the provided job description context.

Competency: "{competency_label}"
SENIORITY: {seniority} ({seniority_hint})
DOMAIN: {domain}{followup_context}

RULES:
1. Follow the question type above; shape the line of questioning as: behavioral / role-specific skills / scenario as appropriate to that type.
2. The question MUST be grounded in the job description excerpts in the user message.
3. whatGoodLooksLike: 2-4 bullets describing what a strong answer covers.
4. Output valid JSON only: {{"question": "...", "whatGoodLooksLike": ["...","..."]}}"""

    user_content = f"""Competency: {competency_label}

Job description context:
{excerpts_text}

Generate exactly one interview question per the system instructions (question type: {question_type}).{user_followup}
Output JSON only."""

    try:
        from openai import OpenAI

        # Single-question generation: MODEL_FAST only (cheap path).
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.model_fast,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=400,
        )
        raw = (response.choices[0].message.content or "").strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            raw = json_match.group(0)
        data = json.loads(raw)
        question_text = str(data.get("question", "")).strip()
        what_good = data.get("whatGoodLooksLike")
        if isinstance(what_good, list):
            what_good = [str(x).strip() for x in what_good if x]
        else:
            what_good = []

        evidence_for_q = [
            {
                "chunk_id": str(e.get("chunk_id") or e.get("chunkId") or ""),
                "page_number": int(e.get("page_number") or e.get("page") or 0),
                "snippet": str(e.get("snippet", "")),
                "sourceType": e.get("sourceType", "jd"),
                "sourceTitle": e.get("sourceTitle", ""),
                "retrieval_source": e.get("retrieval_source"),
                "semantic_score": e.get("semantic_score"),
                "keyword_score": e.get("keyword_score"),
                "final_score": e.get("final_score"),
            }
            for e in evidence
        ]

        return {
            "type": canonical_question_type,
            "competencyId": competency_id,
            "competencyLabel": competency_label,
            "questionText": question_text,
            "whatGoodLooksLike": what_good,
            "evidence": evidence_for_q,
            "evidenceChunkIds": [str(e.get("chunk_id") or e.get("chunkId", "")) for e in evidence],
        }
    except Exception as e:
        logger.warning("_generate_single_question failed for %s: %s", competency_label, e)
        return None


async def generate_questions(
    db: AsyncSession,
    document_id: uuid.UUID,
    num_questions: int,
    role_profile: dict,
    competencies: list[dict],
    session_id: uuid.UUID | None = None,
) -> list[dict]:
    """
    Generate questions tied to competencies. Each question uses retrieval for that competency.

    Args:
        db: AsyncSession
        document_id: Document UUID
        num_questions: Number of questions to generate
        role_profile: { domain, seniority, questionMix }
        competencies: List of { id, label, description?, evidence? }
        session_id: When set, loads ``InterviewSession.performance_profile``; if present,
            :func:`app.services.adaptive_engine.select_next_question_type` drives each
            question (otherwise question mix uses ``questionMix`` as before). Also loads the
            latest ``InterviewAnswer`` in that session so ``behavioral_followup`` can use
            strengths/gaps from the prior evaluation.

    Returns list of {
        id (placeholder, assigned by DB),
        type, competencyId, competencyLabel, questionText, whatGoodLooksLike,
        evidence, evidenceChunkIds
    }
    """
    rp = role_profile or DEFAULT_ROLE_PROFILE.copy()
    qmix = rp.get("questionMix") or DEFAULT_ROLE_PROFILE["questionMix"]
    b = int(qmix.get("behavioral", 40))
    r = int(qmix.get("roleSpecific", 30))
    s = int(qmix.get("scenario", 30))
    total = b + r + s
    if total != 100 and total > 0:
        b, r, s = round(100 * b / total), round(100 * r / total), round(100 * s / total)
        s = 100 - b - r
    type_cycle = []
    for _ in range(b):
        type_cycle.append("behavioral")
    for _ in range(r):
        type_cycle.append("role_specific")
    for _ in range(s):
        type_cycle.append("scenario")

    # Use focusAreas as pseudo-competencies if competencies empty
    comps = competencies
    if not comps:
        focus_areas = rp.get("focusAreas") or DEFAULT_ROLE_PROFILE["focusAreas"]
        comps = [{"id": f"fa-{i}", "label": str(fa)} for i, fa in enumerate(focus_areas[:num_questions])]
    if not comps:
        return []

    # Limit competencies to generate from
    comps = comps[: num_questions] if len(comps) > num_questions else comps
    results = []

    performance_profile_dict: dict | None = None
    last_answer_feedback: dict[str, list[str]] | None = None
    if session_id is not None:
        sess_row = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
        sess_obj = sess_row.scalar_one_or_none()
        if sess_obj is not None:
            raw_profile = getattr(sess_obj, "performance_profile", None)
            if isinstance(raw_profile, dict) and raw_profile:
                performance_profile_dict = raw_profile

            ans_row = await db.execute(
                select(InterviewAnswer)
                .join(InterviewQuestion, InterviewAnswer.question_id == InterviewQuestion.id)
                .where(InterviewQuestion.session_id == session_id)
                .order_by(InterviewAnswer.created_at.desc())
                .limit(1)
            )
            last_ans = ans_row.scalar_one_or_none()
            if last_ans is not None:
                last_answer_feedback = _feedback_from_answer_row(last_ans)

    for i, comp in enumerate(comps):
        if len(results) >= num_questions:
            break
        comp_id = str(comp.get("id", f"comp-{i}"))
        comp_label = str(comp.get("label", "")).strip()
        if not comp_label:
            continue

        # Retrieve evidence for this competency
        evidence = await _retrieve_evidence_for_competency(db, document_id, comp_label)
        if not evidence:
            logger.info("No evidence for competency=%s, skipping", comp_label)
            continue

        adaptive_target: str | None = None
        if performance_profile_dict is not None:
            adaptive_target = select_next_question_type(performance_profile_dict)
            q_type = _ADAPTIVE_TO_CANONICAL_TYPE.get(adaptive_target)
            if not q_type:
                q_type = type_cycle[i % len(type_cycle)] if type_cycle else "behavioral"
                adaptive_target = None
        else:
            q_type = type_cycle[i % len(type_cycle)] if type_cycle else "behavioral"

        q = _generate_single_question(
            competency_id=comp_id,
            competency_label=comp_label,
            canonical_question_type=q_type,
            evidence=evidence,
            role_profile=rp,
            adaptive_target=adaptive_target,
            last_answer_feedback=last_answer_feedback,
        )
        if q and q.get("questionText"):
            results.append(q)

    return results[:num_questions]


def _build_domain_aware_prompt(
    role_profile: dict,
    evidence: list[dict],
    num_questions: int,
) -> tuple[str, str]:
    """Build system and user prompts for domain-aware question generation."""
    excerpt_lines = []
    for i, e in enumerate(evidence):
        snippet = normalize_jd_text(e.get("snippet", "")).strip()
        excerpt_lines.append(f"[{i}] {snippet}")

    excerpts_text = "\n\n".join(excerpt_lines)

    domain = role_profile.get("domain", "general_business")
    seniority = role_profile.get("seniority", "entry")
    focus_areas = role_profile.get("focusAreas") or []
    question_mix = role_profile.get("questionMix") or {}

    b = int(question_mix.get("behavioral", 40))
    r = int(question_mix.get("roleSpecific", 30))
    s = int(question_mix.get("scenario", 30))
    total = b + r + s
    if total != 100 and total > 0:
        b, r, s = round(100 * b / total), round(100 * r / total), round(100 * s / total)
        s = 100 - b - r

    role_specific_hint = DOMAIN_ROLE_SPECIFIC_GUIDANCE.get(domain, DOMAIN_ROLE_SPECIFIC_GUIDANCE["general_business"])
    seniority_hint = "entry-level" if seniority == "entry" else "mid-level" if seniority == "mid" else "senior-level"

    focus_areas_text = ", ".join(focus_areas) if focus_areas else "communication, problem solving"

    system_prompt = f"""You are an expert hiring manager creating interview questions from a job description.
Generate questions that are DOMAIN-AWARE and tailored to the role.

DOMAIN: {domain}
SENIORITY: {seniority} ({seniority_hint})
FOCUS AREAS (each question must map to ONE): {focus_areas_text}

QUESTION TYPES:
- behavioral: Past experience, STAR format, "Tell me about a time..."
- role_specific: Domain knowledge for this role. For {domain}, cover: {role_specific_hint}
- scenario: Hypothetical "What would you do if..." situations

QUESTION MIX (approximate): behavioral {b}%, role_specific {r}%, scenario {s}%

RULES:
1. Each question MUST map to exactly one focusArea from the list above.
2. Scale depth and difficulty by seniority ({seniority_hint}).
3. Ground questions in the evidence excerpts; cite by index [0], [1], etc. in evidence_indices.
4. whatGoodLooksLike: 2-4 bullets describing what a strong answer covers.
5. mustMention: 0-3 bullets of key points the answer should reference (optional, can be empty).

Output valid JSON only, no markdown:
{{"questions": [{{"type": "behavioral"|"role_specific"|"scenario", "focusArea": "...", "question": "...", "whatGoodLooksLike": ["...","..."], "mustMention": ["..."]|[], "evidence_indices": [0,1]}}]}}"""

    user_content = f"""Job description excerpts (each prefixed with index):
{excerpts_text}

Generate exactly {num_questions} questions. Follow the question mix and map each to a focus area.
Output JSON only."""

    return system_prompt, user_content


def _parse_domain_aware_questions(raw: str, evidence: list[dict]) -> list[dict]:
    """Parse LLM JSON output into structured questions."""
    text = raw.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    data = json.loads(text)
    questions = data.get("questions", [])
    result = []

    def _to_str_list(v) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if x]

    for q in questions:
        qtype = str(q.get("type", "")).strip().lower()
        if qtype not in VALID_QUESTION_TYPES:
            qtype = "behavioral"

        idx_list = q.get("evidence_indices") or []
        if not isinstance(idx_list, list):
            idx_list = []
        evidence_for_q = []
        for idx in idx_list:
            if isinstance(idx, int) and 0 <= idx < len(evidence):
                evidence_for_q.append(evidence[idx])
        if not evidence_for_q and evidence:
            evidence_for_q = evidence[:3]

        what_good = _to_str_list(q.get("whatGoodLooksLike") or q.get("rubric") or [])
        must_mention = _to_str_list(q.get("mustMention") or [])
        focus_area = str(q.get("focusArea") or "").strip()

        result.append({
            "type": qtype,
            "focusArea": focus_area,
            "question": str(q.get("question", "")).strip(),
            "whatGoodLooksLike": what_good,
            "mustMention": must_mention,
            "evidence": evidence_for_q,
        })

    return result


def generate_interview_questions(
    role_profile: dict,
    retrieved_evidence_chunks: list[dict],
    num_questions: int,
) -> list[dict]:
    """
    Generate interview questions using roleProfile (domain-aware).

    Args:
        role_profile: { domain, seniority, focusAreas, questionMix }
        retrieved_evidence_chunks: list of {chunk_id, page_number, snippet}
        num_questions: number of questions to generate

    Returns list of {
        type: "behavioral"|"role_specific"|"scenario",
        focusArea: str,
        question: str,
        whatGoodLooksLike: list[str],
        mustMention: list[str],
        evidence: list[dict]
    }
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    rp = role_profile or DEFAULT_ROLE_PROFILE.copy()
    system_prompt, user_content = _build_domain_aware_prompt(rp, retrieved_evidence_chunks, num_questions)

    # Batch question generation uses MODEL_FAST only (not MODEL_HIGH_QUALITY).
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.model_fast,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=min(2000, settings.max_completion_tokens * 4),  # need more for structured JSON
    )

    raw = (response.choices[0].message.content or "").strip()
    return _parse_domain_aware_questions(raw, retrieved_evidence_chunks)


# Domain-specific evaluation considerations
EVALUATION_DOMAIN_HINTS: dict[str, str] = {
    "technical": "When relevant, consider: monitoring, data quality, reliability, CI/CD, testing, observability, scalability.",
    "finance": "When relevant, consider: risk controls, compliance, audit trails, regulatory requirements, due diligence.",
    "healthcare_social_work": "When relevant, consider: ethics, confidentiality, professional boundaries, documentation standards, crisis protocols, advocacy.",
    "sales_marketing": "When relevant, consider: metrics, funnel, attribution, experimentation rigor.",
    "operations": "When relevant, consider: process, prioritization, stakeholder alignment, KPIs.",
    "education": "When relevant, consider: student support, differentiated instruction, assessment alignment.",
    "general_business": "When relevant, consider: collaboration, communication, decision-making clarity.",
}


EVALUATION_SYSTEM_PROMPT = """You are an expert interview evaluator.

Evaluate the candidate's answer using the provided job description context.

You MUST:
1. Score the answer (0–10). If the user message includes role-specific dimensions (below), the top-level `score` must be the **unweighted arithmetic mean** of every `rubric_scores[].score` you output (same number as averaging those dimension scores yourself—no separate holistic guess).
2. Write a concise summary (2–3 sentences) explaining why that score was given, grounded in the rubric and answer
3. Write score_reasoning: 1–2 sentences explaining why this score was given, explicitly tying strengths and gaps to rubric expectations
4. Identify strengths. For each strength you MUST include:
   - text: short label for the strength
   - evidence: a direct quote from the candidate’s answer supporting this strength
   - highlight: verbatim substring from the answer (for UI emphasis; copy-paste from the answer)
   - impact: why this strength is valuable for the role (tie explicitly to the job description, rubric, or competency)
5. Identify gaps vs the rubric and job description. For each gap you MUST include:
   - text: what the candidate said (verbatim quote or tight paraphrase from their answer)
   - missing: what is absent or weak in their answer relative to the rubric/JD
   - expected: what the rubric or job description requires (explicitly reference the relevant JD expectation you are scoring against)
   - jd_alignment: explain how the answer does or does not match the job requirement (tie candidate wording to that JD expectation and spell out the mismatch or partial fit)
   - improvement: specific phrasing they should say instead (concrete example sentences)
6. Write improved_answer: rewrite the candidate’s entire answer into a stronger version that would score 9–10/10 on this question
7. Provide citations from the job description chunks
8. Include rubric_scores: when the user message lists role-specific dimensions (JSON), fill `rubric_scores` with one object per dimension: `name` (exact match), `score` (0–10), `reasoning` (why that score). If there are no such dimensions, use `[]`. The top-level `score` (step 1) must match the mean of those per-dimension scores when any are present.

Rules:
- ONLY use provided context (no hallucination)
- Be specific and structured
- Evidence must reference actual answer text
- Each strength MUST include the four fields text, evidence, highlight, and impact (no omissions): evidence must be a quote from the answer; impact must explain why this strength matters for this role (JD/rubric/competency), not generic praise
- Each gap MUST use the five fields: text, missing, expected, jd_alignment, improvement (no omissions)
- In every gap, explicitly reference the relevant JD expectation and explain the mismatch (or gap) vs that expectation; jd_alignment must summarize alignment vs the job requirement
- improved_answer must be a full rewritten answer (not bullet notes): keep the candidate’s original idea and story arc; add missing depth the gaps called out; add concrete tools, technologies, and metrics when relevant to the role/JD; stay realistic and specific to their situation—no generic filler or buzzwords without substance
- For "citations", use only chunk_id values that appear in the provided chunks; quote or paraphrase chunk text faithfully; page_number must match the chunk
- For each rubric_scores[] entry: score must be between 0 and 10 (float); reasoning must justify that score (not generic praise—tie to the candidate's answer and the dimension)

Return JSON in this exact format (no markdown fences, JSON only):
{
  "score": 0.0,
  "summary": "2–3 sentences explaining why this score was given (rubric fit, strengths, gaps).",
  "score_reasoning": "1–2 sentences why this score was given, explicitly tying strengths and gaps to rubric expectations.",
  "strengths": [
    {
      "text": "short label",
      "evidence": "verbatim quote from the candidate answer",
      "highlight": "exact contiguous phrase copied from the candidate answer above",
      "impact": "why this strength is valuable for the role (tie to JD/rubric)"
    }
  ],
  "gaps": [
    {
      "text": "what the candidate said (from their answer)",
      "missing": "what is missing or weak vs rubric/JD",
      "expected": "relevant JD/rubric requirement (cite the expectation explicitly)",
      "jd_alignment": "how the answer does or does not match the job requirement",
      "improvement": "specific phrasing they should say instead"
    }
  ],
  "citations": [
    { "chunk_id": "...", "page_number": 0, "text": "..." }
  ],
  "improved_answer": "Full rewritten answer that would score 9–10/10: same core idea, added depth, tools/metrics where relevant, realistic and specific.",
  "rubric_scores": [
    {
      "name": "dimension name (must match document dimensions when provided)",
      "score": 0.0,
      "reasoning": "Why this dimension received this score (0–10): cite what the answer showed or missed for this dimension."
    }
  ]
}"""


EVALUATION_SYSTEM_PROMPT_LITE = """You are an expert interview evaluator in LITE mode.

Output valid JSON only (no markdown). Be brief. No deep reasoning, no multi-paragraph analysis, no structured strengths/gaps lists.

You MUST:
1. score (0–10). If the user message includes role-specific dimensions, the top-level score must equal the unweighted mean of every rubric_scores[].score you output.
2. summary: at most 2 short sentences — high-level feedback only (why this score band).
3. score_reasoning: at most ONE short sentence — no lists, no step-by-step logic, no mention of "strengths" or "gaps" as sections.
4. rubric_scores: If "Document evaluation dimensions" lists dimensions, include one object per dimension: name (exact match), score (0–10), reasoning (one short phrase only, ≤25 words). If no dimensions, use [].
5. strengths: MUST be the empty array [].
6. gaps: MUST be the empty array [].
7. citations: MUST be the empty array [].
8. improved_answer: MUST be the empty string "".

Rules:
- Do NOT write strengths, gaps, citations, or improved_answer content — keep those fields empty as specified.
- Do NOT produce long reasoning; keep summary and score_reasoning short.
- For rubric_scores reasoning fields, one short phrase per dimension only.

Return JSON in this exact shape:
{
  "score": 0.0,
  "summary": "At most two short sentences.",
  "score_reasoning": "One short sentence max.",
  "strengths": [],
  "gaps": [],
  "citations": [],
  "improved_answer": "",
  "rubric_scores": []
}"""


def _evaluation_system_prompt_for_mode(evaluation_mode: str) -> str:
    m = (evaluation_mode or "full").strip().lower()
    if m == "lite":
        return EVALUATION_SYSTEM_PROMPT_LITE
    return EVALUATION_SYSTEM_PROMPT


def _format_document_rubric_block(document_rubric_json: list[dict] | None) -> str:
    """Human- and model-readable block for JD-level evaluation dimensions (name + description)."""
    if not document_rubric_json:
        return ""
    payload: list[dict[str, str]] = []
    for d in document_rubric_json:
        if not isinstance(d, dict):
            continue
        name = str(d.get("name", "")).strip()
        if not name:
            continue
        payload.append(
            {
                "name": name,
                "description": str(d.get("description", "")).strip(),
            }
        )
    if not payload:
        return ""
    rubric_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return (
        "## Role-specific dimensions\n\n"
        "You must evaluate the candidate based on the following role-specific dimensions:\n\n"
        f"{rubric_json}\n\n"
        "For each dimension:\n"
        "- assign a score (0–10)\n"
        "- explain reasoning\n\n"
        "Then compute an overall score based on these dimensions "
        "(the JSON field `score` must equal the unweighted mean of those per-dimension scores).\n\n"
    )


def _build_domain_aware_evaluation_prompt(
    question: str,
    question_type: str,
    focus_area: str,
    competency_label: str | None,
    what_good_looks_like: list[str],
    must_mention: list[str],
    evidence: list[dict],
    role_profile: dict,
    answer_text: str,
    document_rubric_json: list[dict] | None = None,
    evaluation_mode: str = "full",
) -> tuple[str, str]:
    """Build prompts: system message (full vs lite) + user message with Q, rubric, chunks, answer."""
    rubric_lines = [f"• {b}" for b in (what_good_looks_like or []) if str(b).strip()]
    if must_mention:
        rubric_lines.append("Must mention: " + "; ".join(must_mention))
    rubric_text = "\n".join(rubric_lines) if rubric_lines else "(No separate rubric bullets; rely on job description chunks.)"

    evidence_lines = []
    for i, e in enumerate(evidence or []):
        s = (e.get("text") or e.get("snippet") or "").strip()
        cid = e.get("chunk_id") or e.get("chunkId") or ""
        pg = e.get("page_number") if e.get("page_number") is not None else e.get("page")
        try:
            pnv = int(pg) if pg is not None else 0
        except (TypeError, ValueError):
            pnv = 0
        sec = e.get("section_type")
        section_type = str(sec).strip() if sec not in (None, "") else "general"
        src_type = e.get("sourceType", "jd")
        src_title = e.get("sourceTitle", "")
        if s:
            meta = (
                f"[{i}] chunk_id={cid} section_type={section_type} "
                f"sourceType={src_type} sourceTitle={src_title} page_number={pnv}"
            )
            evidence_lines.append(f"{meta}\n{s[:500]}")
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "(No job description chunks provided.)"

    domain = role_profile.get("domain", "general_business")
    seniority = role_profile.get("seniority", "entry")
    domain_hint = EVALUATION_DOMAIN_HINTS.get(domain, EVALUATION_DOMAIN_HINTS["general_business"])

    header = f"""Session context (use when tuning expectations):
- Question type: {question_type}
- Focus area: {focus_area}
- Competency: {competency_label or "(none)"}
- Role domain: {domain} | Seniority: {seniority}
- Domain note: {domain_hint}

"""

    doc_rubric_block = _format_document_rubric_block(document_rubric_json)

    user_content = f"""{header}## Interview question
{question}

{doc_rubric_block}## Rubric and expected skills (from role / question setup)
{rubric_text}

## Retrieved job description chunks (only valid sources for "citations"; cite these chunk_id values exactly)
{evidence_text}

## Candidate's answer (quote from this text in strengths' evidence where applicable)
{answer_text}
"""

    return _evaluation_system_prompt_for_mode(evaluation_mode), user_content


def _citation_from_evidence(e: dict) -> dict:
    """Build {chunkId, page?, sourceTitle, sourceType} from evidence item."""
    chunk_id = str(e.get("chunk_id") or e.get("chunkId") or "")
    page = e.get("page_number") or e.get("page")
    return {
        "chunkId": chunk_id,
        "page": int(page) if page is not None else None,
        "sourceTitle": str(e.get("sourceTitle", "")),
        "sourceType": str(e.get("sourceType", "jd")),
    }


def _citation_row_from_evidence_index(ev: list[dict], idx: int) -> dict | None:
    """Build {chunk_id, page_number, text} from evidence list index."""
    if not isinstance(idx, int) or idx < 0 or idx >= len(ev):
        return None
    e = ev[idx]
    chunk_id = str(e.get("chunk_id") or e.get("chunkId") or "")
    pg = e.get("page_number") or e.get("page")
    try:
        page_number = int(pg) if pg is not None else 0
    except (TypeError, ValueError):
        page_number = 0
    body = str(e.get("text") or e.get("snippet") or "").strip()
    return {
        "chunk_id": chunk_id,
        "page_number": page_number,
        "text": body[:800],
    }


def _strip_json_markdown_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_balanced_json_object(s: str) -> str | None:
    """First top-level `{...}` with string-aware brace matching (avoids greedy-regex mistakes)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    i = start
    n = len(s)
    in_string = False
    escape = False
    while i < n:
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
        i += 1
    return None


def _remove_trailing_commas_json(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _loads_evaluation_json(raw: str) -> dict:
    """
    Parse model evaluation JSON: strip fences, extract balanced object, then strict parse,
    trailing-comma fix, or json_repair fallback.
    """
    text = _strip_json_markdown_fence(raw)
    extracted = _extract_first_balanced_json_object(text)
    if extracted:
        text = extracted
    text = text.strip()
    last_err: Exception | None = None
    candidates = [text, _remove_trailing_commas_json(text)]
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
            last_err = ValueError(f"Expected JSON object, got {type(data).__name__}")
        except json.JSONDecodeError as e:
            last_err = e
        if json_repair_loads is not None:
            try:
                data = json_repair_loads(candidate)
                if isinstance(data, dict):
                    return data
                last_err = ValueError(f"Expected JSON object, got {type(data).__name__}")
            except Exception as e:  # noqa: BLE001 — lenient parser may raise varied errors
                last_err = e
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("Expected JSON object", text, 0)


def _fallback_evaluation_parse(evidence: list[dict]) -> dict:
    """Minimal valid shape when the model output is not parseable JSON."""
    return {
        "score": 5.0,
        "summary": (
            "The automated evaluation response could not be read (invalid or truncated JSON). "
            "Your answer was saved. Try submitting again, or use a slightly shorter answer if the issue persists."
        ),
        "strengths": [],
        "gaps": [],
        "citations": [],
        "strengths_cited": [],
        "gaps_cited": [],
        "improved_answer": "",
        "follow_up_questions": [],
        "suggested_followup": None,
        "evidence_used": [],
        "score_reasoning": "",
        "rubric_scores": [],
        "_llm_citation_entries": 0,
    }


def _normalize_rubric_scores_list(raw: list | None) -> list[dict[str, float | str]]:
    """
    Canonical rubric_scores: name (str), score (float 0–10 per dimension), reasoning (explains that score).

    Empty or missing reasoning is replaced so downstream consumers always get an explanatory string.
    """
    out: list[dict[str, float | str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        rs = item.get("score")
        try:
            score_f = max(0.0, min(10.0, float(rs)))
        except (TypeError, ValueError):
            score_f = 0.0
        reasoning = str(item.get("reasoning", "")).strip()
        if not reasoning:
            reasoning = (
                f"Assigned {score_f:.1f}/10 for this dimension: the answer's fit to «{name}» "
                "given the question, rubric, and job description context."
            )
        out.append({"name": name, "score": score_f, "reasoning": reasoning})
    return out


# Public alias for routers/tests that need the same canonical shape.
normalize_rubric_scores_output = _normalize_rubric_scores_list


def _dedupe_citations(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("chunk_id", ""))
        if not cid:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        out.append(
            {
                "chunk_id": cid,
                "page_number": int(r.get("page_number", 0) or 0),
                "text": str(r.get("text", "")),
            }
        )
    return out


def _parse_evaluation_response(raw: str, evidence: list[dict]) -> dict:
    """
    Parse LLM evaluation JSON.

    Returns score (0-10), summary, score_reasoning, strengths [{text, evidence, highlight, impact}],
    gaps [{text, missing, expected, jd_alignment, improvement}], citations [{chunk_id, page_number, text}],
    strengths_cited/gaps_cited for API compatibility, evidence_used, and follow-up fields.
    """
    try:
        data = _loads_evaluation_json(raw)
    except Exception as e:
        logger.warning(
            "evaluation JSON parse failed: %s; raw_snippet=%r",
            e,
            (raw or "")[:1500],
        )
        return _fallback_evaluation_parse(evidence)

    def _indices_for(item: dict) -> list[int]:
        idx_list = item.get("citation_indices")
        if not isinstance(idx_list, list):
            return []
        out: list[int] = []
        for idx in idx_list:
            if isinstance(idx, int) and 0 <= idx < len(evidence):
                out.append(idx)
        return out

    def _parse_cited_strengths_gaps(
        raw_strengths: list[dict] | list[str],
        raw_gaps: list[dict] | list[str],
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """Return strengths_out, gaps_out, strengths_cited, gaps_cited."""
        s_list = raw_strengths if isinstance(raw_strengths, list) else []
        g_list = raw_gaps if isinstance(raw_gaps, list) else []
        if s_list and isinstance(s_list[0], str):
            s_list = [
                {"text": s, "evidence": "", "citation_indices": [], "highlight": "", "impact": ""}
                for s in s_list
            ]
        if g_list and isinstance(g_list[0], str):
            g_list = [
                {
                    "text": g,
                    "missing": "",
                    "expected": "",
                    "jd_alignment": "",
                    "improvement": "",
                    "citation_indices": [],
                }
                for g in g_list
            ]

        strengths_out: list[dict] = []
        gaps_out: list[dict] = []
        strengths_cited: list[dict] = []
        gaps_cited: list[dict] = []

        for item in s_list:
            if not isinstance(item, dict):
                continue
            t = str(item.get("text", "")).strip()
            ev_txt = str(item.get("evidence", "")).strip()
            if not ev_txt and isinstance(item.get("citation_indices"), list):
                # Legacy: only indices, no evidence string
                parts = []
                for idx in _indices_for(item):
                    row = _citation_row_from_evidence_index(evidence, idx)
                    if row:
                        parts.append(row["text"])
                ev_txt = "; ".join(parts) if parts else "JD does not specify"
            hl_raw = str(item.get("highlight", "")).strip()
            impact = str(item.get("impact", "")).strip()
            strengths_out.append(
                {
                    "text": t,
                    "evidence": ev_txt or "JD does not specify",
                    "highlight": hl_raw,
                    "impact": impact,
                }
            )
            idx_list = item.get("citation_indices")
            if not isinstance(idx_list, list):
                idx_list = []
            citations = []
            for idx in idx_list:
                if isinstance(idx, int) and 0 <= idx < len(evidence):
                    citations.append(_citation_from_evidence(evidence[idx]))
            strengths_cited.append({"text": t, "citations": citations})

        for item in g_list:
            if not isinstance(item, dict):
                continue
            t = str(item.get("text", "")).strip()
            missing = str(item.get("missing", "")).strip()
            exp = str(item.get("expected", "")).strip()
            jd_alignment = str(item.get("jd_alignment", "")).strip()
            improvement = str(item.get("improvement", "")).strip()
            if not exp:
                exp = "Align with rubric and JD evidence."
            gaps_out.append(
                {
                    "text": t,
                    "missing": missing,
                    "expected": exp,
                    "jd_alignment": jd_alignment,
                    "improvement": improvement,
                }
            )
            idx_list = item.get("citation_indices")
            if not isinstance(idx_list, list):
                idx_list = []
            citations = []
            for idx in idx_list:
                if isinstance(idx, int) and 0 <= idx < len(evidence):
                    citations.append(_citation_from_evidence(evidence[idx]))
            gaps_cited.append({"text": t, "citations": citations})

        return strengths_out, gaps_out, strengths_cited, gaps_cited

    raw_strengths = data.get("strengths") or []
    raw_gaps = data.get("gaps") or []
    strengths, gaps, strengths_cited, gaps_cited = _parse_cited_strengths_gaps(raw_strengths, raw_gaps)

    # Top-level citations: LLM optional list + union of all cited indices
    citation_rows: list[dict] = []
    raw_citations = data.get("citations")
    if isinstance(raw_citations, list):
        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("chunk_id") or c.get("chunkId") or "")
            if not cid:
                continue
            allowed = {str(e.get("chunk_id") or e.get("chunkId", "")) for e in evidence}
            if cid not in allowed:
                continue
            pg = c.get("page_number") if c.get("page_number") is not None else c.get("page")
            try:
                pn = int(pg) if pg is not None else 0
            except (TypeError, ValueError):
                pn = 0
            citation_rows.append(
                {
                    "chunk_id": cid,
                    "page_number": pn,
                    "text": str(c.get("text", "")).strip()[:800],
                }
            )

    for coll in (raw_strengths, raw_gaps):
        if not isinstance(coll, list):
            continue
        for item in coll:
            if not isinstance(item, dict):
                continue
            for idx in item.get("citation_indices") or []:
                if isinstance(idx, int):
                    row = _citation_row_from_evidence_index(evidence, idx)
                    if row:
                        citation_rows.append(row)

    citations = _dedupe_citations(citation_rows)

    # evidence_used for Reference / legacy API (same shape as before)
    seen_ids: set[str] = set()
    evidence_used: list[dict] = []
    for row in citations:
        cid = row["chunk_id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        for e in evidence:
            if str(e.get("chunk_id") or e.get("chunkId", "")) == cid:
                pg = row.get("page_number")
                evidence_used.append(
                    {
                        "quote": str(e.get("text") or e.get("snippet") or "").strip()[:400],
                        "sourceId": cid,
                        "sourceType": str(e.get("sourceType", "jd")),
                        "sourceTitle": str(e.get("sourceTitle", "")),
                        "page": pg,
                        "chunkId": cid,
                    }
                )
                break

    score = data.get("score")
    if score is not None:
        try:
            score = max(0, min(10, float(score)))
        except (TypeError, ValueError):
            score = 5.0
    else:
        score = 5.0

    sug = data.get("suggested_followup")
    follow_up = [str(sug).strip()] if sug else []

    summary = str(data.get("summary") or "").strip()
    score_reasoning = str(data.get("score_reasoning") or "").strip()

    rubric_scores = _normalize_rubric_scores_list(data.get("rubric_scores"))

    raw_cit_list = data.get("citations")
    llm_citation_entries = len(
        [x for x in (raw_cit_list if isinstance(raw_cit_list, list) else []) if isinstance(x, dict)]
    )

    return {
        "score": score,
        "summary": summary,
        "score_reasoning": score_reasoning,
        "strengths": strengths,
        "gaps": gaps,
        "citations": citations,
        "strengths_cited": strengths_cited,
        "gaps_cited": gaps_cited,
        "improved_answer": str(data.get("improved_answer") or "").strip(),
        "follow_up_questions": follow_up,
        "suggested_followup": str(sug).strip() if sug else None,
        "evidence_used": evidence_used,
        "rubric_scores": rubric_scores,
        "_llm_citation_entries": llm_citation_entries,
    }


def _chunk_id_map(retrieved_chunks: list[dict]) -> dict[str, dict]:
    """chunk_id -> chunk dict."""
    m: dict[str, dict] = {}
    for e in retrieved_chunks or []:
        if not isinstance(e, dict):
            continue
        cid = str(e.get("chunk_id") or e.get("chunkId") or "").strip()
        if cid:
            m[cid] = e
    return m


def _evidence_used_from_citations(
    citations: list[dict],
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Rebuild evidence_used from validated citations and stored chunk text."""
    chunk_map = _chunk_id_map(retrieved_chunks)
    seen: set[str] = set()
    out: list[dict] = []
    for row in citations or []:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("chunk_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        e = chunk_map.get(cid)
        quote = str(row.get("text") or "").strip()
        if e is not None:
            if not quote:
                quote = str(e.get("text") or e.get("snippet") or "").strip()[:400]
            pg = row.get("page_number")
            try:
                page = int(pg) if pg is not None else int(e.get("page_number") or 0)
            except (TypeError, ValueError):
                page = int(e.get("page_number") or 0)
            out.append(
                {
                    "quote": quote[:400],
                    "sourceId": cid,
                    "sourceType": str(e.get("sourceType", "jd")),
                    "sourceTitle": str(e.get("sourceTitle", "")),
                    "page": page,
                    "chunkId": cid,
                }
            )
        else:
            try:
                pn = int(row.get("page_number") or 0)
            except (TypeError, ValueError):
                pn = 0
            out.append(
                {
                    "quote": quote[:400],
                    "sourceId": cid,
                    "sourceType": "jd",
                    "sourceTitle": "",
                    "page": pn,
                    "chunkId": cid,
                }
            )
    return out


def _find_verbatim_span_in_answer(user_answer: str, candidate: str) -> str:
    """Return the substring of user_answer that matches candidate (case-insensitive), or ""."""
    if not user_answer or not candidate or not candidate.strip():
        return ""
    ua = user_answer
    cand = candidate.strip()
    try:
        m = re.search(re.escape(cand), ua, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0)
    except re.error:
        pass
    lo, cl = ua.lower(), cand.lower()
    i = lo.find(cl)
    if i >= 0:
        return ua[i : i + len(cand)]
    return ""


def _longest_common_substring_span(a: str, b: str, min_len: int) -> tuple[str, int, int] | None:
    """Return (substring_of_a, start_in_a, len) for longest common contiguous match between a and b."""
    if not a or not b:
        return None
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    if m.size < min_len:
        return None
    return (a[m.a : m.a + m.size], m.a, m.size)


def enrich_strength_highlights(strengths: list[dict], user_answer: str) -> list[dict]:
    """
    Ensure each strength has a ``highlight`` that is an exact phrase from ``user_answer``
    (LLM value validated, else substring / fuzzy match from evidence or text).
    """
    ua = (user_answer or "").strip()
    out: list[dict] = []
    for s in strengths or []:
        if not isinstance(s, dict):
            continue
        ns = dict(s)
        ev = str(ns.get("evidence", "")).strip()
        tx = str(ns.get("text", "")).strip()
        hl_in = str(ns.get("highlight", "")).strip()

        resolved = ""
        if ua:
            resolved = _find_verbatim_span_in_answer(ua, hl_in)
            if not resolved and ev:
                resolved = _find_verbatim_span_in_answer(ua, ev)
            if not resolved and ev and len(ev) >= 4:
                lcs = _longest_common_substring_span(ua, ev, 5)
                if lcs:
                    resolved = lcs[0]
            if not resolved and tx and len(tx) >= 6:
                lcs = _longest_common_substring_span(ua, tx, 8)
                if lcs:
                    resolved = lcs[0]
            if not resolved and ev:
                # Sentence-level: best ratio between answer sentences and evidence
                best = ""
                best_r = 0.0
                for sent in re.split(r"(?<=[.!?])\s+|\n+", ua):
                    snt = sent.strip()
                    if len(snt) < 12:
                        continue
                    r = difflib.SequenceMatcher(None, snt.lower(), ev.lower()).ratio()
                    if r > best_r and r >= 0.42:
                        best_r = r
                        best = snt
                if best and len(best) <= 400:
                    resolved = best

        ns["highlight"] = (resolved or "")[:400]
        out.append(ns)
    return out


def _filter_citation_dict(cit: dict, allowed: set[str]) -> bool:
    ck = str(cit.get("chunkId") or cit.get("chunk_id") or "").strip()
    return ck in allowed


def validate_evaluation_output(
    response: dict,
    retrieved_chunks: list[dict],
    evaluation_mode: str = "full",
) -> dict:
    """
    Post-LLM validation: drop citations whose chunk_id is not in retrieved chunks;
    rebuild evidence_used; prune invalid nested citations; ensure non-empty strengths/gaps (full only).
    """
    out = dict(response)
    mode = (evaluation_mode or "full").strip().lower()
    if mode == "lite":
        out["citations"] = []
        out["evidence_used"] = []
        out["improved_answer"] = ""
        out["strengths"] = []
        out["gaps"] = []
        out["strengths_cited"] = []
        out["gaps_cited"] = []
        out["score_reasoning"] = str(out.get("score_reasoning") or "").strip()
        out["rubric_scores"] = _normalize_rubric_scores_list(out.get("rubric_scores"))
        return out
    allowed = {
        str(e.get("chunk_id") or e.get("chunkId") or "").strip()
        for e in (retrieved_chunks or [])
        if isinstance(e, dict) and (e.get("chunk_id") or e.get("chunkId"))
    }
    chunk_map = _chunk_id_map(retrieved_chunks)

    # --- Citations: keep only chunk_ids present in retrieved context ---
    filtered_citations: list[dict] = []
    for c in list(out.get("citations") or []):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("chunk_id") or c.get("chunkId") or "").strip()
        if cid not in allowed:
            continue
        src = chunk_map.get(cid, {})
        pg = c.get("page_number") if c.get("page_number") is not None else c.get("page")
        try:
            pn = int(pg) if pg is not None else int(src.get("page_number") or 0)
        except (TypeError, ValueError):
            pn = int(src.get("page_number") or 0)
        txt = str(c.get("text") or "").strip()
        if not txt:
            txt = str(src.get("text") or src.get("snippet") or "").strip()[:800]
        filtered_citations.append({"chunk_id": cid, "page_number": pn, "text": txt})
    out["citations"] = _dedupe_citations(filtered_citations)
    out["evidence_used"] = _evidence_used_from_citations(out["citations"], retrieved_chunks)

    # --- Strengths / gaps: drop empty dicts; require at least one item each ---
    strengths: list[dict] = []
    for s in list(out.get("strengths") or []):
        if not isinstance(s, dict):
            continue
        t = str(s.get("text", "")).strip()
        if not t:
            continue
        ev = str(s.get("evidence", "")).strip()
        hl = str(s.get("highlight", "")).strip()
        impact = str(s.get("impact", "")).strip()
        strengths.append(
            {
                "text": t,
                "evidence": ev or "See answer and rubric context.",
                "highlight": hl,
                "impact": impact,
            }
        )

    gaps: list[dict] = []
    for g in list(out.get("gaps") or []):
        if not isinstance(g, dict):
            continue
        t = str(g.get("text", "")).strip()
        if not t:
            continue
        missing = str(g.get("missing", "")).strip()
        exp = str(g.get("expected", "")).strip()
        jd_alignment = str(g.get("jd_alignment", "")).strip()
        improvement = str(g.get("improvement", "")).strip()
        gaps.append(
            {
                "text": t,
                "missing": missing,
                "expected": exp or "Align with rubric and job description context.",
                "jd_alignment": jd_alignment,
                "improvement": improvement,
            }
        )

    if not strengths:
        strengths = [
            {
                "text": "No specific strengths were identified in this evaluation pass.",
                "evidence": "The model did not return quotable strengths; compare your answer to the rubric and job description.",
                "highlight": "",
                "impact": "",
            }
        ]
    if not gaps:
        gaps = [
            {
                "text": "No specific gaps were identified in this evaluation pass.",
                "missing": "",
                "expected": "Strengthen your answer using the rubric bullets and cited job description requirements.",
                "jd_alignment": "",
                "improvement": "",
            }
        ]

    out["strengths"] = strengths
    out["gaps"] = gaps
    out["score_reasoning"] = str(out.get("score_reasoning") or "").strip()

    # Re-align cited lists to strengths/gaps text (same length, valid nested cites only)
    sc_old = out.get("strengths_cited") or []
    gc_old = out.get("gaps_cited") or []
    out["strengths_cited"] = []
    for i, s in enumerate(strengths):
        old = sc_old[i] if i < len(sc_old) and isinstance(sc_old[i], dict) else {}
        cites = [
            c
            for c in (old.get("citations") or [])
            if isinstance(c, dict) and _filter_citation_dict(c, allowed)
        ]
        out["strengths_cited"].append({"text": s["text"], "citations": cites})

    out["gaps_cited"] = []
    for i, g in enumerate(gaps):
        old = gc_old[i] if i < len(gc_old) and isinstance(gc_old[i], dict) else {}
        cites = [
            c
            for c in (old.get("citations") or [])
            if isinstance(c, dict) and _filter_citation_dict(c, allowed)
        ]
        out["gaps_cited"].append({"text": g["text"], "citations": cites})

    out["rubric_scores"] = _normalize_rubric_scores_list(out.get("rubric_scores"))

    return out


def evaluate_answer(
    question: str,
    question_type: str,
    focus_area: str,
    what_good_looks_like: list[str],
    must_mention: list[str],
    role_profile: dict,
    user_answer: str,
    evidence: list[dict],
    competency_label: str | None = None,
    document_rubric_json: list[dict] | None = None,
    evaluation_mode: str = "full",
    evaluation_cache_document_id: uuid.UUID | None = None,
) -> dict:
    """
    Evidence-cited, competency-aware evaluation of a candidate answer.

    Returns dict with:
      score (float 0-10),
      summary (str, why the score was given),
      score_reasoning (str, rubric-tied why this score),
      strengths: list of {text, evidence, highlight, impact},
      gaps: list of {text, missing, expected, jd_alignment, improvement},
      citations: list of {chunk_id, page_number, text},
      strengths_cited, gaps_cited (legacy cited shape for API),
      improved_answer (full rewrite targeting 9–10/10 per prompt rules),
      rubric_scores: list of {name, score, reasoning} when document JD dimensions are provided,
      follow_up_questions, suggested_followup, evidence_used,
      plus evidence_for_scoring added by the caller stack.

    Parsed output is passed through :func:`validate_evaluation_output` (citation IDs,
    non-empty strengths/gaps). One retry is attempted if all model citations were invalid.

    Model: :attr:`settings.openai_eval_chat_model` (default ``MODEL_FAST``;
    when ``USE_HIGH_QUALITY_EVAL`` is true, uses ``MODEL_HIGH_QUALITY``).
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    rp = role_profile or DEFAULT_ROLE_PROFILE.copy()
    mode = (evaluation_mode or "full").strip().lower()
    if mode not in ("lite", "full"):
        mode = "full"

    # Canonical {chunk_id, text, page_number, section_type} for prompt + citation_indices
    evidence_norm = normalize_evaluation_evidence(evidence or [])

    eval_cache_key: str | None = None
    if (
        evaluation_cache_document_id
        and settings.cache_ttl_evaluation_seconds > 0
    ):
        eval_cache_key = evaluation_cache_key(
            evaluation_cache_document_id,
            question,
            user_answer,
            document_rubric_json,
            evaluation_mode=mode,
        )
        cached_eval = cache_get_json(eval_cache_key)
        if isinstance(cached_eval, dict):
            parsed = copy.deepcopy(cached_eval)
            parsed = validate_evaluation_output(parsed, evidence_norm, mode)
            if mode != "lite":
                parsed["strengths"] = enrich_strength_highlights(parsed.get("strengths") or [], user_answer)
            parsed["evidence_for_scoring"] = list(evidence_norm)
            return parsed

    system_prompt, user_content = _build_domain_aware_evaluation_prompt(
        question=question,
        question_type=question_type,
        focus_area=focus_area,
        competency_label=competency_label,
        what_good_looks_like=what_good_looks_like or [],
        must_mention=must_mention or [],
        evidence=evidence_norm,
        role_profile=rp,
        answer_text=user_answer,
        document_rubric_json=document_rubric_json,
        evaluation_mode=mode,
    )

    client = OpenAI(api_key=settings.openai_api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if mode == "lite":
        eval_max_tokens = min(900, max(settings.max_completion_tokens * 3, 500))
    else:
        eval_max_tokens = min(2048, max(settings.max_completion_tokens * 4, 1200))
    eval_model = settings.openai_eval_chat_model
    response = client.chat.completions.create(
        model=eval_model,
        messages=messages,
        max_tokens=eval_max_tokens,
    )

    raw = (response.choices[0].message.content or "").strip()
    parsed = _parse_evaluation_response(raw, evidence_norm)
    llm_citation_attempts = int(parsed.pop("_llm_citation_entries", 0) or 0)
    parsed = validate_evaluation_output(parsed, evidence_norm, mode)
    if mode != "lite":
        parsed["strengths"] = enrich_strength_highlights(parsed.get("strengths") or [], user_answer)
    citation_count_after = len(parsed.get("citations") or [])

    # Retry once if the model cited chunk_ids not in context (all filtered) but chunks exist (full mode only)
    if (
        mode != "lite"
        and llm_citation_attempts > 0
        and citation_count_after == 0
        and evidence_norm
    ):
        retry_user = (
            user_content
            + "\n\nIMPORTANT: Your previous reply listed citation chunk_ids that do not appear in "
            'the "Retrieved job description chunks" section. Every citations[].chunk_id must match '
            "a chunk_id from that section exactly. Return JSON only."
        )
        response = client.chat.completions.create(
            model=eval_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user},
            ],
            max_tokens=eval_max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _parse_evaluation_response(raw, evidence_norm)
        parsed.pop("_llm_citation_entries", None)
        parsed = validate_evaluation_output(parsed, evidence_norm, mode)
        parsed["strengths"] = enrich_strength_highlights(parsed.get("strengths") or [], user_answer)

    if eval_cache_key and settings.cache_ttl_evaluation_seconds > 0:
        to_cache = copy.deepcopy(parsed)
        cache_set_json(eval_cache_key, to_cache, settings.cache_ttl_evaluation_seconds)

    parsed["evidence_for_scoring"] = list(evidence_norm)
    return parsed


AUXILIARY_SOURCE_TYPES = ["resume", "company", "notes"]


async def _has_auxiliary_sources(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID | None = None
) -> bool:
    """True if document has resume/company/notes sources or user has account-level resume."""
    from sqlalchemy import func
    r = await db.execute(
        select(func.count()).select_from(InterviewSource).where(
            InterviewSource.document_id == document_id,
            InterviewSource.source_type.in_(AUXILIARY_SOURCE_TYPES),
        )
    )
    if (r.scalar() or 0) > 0:
        return True
    if user_id and await get_user_resume_document_id(db, user_id):
        return True
    return False


async def _retrieve_auxiliary_evidence(
    db: AsyncSession,
    document_id: uuid.UUID,
    query: str,
    top_k: int = 4,
    user_id: uuid.UUID | None = None,
) -> list[dict]:
    """Retrieve from resume/company/notes (incl. user's account resume) for tailored phrasing."""
    additional_ids: list[uuid.UUID] = []
    if user_id:
        resume_doc_id = await get_user_resume_document_id(db, user_id)
        if resume_doc_id:
            additional_ids.append(resume_doc_id)
    query_embedding = embed_query(query)
    chunks = await retrieve_chunks(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        query_text=query,
        top_k=top_k,
        include_low_signal=False,
        section_types=None,
        doc_domain=None,
        source_types=AUXILIARY_SOURCE_TYPES,
        additional_document_ids=additional_ids if additional_ids else None,
    )
    return [_retrieval_dict_to_evidence_item(c) for c in chunks]


async def _retrieve_evidence_for_evaluation_query(
    db: AsyncSession,
    document_id: uuid.UUID,
    question: str,
    user_answer: str,
    competency_label: str | None,
    focus_area: str | None,
    top_k: int | None = None,
    session_pool: list[dict] | None = None,
) -> list[dict]:
    """
    Retrieve top relevant JD chunks for the evaluation prompt (question + answer + competency).

    When ``session_pool`` is provided (non-empty), re-ranks that pool with no vector query.

    Otherwise cached by (document_id + question text hash) when CACHE_TTL_RETRIEVAL_SECONDS > 0,
    then embed + retrieve_chunks.
    """
    k = top_k if top_k is not None else EVALUATION_QUERY_TOP_K
    parts = [question, (user_answer or "")[:500], competency_label or "", focus_area or ""]
    query_text = " ".join(p.strip() for p in parts if p and str(p).strip()).strip() or "job description requirements"

    if session_pool is not None and len(session_pool) > 0:
        ranked = _rank_pool_for_query(session_pool, query_text, k)
        if ranked:
            return ranked

    retrieval_key: str | None = None
    if settings.cache_ttl_retrieval_seconds > 0:
        retrieval_key = retrieval_cache_key(document_id, question)
        cached = await asyncio.to_thread(cache_get_json, retrieval_key)
        if isinstance(cached, list):
            return cached

    query_embedding = embed_query(query_text)
    chunks = await retrieve_chunks(
        db=db,
        document_id=document_id,
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=k,
        include_low_signal=False,
        section_types=None,
        doc_domain="job_description",
        source_types=["jd"],
    )
    result = [_retrieval_dict_to_evidence_item(c) for c in chunks]
    if retrieval_key and settings.cache_ttl_retrieval_seconds > 0:
        await asyncio.to_thread(
            cache_set_json,
            retrieval_key,
            result,
            settings.cache_ttl_retrieval_seconds,
        )
    return result


async def evaluate_answer_with_retrieval(
    db: AsyncSession,
    document_id: uuid.UUID,
    user_id: uuid.UUID | None,
    question: str,
    question_type: str,
    focus_area: str,
    competency_id: str | None,
    competency_label: str | None,
    evidence_chunk_ids: list[str],
    what_good_looks_like: list[str],
    must_mention: list[str],
    role_profile: dict,
    user_answer: str,
    evidence: list[dict],
    evaluation_mode: str = "full",
    session_id: uuid.UUID | None = None,
) -> dict:
    """
    Evaluate with question-linked evidence. Merges rubric chunks, optional competency
    and auxiliary retrieval, then top relevant JD chunks for (question + answer + competency),
    normalizes to citation-ready shapes, and calls :func:`evaluate_answer`.

    When ``session_id`` is set, JD chunks are cached per session (one vector retrieval per session);
    per-question relevance uses re-ranking from that pool instead of a separate vector query.
    """
    session_pool: list[dict] | None = None
    if session_id is not None:
        try:
            session_pool = await _get_or_create_session_jd_pool(
                db, document_id, session_id, role_profile
            )
        except Exception as e:
            logger.warning("Session JD pool retrieval failed: %s", e)
            session_pool = []

    ev = list(evidence or [])
    seen = {str(e.get("chunk_id") or e.get("chunkId", "")) for e in ev}

    # Thin JD evidence: prefer session pool (no extra vector query), then competency retrieval
    if len(ev) < 2 and competency_label:
        if session_pool:
            try:
                extra = _rank_pool_for_query(session_pool, competency_label, COMPETENCY_EVIDENCE_TOP_K)
                for e in extra:
                    cid = str(e.get("chunk_id") or e.get("chunkId", ""))
                    if cid and cid not in seen:
                        seen.add(cid)
                        ev.append(e)
            except Exception as e:
                logger.warning("Session pool competency fill failed: %s", e)
        if len(ev) < 2:
            try:
                extra = await _retrieve_evidence_for_competency(
                    db=db,
                    document_id=document_id,
                    competency_label=competency_label,
                )
                for e in extra:
                    cid = str(e.get("chunk_id") or e.get("chunkId", ""))
                    if cid and cid not in seen:
                        seen.add(cid)
                        ev.append(e)
            except Exception as e:
                logger.warning("Optional retrieval for evaluation failed: %s", e)

    # Auxiliary sources: retrieve for tailored phrasing (resume, company, notes, incl. user account resume)
    try:
        if await _has_auxiliary_sources(db, document_id, user_id):
            query = f"{competency_label or focus_area or ''} {user_answer[:200]}".strip() or "experience skills"
            auxiliary = await _retrieve_auxiliary_evidence(db, document_id, query, top_k=4, user_id=user_id)
            for e in auxiliary:
                cid = str(e.get("chunk_id") or e.get("chunkId", ""))
                if cid and cid not in seen:
                    seen.add(cid)
                    ev.append(e)
    except Exception as e:
        logger.warning("Auxiliary retrieval for evaluation failed: %s", e)

    # Top relevant JD chunks for evaluation (question + answer + competency)
    try:
        query_ev = await _retrieve_evidence_for_evaluation_query(
            db=db,
            document_id=document_id,
            question=question,
            user_answer=user_answer,
            competency_label=competency_label,
            focus_area=focus_area,
            session_pool=session_pool if session_pool else None,
        )
        for e in query_ev:
            cid = str(e.get("chunk_id") or e.get("chunkId", ""))
            if cid and cid not in seen:
                seen.add(cid)
                ev.append(e)
    except Exception as e:
        logger.warning("Evaluation query retrieval failed: %s", e)

    doc_rubric: list[dict] | None = None
    try:
        doc_r = await db.execute(select(Document).where(Document.id == document_id))
        doc_row = doc_r.scalar_one_or_none()
        raw_rj = getattr(doc_row, "rubric_json", None) if doc_row else None
        if isinstance(raw_rj, list) and raw_rj:
            doc_rubric = [x for x in raw_rj if isinstance(x, dict) and str(x.get("name", "")).strip()]
            if not doc_rubric:
                doc_rubric = None
    except Exception as e:
        logger.warning("Could not load document.rubric_json for evaluation: %s", e)

    return evaluate_answer(
        question=question,
        question_type=question_type,
        focus_area=focus_area,
        what_good_looks_like=what_good_looks_like or [],
        must_mention=must_mention or [],
        role_profile=role_profile or {},
        user_answer=user_answer,
        evidence=ev,
        competency_label=competency_label,
        document_rubric_json=doc_rubric,
        evaluation_mode=evaluation_mode,
        evaluation_cache_document_id=document_id,
    )
