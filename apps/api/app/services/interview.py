"""Interview Prep: evidence retrieval + domain-aware question generation from job descriptions."""

import json
import logging
import re
import uuid

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Document, InterviewSource
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
    return [
        {
            "chunk_id": c["chunk_id"],
            "chunkId": c.get("chunkId") or c["chunk_id"],
            "page_number": c.get("page_number"),
            "page": c.get("page"),
            "snippet": c.get("snippet") or c.get("text", ""),
            "sourceType": c.get("sourceType", "jd"),
            "sourceTitle": c.get("sourceTitle", ""),
            "retrieval_source": c.get("retrieval_source"),
            "semantic_score": c.get("semantic_score"),
            "keyword_score": c.get("keyword_score"),
            "final_score": c.get("final_score"),
        }
        for c in chunks
    ]


def _generate_single_question(
    competency_id: str,
    competency_label: str,
    question_type: str,
    evidence: list[dict],
    role_profile: dict,
) -> dict | None:
    """Generate one question for a competency using its evidence. Returns None if no evidence."""
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

    system_prompt = f"""You are an expert hiring manager. Create ONE interview question for the competency "{competency_label}".

QUESTION TYPE: {question_type}
- behavioral: Past experience, STAR format, "Tell me about a time..."
- role_specific: Domain knowledge/skills for this competency
- scenario: Hypothetical "What would you do if..." situation

SENIORITY: {seniority} ({seniority_hint})
DOMAIN: {domain}

RULES:
1. The question MUST clearly reflect the JD evidence excerpts below.
2. Ground the question in specific requirements/skills from the excerpts.
3. whatGoodLooksLike: 2-4 bullets describing what a strong answer covers.
4. Output valid JSON only: {{"question": "...", "whatGoodLooksLike": ["...","..."]}}"""

    user_content = f"""Competency: {competency_label}

Job description excerpts:
{excerpts_text}

Generate exactly 1 {question_type} question. Output JSON only."""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
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
            "type": question_type,
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
) -> list[dict]:
    """
    Generate questions tied to competencies. Each question uses retrieval for that competency.

    Args:
        db: AsyncSession
        document_id: Document UUID
        num_questions: Number of questions to generate
        role_profile: { domain, seniority, questionMix }
        competencies: List of { id, label, description?, evidence? }

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

        q_type = type_cycle[i % len(type_cycle)] if type_cycle else "behavioral"
        q = _generate_single_question(
            competency_id=comp_id,
            competency_label=comp_label,
            question_type=q_type,
            evidence=evidence,
            role_profile=rp,
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

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
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
) -> tuple[str, str]:
    """Build prompts for evidence-cited, competency-aware evaluation."""
    rubric_text = "\n".join(f"• {b}" for b in (what_good_looks_like or []))
    if must_mention:
        rubric_text += "\n\nMust mention: " + "; ".join(must_mention)

    evidence_lines = []
    for i, e in enumerate(evidence or []):
        s = (e.get("snippet") or "").strip()
        cid = e.get("chunk_id") or e.get("chunkId") or ""
        pg = e.get("page_number") or e.get("page")
        src_type = e.get("sourceType", "jd")
        src_title = e.get("sourceTitle", "")
        if s:
            meta = f"[{i}] chunkId={cid} sourceType={src_type} sourceTitle={src_title}"
            pg_part = f" page={pg}" if pg is not None else ""
            evidence_lines.append(f"{meta}{pg_part}:\n{s[:500]}")
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "(No evidence - job description does not specify details for this area)"

    comp_context = f" | COMPETENCY: {competency_label}" if competency_label else ""
    domain = role_profile.get("domain", "general_business")
    seniority = role_profile.get("seniority", "entry")
    domain_hint = EVALUATION_DOMAIN_HINTS.get(domain, EVALUATION_DOMAIN_HINTS["general_business"])
    seniority_hint = "entry-level depth" if seniority == "entry" else "mid-level depth" if seniority == "mid" else "senior-level depth"

    system_prompt = f"""You are an interviewer evaluating a candidate's answer. Be DOMAIN-AWARE and EVIDENCE-CITED.

DOMAIN: {domain} | SENIORITY: {seniority} ({seniority_hint})
QUESTION TYPE: {question_type} | FOCUS AREA: {focus_area}{comp_context}
ADDITIONAL DOMAIN CONSIDERATIONS: {domain_hint}

CRITICAL RULES:
1. Evaluate ONLY against whatGoodLooksLike and mustMention. Tune expectations by domain and seniority.
2. EVERY strength and gap MUST have at least 1 citation (indices into the evidence list) OR explicitly say "JD does not specify" in the text.
3. NEVER fabricate or hallucinate citations. Only use evidence indices 0,1,2,... from the provided excerpts.
4. For each strength/gap, cite the specific evidence that supports it. If no evidence exists for a point, write "JD does not specify" in that item's text.
5. Evidence may include JD, resume, company, or notes (see sourceType per excerpt). When suggesting tailored phrasing (e.g. from candidate's resume), always cite which source you used.
6. improved_answer: one paragraph grounded ONLY in the provided evidence.
7. suggested_followup: one optional follow-up question (string, or null).
8. Score 0-10: 0=no relevant content, 10=fully addresses rubric with evidence.

Output valid JSON only, no markdown:
{{
  "score": N,
  "strengths": [{{"text": "...", "citation_indices": [0,1]}}],
  "gaps": [{{"text": "...", "citation_indices": [0]}}],
  "improved_answer": "...",
  "suggested_followup": "..." or null
}}
citation_indices: 0-based indices into the evidence list. Use [] only when text says "JD does not specify"."""

    user_content = f"""Question: {question}

Rubric (what a strong answer should cover):
{rubric_text}

Evidence (JD, resume, company, or notes - cite by index 0,1,2,...):
{evidence_text}

Candidate's answer:
{answer_text}

Output JSON only. Every strength/gap must have citation_indices or text containing "JD does not specify". No fabricated sources."""

    return system_prompt, user_content


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


def _parse_evaluation_response(raw: str, evidence: list[dict]) -> dict:
    """
    Parse LLM evaluation JSON into cited strengths/gaps.
    strengths: [{text, citations: [{chunkId, page?, sourceTitle, sourceType}]}]
    gaps: [{text, citations: [...]}]
    """
    text = raw.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)
    data = json.loads(text)

    def _parse_cited_items(items: list, ev: list[dict]) -> list[dict]:
        result = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str):
                result.append({"text": str(item).strip(), "citations": []})
                continue
            if not isinstance(item, dict):
                continue
            t = str(item.get("text", "")).strip()
            idx_list = item.get("citation_indices")
            if not isinstance(idx_list, list):
                idx_list = []
            citations = []
            for idx in idx_list:
                if isinstance(idx, int) and 0 <= idx < len(ev):
                    citations.append(_citation_from_evidence(ev[idx]))
            result.append({"text": t, "citations": citations})
        return result

    raw_strengths = data.get("strengths") or []
    raw_gaps = data.get("gaps") or []
    # Backward compat: LLM may return flat string arrays
    if raw_strengths and isinstance(raw_strengths[0], str):
        raw_strengths = [{"text": s, "citation_indices": []} for s in raw_strengths]
    if raw_gaps and isinstance(raw_gaps[0], str):
        raw_gaps = [{"text": g, "citation_indices": []} for g in raw_gaps]
    strengths = _parse_cited_items(raw_strengths, evidence)
    gaps = _parse_cited_items(raw_gaps, evidence)

    # Backward compat: flat lists for existing UX
    strengths_flat = [s["text"] for s in strengths if s["text"]]
    gaps_flat = [g["text"] for g in gaps if g["text"]]

    # Aggregate evidence_used from all citations (for Reference section)
    seen_ids: set[str] = set()
    evidence_used = []
    for s in strengths + gaps:
        for c in s.get("citations", []):
            cid = c.get("chunkId", "")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                for e in evidence:
                    if str(e.get("chunk_id") or e.get("chunkId", "")) == cid:
                        evidence_used.append({
                            "quote": str(e.get("snippet", "")).strip()[:400],
                            "sourceId": cid,
                            "sourceType": c.get("sourceType", "jd"),
                            "sourceTitle": c.get("sourceTitle", ""),
                            "page": c.get("page"),
                            "chunkId": cid,
                        })
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

    return {
        "score": score,
        "strengths": strengths_flat,
        "gaps": gaps_flat,
        "strengths_cited": strengths,
        "gaps_cited": gaps,
        "improved_answer": str(data.get("improved_answer") or "").strip(),
        "follow_up_questions": follow_up,
        "suggested_followup": str(sug).strip() if sug else None,
        "evidence_used": evidence_used,
    }


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
) -> dict:
    """
    Evidence-cited, competency-aware evaluation of a candidate answer.

    Returns {score, strengths, gaps, strengths_cited, gaps_cited, improved_answer,
    follow_up_questions, suggested_followup, evidence_used}.
    Every strength/gap is citation-backed or explicitly "JD does not specify".
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    rp = role_profile or DEFAULT_ROLE_PROFILE.copy()

    system_prompt, user_content = _build_domain_aware_evaluation_prompt(
        question=question,
        question_type=question_type,
        focus_area=focus_area,
        competency_label=competency_label,
        what_good_looks_like=what_good_looks_like or [],
        must_mention=must_mention or [],
        evidence=evidence or [],
        role_profile=rp,
        answer_text=user_answer,
    )

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=min(1000, settings.max_completion_tokens * 2),
    )

    raw = (response.choices[0].message.content or "").strip()
    return _parse_evaluation_response(raw, evidence or [])


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
    return [
        {
            "chunk_id": c["chunk_id"],
            "chunkId": c.get("chunkId") or c["chunk_id"],
            "page_number": c.get("page_number"),
            "page": c.get("page"),
            "snippet": c.get("snippet") or c.get("text", ""),
            "sourceType": c.get("sourceType", "jd"),
            "sourceTitle": c.get("sourceTitle", ""),
            "retrieval_source": c.get("retrieval_source"),
            "semantic_score": c.get("semantic_score"),
            "keyword_score": c.get("keyword_score"),
            "final_score": c.get("final_score"),
        }
        for c in chunks
    ]


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
) -> dict:
    """
    Evaluate with question-linked evidence. If evidence is thin, retrieve by competency.
    When document has resume/company/notes or user has account resume, retrieve for tailored phrasing.
    """
    ev = list(evidence or [])
    seen = {str(e.get("chunk_id") or e.get("chunkId", "")) for e in ev}

    # Thin JD evidence: retrieve more from JD
    if len(ev) < 2 and competency_label:
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
    )
