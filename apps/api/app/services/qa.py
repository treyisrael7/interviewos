"""Grounded Q&A: structured recruiter-style fit reasoning over retrieved excerpts."""

import json
import logging
import re
from openai import OpenAI

from app.core.config import settings
from app.services.jd_sections import normalize_jd_text
from app.services.token_budget import budget_grounded_qa_prompt

logger = logging.getLogger(__name__)

# OpenAI sampling: keep deterministic, within 0–0.3 per product guidance.
QA_TEMPERATURE = 0.2

_SYSTEM_PROMPT = """You are an AI recruiter analyzing candidate-job fit.

You are given:
1. Job description excerpts
2. Candidate resume excerpts

Tasks:
- Extract key job requirements
- Match them to candidate experience
- Identify gaps
- Assign a fit score (0–100)
- Provide reasoning grounded ONLY in provided excerpts

Rules:
- Do not hallucinate
- If evidence is missing, mark as a gap
- Always cite evidence implicitly in reasoning

Return STRICT JSON only."""

_JSON_SHAPE_INSTRUCTIONS = """
Return one JSON object with exactly these keys (no markdown, no code fences, no extra keys):
- "key_job_requirements": array of strings (each requirement must be grounded in job description excerpts)
- "matches": array of objects, each with "requirement" (string), "candidate_experience" (string), "alignment_notes" (string)
- "gaps": array of objects, each with "requirement" (string), "reason" (string)
- "fit_score": integer from 0 through 100
- "reasoning": string (overall narrative; only claims supported by the excerpts above)
"""


def _split_jd_resume_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    jd: list[dict] = []
    resume: list[dict] = []
    for c in chunks:
        st = str(c.get("source_type") or "").strip().upper()
        if st == "RESUME":
            resume.append(c)
        else:
            jd.append(c)
    return jd, resume


def _format_excerpt_block(label: str, chunks: list[dict], start_index: int) -> tuple[str, int]:
    """Build labeled excerpt lines with [p{page}-c{i}] markers; returns (text, next_index)."""
    lines: list[str] = []
    i = start_index
    for c in chunks:
        page = c.get("page_number", 0)
        marker = f"[p{page}-c{i}]"
        snippet = normalize_jd_text(c.get("snippet", "")).strip()
        lines.append(f"{marker} {snippet}")
        i += 1
    body = "\n\n".join(lines) if lines else "(No excerpt text in this section.)"
    return f"### {label}\n{body}", i


def _build_user_content_for_budget(jd_chunks: list[dict], resume_chunks: list[dict], question: str) -> str:
    jd_block, next_idx = _format_excerpt_block("Job description excerpts", jd_chunks, start_index=1)
    resume_block, _ = _format_excerpt_block(
        "Candidate resume excerpts", resume_chunks, start_index=next_idx
    )
    return f"""{jd_block}

{resume_block}

User question / focus:
{question.strip()}

{_JSON_SHAPE_INSTRUCTIONS.strip()}
"""


def _extract_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    return m.group(0) if m else raw


# Chunks are list of {chunk_id, page_number, snippet, source_type?, ...}
# Returns (answer_json_string, citations)
def generate_grounded_answer(
    question: str,
    chunks: list[dict],
    max_tokens: int | None = None,
) -> tuple[str, list[dict]]:
    """
    Call OpenAI with structured recruiter reasoning. ``answer`` is a JSON string
    (validated when parseable). Citations list mirrors retrieved chunks for API compatibility.
    """
    max_tokens = max_tokens or settings.max_completion_tokens
    max_tokens = max(max_tokens, 1200)

    if not chunks:
        return (
            "I don't have enough information in this document to answer that.",
            [],
        )

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    budget_chunks, user_content, effective_max_tokens = budget_grounded_qa_prompt(
        question=question,
        chunks=chunks,
        system_prompt=_SYSTEM_PROMPT,
        split_chunks=_split_jd_resume_chunks,
        build_user_content=lambda jd_c, rs_c: _build_user_content_for_budget(jd_c, rs_c, question),
        requested_completion_tokens=max_tokens,
        total_budget=settings.max_llm_budget_tokens,
    )

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.model_fast,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=effective_max_tokens,
        temperature=QA_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    raw = (response.choices[0].message.content or "").strip()
    answer: str
    try:
        payload = json.loads(_extract_json_object(raw))
        if not isinstance(payload, dict):
            raise TypeError("root must be object")
        answer = json.dumps(payload, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("QA JSON parse failed, returning raw model text: %s", e)
        answer = raw

    citations = [
        {
            "chunk_id": str(c.get("chunk_id") or c.get("chunkId") or ""),
            "page_number": int(c.get("page_number") if c.get("page_number") is not None else c.get("page") or 0),
            "snippet": normalize_jd_text(c.get("snippet", "")),
        }
        for c in budget_chunks
    ]

    return answer, citations
