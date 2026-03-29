"""JD + resume + gap–aware interview question generation (structured JSON).

This complements :func:`app.services.interview.generate_interview_questions`, which is
role-profile + single-corpus based. Use this module when you have separate JD chunks,
resume chunks, and structured fit gaps.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_CHUNK = 1000
_MAX_JD_CHUNKS = 24
_MAX_RESUME_CHUNKS = 24
_DEFAULT_MAX_QUESTIONS = 12

QuestionType = Literal["behavioral", "technical", "gap"]

_QUESTIONS_JSON_SCHEMA: dict[str, Any] = {
    "name": "interview_questions_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "Behavioral, technical, and gap-targeted interview questions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["behavioral", "technical", "gap"],
                            "description": "behavioral: soft skills/situations; technical: role skills; gap: probes missing JD areas using adjacent resume context.",
                        },
                        "question": {
                            "type": "string",
                            "description": "Single clear interview question, grounded in provided excerpts and gaps.",
                        },
                    },
                    "required": ["type", "question"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}


class InterviewQuestionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: QuestionType
    question: str

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        if s in ("behavioral", "technical", "gap"):
            return s
        return "behavioral"

    @field_validator("question", mode="before")
    @classmethod
    def _strip_q(cls, v: Any) -> str:
        return str(v or "").strip()


class InterviewQuestionsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[InterviewQuestionItem] = Field(default_factory=list)


def _chunk_text(c: dict) -> str:
    return (c.get("snippet") or c.get("text") or "").strip()


def _format_chunk_block(title: str, chunks: list[dict], cap: int) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks[:cap]):
        body = _chunk_text(c)[:_MAX_CHARS_PER_CHUNK]
        if not body:
            continue
        lines.append(f"[{i}] {body}")
    if not lines:
        return f"{title}:\n(no text)"
    return f"{title}:\n" + "\n\n".join(lines)


def _format_gaps(gaps: list[dict]) -> str:
    if not gaps:
        return "Gaps:\n(none listed — still generate a few gap-style probes for weak or unstated areas vs the JD.)"
    parts: list[str] = []
    for i, g in enumerate(gaps):
        req = str(g.get("requirement") or g.get("gap") or "").strip()
        reason = str(g.get("reason") or "").strip()
        imp = str(g.get("importance") or "").strip()
        line = f"[{i}] Requirement: {req or '(unspecified)'}"
        if reason:
            line += f"\n    Why it is a gap: {reason}"
        if imp:
            line += f"\n    Importance: {imp}"
        parts.append(line)
    return "Gaps (prioritize gap-type questions here):\n" + "\n\n".join(parts)


def _extract_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    return m.group(0) if m else raw


_SYSTEM_PROMPT = """You are an expert interviewer. You write interview questions for a hiring panel.

You are given:
1. Job description excerpts
2. Candidate resume excerpts
3. Identified gaps (JD requirements weak or missing on the resume)

Produce a balanced set of questions:
- **behavioral**: past behavior, collaboration, conflict, ownership, leadership signals — tied to themes visible in the JD when possible.
- **technical**: skills, tools, depth, tradeoffs, debugging, design — grounded in technologies/responsibilities mentioned in the JD (and fairly calibrated to what the resume shows).
- **gap**: target specific gaps. Example: if the gap is missing AWS, ask how they have approached cloud infrastructure or production deployments in the past, or how they would ramp on AWS — invite them to surface *adjacent* experience without fabricating credentials. Never accuse; stay professional and open-ended.

Rules:
- Ground every question in the provided JD/resume/gap text. Do not invent employers, projects, or certifications not supported by the excerpts.
- Questions must be specific and non-generic (avoid "Tell me about yourself", "What are your strengths?", "Why should we hire you?").
- Include several gap-type questions when gaps are listed; each should map to a concrete gap where possible.
- Return STRICT JSON matching the schema (no markdown)."""


def _coerce_payload(data: dict[str, Any]) -> dict[str, Any]:
    if "questions" not in data or data["questions"] is None:
        data["questions"] = []
    return data


def _parse_result(content: str) -> InterviewQuestionsResult:
    cleaned = _extract_json_object(content)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise TypeError("root must be object")
    return InterviewQuestionsResult.model_validate(_coerce_payload(payload))


def generate_interview_questions(
    jd_chunks: list[dict],
    resume_chunks: list[dict],
    gaps: list[dict],
    *,
    max_questions: int = _DEFAULT_MAX_QUESTIONS,
) -> dict[str, Any]:
    """
    Generate behavioral, technical, and gap-targeted interview questions.

    Args:
        jd_chunks: Chunk dicts (e.g. from retrieval) with ``snippet`` / ``text``.
        resume_chunks: Same shape for the candidate resume.
        gaps: Dicts with at least ``requirement`` (and optionally ``reason``, ``importance``).
        max_questions: Soft cap stated in the user prompt (model may return slightly fewer).

    Returns:
        ``{"questions": [{"type": "behavioral"|"technical"|"gap", "question": str}, ...]}``

    Raises:
        ValueError: if OpenAI is not configured.
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    jd_has = any(bool(_chunk_text(c)) for c in jd_chunks)
    rs_has = any(bool(_chunk_text(c)) for c in resume_chunks)
    if not jd_has and not rs_has and not gaps:
        return {"questions": []}

    jd_block = _format_chunk_block("JOB DESCRIPTION EXCERPTS", jd_chunks, _MAX_JD_CHUNKS)
    resume_block = _format_chunk_block("RESUME EXCERPTS", resume_chunks, _MAX_RESUME_CHUNKS)
    gaps_block = _format_gaps(gaps)

    user_content = f"""{jd_block}

{resume_block}

{gaps_block}

Generate up to {max_questions} questions total, with a mix of behavioral, technical, and gap types.
Each gap listed should usually have at least one corresponding **gap**-type question unless gaps are empty."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_tokens = min(3000, max(800, settings.max_completion_tokens * 5))
    client = OpenAI(api_key=settings.openai_api_key)

    def _call(with_schema: bool) -> str:
        kwargs: dict[str, Any] = {
            "model": settings.model_fast,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if with_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": _QUESTIONS_JSON_SCHEMA,
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    raw: str
    try:
        raw = _call(with_schema=True)
    except Exception as e:
        logger.warning(
            "interview_questions: json_schema failed (%s); falling back to json_object",
            e,
        )
        raw = _call(with_schema=False)

    try:
        result = _parse_result(raw)
    except Exception as e:
        logger.exception("interview_questions: parse failed: %s", e)
        return {"questions": []}

    cap = max_questions if max_questions > 0 else _DEFAULT_MAX_QUESTIONS
    trimmed = result.questions[:cap]
    return {"questions": [q.model_dump() for q in trimmed]}
