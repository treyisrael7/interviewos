"""LLM extraction of evaluation dimensions (rubric) from job description text.

Uses ``settings.model_fast`` (default: gpt-4o-mini). Final answer evaluation may use a
higher-quality model via ``USE_HIGH_QUALITY_EVAL``; rubric extraction always stays on the
default/cheaper chat model.
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

RUBRIC_DIM_MIN = 4
RUBRIC_DIM_MAX = 6

_SYSTEM_PROMPT = """You are an expert recruiter. From the job description, propose evaluation dimensions for interview scoring.

Output valid JSON only (no markdown), exactly this shape:
{"dimensions": [{"name": "Short title", "description": "1-3 sentences: what to assess and what strong signals look like for THIS role"}]}

Rules:
- Produce between 4 and 6 dimensions (inclusive).
- Each dimension must be specific to THIS role, company context, and responsibilities in the text. Tie names and descriptions to tools, domain, seniority, and outcomes mentioned in the job description.
- Avoid vague labels like "Communication", "Teamwork", or "Problem solving" unless the JD explicitly emphasizes them; prefer concrete capability areas (e.g. named stacks, regulatory context, stakeholder types, metrics).
- Descriptions should help an interviewer score answers: what evidence would justify a high vs weak rating.
- If the job description is sparse, infer reasonable role-specific dimensions only from what is stated—do not invent employers or requirements not implied by the text."""


def _extract_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    return m.group(0) if m else raw


def _normalize_dimensions(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        desc = str(entry.get("description", "")).strip()
        if not name:
            continue
        out.append({"name": name, "description": desc or name})
    return out


def extract_rubric_from_jd(jd_text: str) -> list[dict[str, str]]:
    """
    Use the configured chat model to extract 4–6 role-specific evaluation dimensions from ``jd_text``.

    Returns a list of ``{"name": str, "description": str}`` items.

    Raises:
        ValueError: if ``jd_text`` is empty/whitespace-only, or OpenAI is not configured.
    """
    text = (jd_text or "").strip()
    if not text:
        raise ValueError("jd_text is empty")

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    # Keep prompt bounded; front matter usually has title + requirements.
    max_chars = 14_000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[…truncated for length]"

    user_content = f"""Job description:

{text}

Return JSON with exactly {RUBRIC_DIM_MIN}–{RUBRIC_DIM_MAX} dimensions as specified."""

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.model_fast,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=min(1200, settings.max_completion_tokens * 6),
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        payload = json.loads(_extract_json_object(raw))
        dims = payload.get("dimensions")
        if isinstance(payload, list):
            dims = payload
        normalized = _normalize_dimensions(dims if isinstance(dims, list) else [])

        if RUBRIC_DIM_MIN <= len(normalized) <= RUBRIC_DIM_MAX:
            return normalized
        if len(normalized) > RUBRIC_DIM_MAX:
            return normalized[:RUBRIC_DIM_MAX]
        if len(normalized) < RUBRIC_DIM_MIN:
            # One repair pass: model returned too few items.
            repair = client.chat.completions.create(
                model=settings.model_fast,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_content
                        + f"\n\nYour previous output had only {len(normalized)} dimensions. "
                        f"Return JSON with between {RUBRIC_DIM_MIN} and {RUBRIC_DIM_MAX} dimensions only.",
                    },
                ],
                max_tokens=min(1200, settings.max_completion_tokens * 6),
                response_format={"type": "json_object"},
            )
            raw2 = (repair.choices[0].message.content or "").strip()
            payload2 = json.loads(_extract_json_object(raw2))
            dims2 = payload2.get("dimensions")
            if isinstance(payload2, list):
                dims2 = payload2
            normalized = _normalize_dimensions(dims2 if isinstance(dims2, list) else [])
            if len(normalized) > RUBRIC_DIM_MAX:
                normalized = normalized[:RUBRIC_DIM_MAX]
        if len(normalized) < RUBRIC_DIM_MIN:
            logger.warning(
                "Rubric extraction returned %s dimensions (expected %s–%s)",
                len(normalized),
                RUBRIC_DIM_MIN,
                RUBRIC_DIM_MAX,
            )
        return normalized
    except json.JSONDecodeError as e:
        logger.exception("Rubric extraction JSON parse failed: %s", e)
        return []
    except Exception as e:
        logger.exception("Rubric extraction failed: %s", e)
        return []
