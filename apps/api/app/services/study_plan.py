"""Role-specific interview study plan generation from JD intelligence."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.config import settings
from app.models import Document

logger = logging.getLogger(__name__)


class StudyPlanDay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    day: int = Field(ge=1)
    theme: str = ""
    topics: list[str] = Field(default_factory=list)
    drills: list[str] = Field(default_factory=list)
    mock_target: str = ""

    @field_validator("topics", "drills", mode="before")
    @classmethod
    def _normalize_string_list(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in v:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out[:6]


class StudyPlanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    role_title: str
    duration_days: int = Field(ge=7, le=14)
    summary: str
    daily_plan: list[StudyPlanDay] = Field(default_factory=list)


_STUDY_PLAN_SCHEMA: dict[str, Any] = {
    "name": "study_plan_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "role_title": {"type": "string"},
            "duration_days": {"type": "integer", "minimum": 7, "maximum": 14},
            "summary": {"type": "string"},
            "daily_plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "day": {"type": "integer", "minimum": 1},
                        "theme": {"type": "string"},
                        "topics": {"type": "array", "items": {"type": "string"}},
                        "drills": {"type": "array", "items": {"type": "string"}},
                        "mock_target": {"type": "string"},
                    },
                    "required": ["day", "theme", "topics", "drills", "mock_target"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["title", "role_title", "duration_days", "summary", "daily_plan"],
        "additionalProperties": False,
    },
}


def _unique_strings(items: list[Any], max_items: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _collect_jd_topics(document: Document) -> list[str]:
    jd_struct = document.jd_extraction_json if isinstance(document.jd_extraction_json, dict) else {}
    role_profile = document.role_profile if isinstance(document.role_profile, dict) else {}
    competencies = document.competencies if isinstance(document.competencies, list) else []
    rubric = document.rubric_json if isinstance(document.rubric_json, list) else []

    items: list[Any] = []
    items.extend(jd_struct.get("required_skills") or [])
    items.extend(jd_struct.get("preferred_skills") or [])
    items.extend(jd_struct.get("tools") or [])
    items.extend(jd_struct.get("cloud_platforms") or [])
    items.extend(role_profile.get("focusAreas") or [])
    items.extend(
        c.get("label", "")
        for c in competencies
        if isinstance(c, dict)
    )
    items.extend(
        r.get("name", "")
        for r in rubric
        if isinstance(r, dict)
    )

    topics = _unique_strings(items, max_items=20)
    if topics:
        return topics

    title_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#/.-]{2,}", document.filename or "")
    seed = _unique_strings(title_tokens, max_items=6)
    return seed or ["Core role skills", "Role responsibilities", "Interview communication"]


def _fallback_day(day_num: int, topic: str) -> StudyPlanDay:
    return StudyPlanDay(
        day=day_num,
        theme=f"Build confidence in {topic}",
        topics=[topic, f"{topic} fundamentals"],
        drills=[
            f"Create a 3-bullet STAR story demonstrating {topic}.",
            f"Practice a 2-minute answer explaining your approach to {topic}.",
        ],
        mock_target=f"Score 7/10 on a mock answer for a {topic}-focused interview question.",
    )


def _build_fallback_plan(document: Document, days: int, focus: str | None) -> StudyPlanResult:
    role_profile = document.role_profile if isinstance(document.role_profile, dict) else {}
    role_title = str(role_profile.get("roleTitleGuess") or "").strip() or "Target role"
    topics = _collect_jd_topics(document)
    if focus and focus.strip():
        topics = _unique_strings([focus.strip(), *topics], max_items=24)

    daily: list[StudyPlanDay] = []
    for i in range(days):
        topic = topics[i % len(topics)]
        daily.append(_fallback_day(i + 1, topic))

    return StudyPlanResult(
        title=f"{days}-Day Interview Study Plan",
        role_title=role_title,
        duration_days=days,
        summary=(
            f"Structured {days}-day plan built from job-description signals for {role_title}. "
            "Follow daily topic drills and hit mock targets before interview week."
        ),
        daily_plan=daily,
    )


def _extract_json_object(raw: str) -> str:
    text = (raw or "").strip()
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


def _normalize_days(plan: StudyPlanResult, fallback: StudyPlanResult, days: int) -> StudyPlanResult:
    normalized_days: list[StudyPlanDay] = []
    for idx in range(days):
        if idx < len(plan.daily_plan):
            item = plan.daily_plan[idx]
            topic_fallback = fallback.daily_plan[idx]
            normalized_days.append(
                StudyPlanDay(
                    day=idx + 1,
                    theme=item.theme or topic_fallback.theme,
                    topics=item.topics or topic_fallback.topics,
                    drills=item.drills or topic_fallback.drills,
                    mock_target=item.mock_target or topic_fallback.mock_target,
                )
            )
        else:
            normalized_days.append(fallback.daily_plan[idx])
    plan.daily_plan = normalized_days
    plan.duration_days = days
    return plan


def _jd_payload(document: Document) -> dict[str, Any]:
    return {
        "filename": document.filename,
        "doc_domain": document.doc_domain,
        "role_profile": document.role_profile if isinstance(document.role_profile, dict) else {},
        "jd_extraction_json": (
            document.jd_extraction_json if isinstance(document.jd_extraction_json, dict) else {}
        ),
        "competencies": (
            document.competencies if isinstance(document.competencies, list) else []
        ),
        "rubric_json": (
            document.rubric_json if isinstance(document.rubric_json, list) else []
        ),
    }


def generate_role_study_plan(
    *,
    document: Document,
    days: int,
    focus: str | None = None,
) -> dict[str, Any]:
    """Return a role-specific 7–14 day plan from JD intelligence."""
    fallback = _build_fallback_plan(document, days=days, focus=focus)

    if not settings.openai_api_key:
        return fallback.model_dump()

    client = OpenAI(api_key=settings.openai_api_key)
    jd_payload = _jd_payload(document)
    role_title_hint = (
        str((jd_payload.get("role_profile") or {}).get("roleTitleGuess") or "").strip()
        or "Target role"
    )

    system_prompt = """You create concise, practical interview study plans from structured job-description signals.

Output JSON only and follow the schema exactly.

Rules:
- Plan length must match duration_days exactly.
- Include daily topics, concrete drills, and a mock_target for each day.
- Make drills actionable (e.g., timed STAR reps, whiteboard practice, domain drills, rubric self-score).
- Use only the JD signals provided; do not invent employer-specific facts.
- Keep each day realistic for 60–120 minutes of prep."""

    user_prompt = (
        f"Create a {days}-day interview prep plan.\n"
        f"Role title hint: {role_title_hint}\n"
        f"Optional focus: {focus.strip() if focus and focus.strip() else '(none)'}\n\n"
        f"JD intelligence payload:\n{json.dumps(jd_payload, ensure_ascii=True)}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.model_fast,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=min(2200, settings.max_completion_tokens * 5),
            response_format={"type": "json_schema", "json_schema": _STUDY_PLAN_SCHEMA},
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(_extract_json_object(raw))
        result = StudyPlanResult.model_validate(parsed)
        result = _normalize_days(result, fallback=fallback, days=days)
        return result.model_dump()
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("study_plan validation failed; using fallback: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("study_plan generation failed; using fallback: %s", exc)
    return fallback.model_dump()

