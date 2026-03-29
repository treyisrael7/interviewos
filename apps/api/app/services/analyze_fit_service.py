"""JD ↔ resume fit analysis using retrieved chunks and structured LLM output.

Exactly **one** chat-completions request is made per successful analysis path
(no retries or staged prompts). A per-request counter logs an error if that
invariant is violated.

Chunks are expected from :func:`app.services.retrieval.retrieve_chunks` with
``source_type`` in ``{"JD", "RESUME", "OTHER"}``. They are passed through
:func:`compress_chunks` before the LLM call to cut token use while keeping
high-signal sentences (skills, tools, experience, requirements).

Only ``JD`` and ``RESUME`` buckets feed role-specific context; ``OTHER`` is
grouped with JD for requirement context unless you filter upstream.

``fit_score`` is computed deterministically on the server from matches/gaps;
the model may supply ``fit_score_hint`` for transparency only.

``recommendations`` (one per gap when present) are model-generated but
``gap`` text is normalized to the structured gap requirement on the server.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import settings
from app.services.token_budget import TOKEN_BUDGET_SAFETY_SLACK, estimate_tokens

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_CHUNK = 1200
_MAX_CHUNKS_PER_SIDE = 28
# Target cap for the user-message excerpt block (JOB + RESUME formatted bodies + wrappers).
_ANALYZE_FIT_USER_EXCERPT_TOKEN_BUDGET = 3000
# Reserved for fixed user prompt framing (question, user_id line, closing instruction).
_ANALYZE_FIT_USER_FRAME_TOKEN_EST = 450
_MAX_COMPRESS_SENTENCES = 3
_MIN_COMPRESS_SENTENCES = 1
_MAX_CHARS_PER_COMPRESSED_CHUNK = 480

ImportanceLevel = Literal["low", "medium", "high"]

# Weighted coverage: how much each requirement counts toward the denominator.
_IMPORTANCE_WEIGHT: dict[str, float] = {
    "low": 1.0,
    "medium": 1.5,
    "high": 2.5,
}
# Per-gap penalty (always), plus extra when the gap is high-importance ("critical").
_GAP_BASE_PENALTY = 2.5
_CRITICAL_GAP_EXTRA = 6.0


def compute_fit_score(
    matches: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Deterministic fit score from LLM-structured matches and gaps.

    Uses weighted coverage: sum(importance_weight * confidence) over matches,
    divided by sum(importance_weight) over all requirements (matches + gaps),
    times 100, minus gap penalties (larger when gap importance is ``high``).

    Also returns ``matched_count``, ``total_requirements``, ``gap_count``,
    ``gap_penalty``, and ``coverage_raw`` (pre-penalty percentage).
    """
    matched_count = len(matches)
    gap_count = len(gaps)
    total_requirements = matched_count + gap_count

    def _w(item: dict[str, Any]) -> float:
        return float(_IMPORTANCE_WEIGHT.get(str(item.get("importance") or "medium"), 1.5))

    if total_requirements == 0:
        return {
            "fit_score": 0,
            "matched_count": 0,
            "total_requirements": 0,
            "gap_count": 0,
            "gap_penalty": 0.0,
            "coverage_raw": 0.0,
        }

    total_weight = sum(_w(m) for m in matches) + sum(_w(g) for g in gaps)
    if total_weight <= 0:
        total_weight = float(total_requirements)

    matched_weighted = sum(_w(m) * float(m.get("confidence") or 0.0) for m in matches)
    coverage_raw = (matched_weighted / total_weight) * 100.0

    gap_penalty = 0.0
    for g in gaps:
        gap_penalty += _GAP_BASE_PENALTY
        if str(g.get("importance") or "").lower() == "high":
            gap_penalty += _CRITICAL_GAP_EXTRA

    fit = int(round(coverage_raw - gap_penalty))
    fit = max(0, min(100, fit))

    return {
        "fit_score": fit,
        "matched_count": matched_count,
        "total_requirements": total_requirements,
        "gap_count": gap_count,
        "gap_penalty": round(gap_penalty, 2),
        "coverage_raw": round(coverage_raw, 2),
    }


# OpenAI Structured Outputs (strict JSON schema). All object keys required; no extras.
_ANALYZE_FIT_JSON_SCHEMA: dict[str, Any] = {
    "name": "analyze_fit_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "description": "JD requirements with resume support.",
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "type": "string",
                            "description": "A concrete requirement or qualification from the job text.",
                        },
                        "resume_evidence": {
                            "type": "string",
                            "description": "Verbatim or tightly paraphrased evidence from the resume excerpts; empty if none.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0–1.0 confidence that the resume supports this requirement.",
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Job-side importance: high = must-have / critical skill.",
                        },
                    },
                    "required": ["requirement", "resume_evidence", "confidence", "importance"],
                    "additionalProperties": False,
                },
            },
            "gaps": {
                "type": "array",
                "description": "Important JD requirements not adequately supported by the resume.",
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement": {
                            "type": "string",
                            "description": "The unmet or weakly met requirement.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why the resume does not demonstrate this (cite lack of evidence).",
                        },
                        "importance": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "How critical this gap is for the role.",
                        },
                    },
                    "required": ["requirement", "reason", "importance"],
                    "additionalProperties": False,
                },
            },
            "fit_score_hint": {
                "type": "integer",
                "description": "Your subjective 0–100 estimate; the server replaces fit_score with a deterministic value.",
                "minimum": 0,
                "maximum": 100,
            },
            "summary": {
                "type": "string",
                "description": "Short narrative of overall fit, strengths, and gaps.",
            },
            "recommendations": {
                "type": "array",
                "description": "One entry per gap (same order as gaps). Empty if gaps is empty.",
                "items": {
                    "type": "object",
                    "properties": {
                        "gap": {
                            "type": "string",
                            "description": "Must match the corresponding gap.requirement text.",
                        },
                        "suggestion": {
                            "type": "string",
                            "description": "Specific resume change: name a JD skill/keyword to add, how to frame a project or role from the resume, and what to edit. No generic career advice.",
                        },
                        "example_resume_line": {
                            "type": "string",
                            "description": "One concrete bullet or phrase tailored to the candidate's resume context.",
                        },
                    },
                    "required": ["gap", "suggestion", "example_resume_line"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["matches", "gaps", "fit_score_hint", "summary", "recommendations"],
        "additionalProperties": False,
    },
}


class FitMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    resume_evidence: str
    confidence: float
    importance: ImportanceLevel = "medium"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, x))

    @field_validator("importance", mode="before")
    @classmethod
    def _normalize_importance(cls, v: Any) -> str:
        s = str(v or "medium").strip().lower()
        if s in ("low", "medium", "high"):
            return s
        return "medium"


class FitGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    reason: str
    importance: ImportanceLevel = "medium"

    @field_validator("importance", mode="before")
    @classmethod
    def _normalize_importance(cls, v: Any) -> str:
        s = str(v or "medium").strip().lower()
        if s in ("low", "medium", "high"):
            return s
        return "medium"


class FitRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gap: str
    suggestion: str
    example_resume_line: str


class AnalyzeFitLLMResult(BaseModel):
    """Parsed model payload before deterministic scoring."""

    model_config = ConfigDict(extra="forbid")

    matches: list[FitMatch]
    gaps: list[FitGap]
    fit_score_hint: int = Field(ge=0, le=100)
    summary: str
    recommendations: list[FitRecommendation] = Field(default_factory=list)

    @field_validator("fit_score_hint", mode="before")
    @classmethod
    def _coerce_hint(cls, v: Any) -> int:
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return 0

    @field_validator("fit_score_hint")
    @classmethod
    def _clamp_hint(cls, v: int) -> int:
        return max(0, min(100, v))


class AnalyzeFitResult(BaseModel):
    """Validated API shape returned to callers."""

    model_config = ConfigDict(extra="forbid")

    matches: list[FitMatch]
    gaps: list[FitGap]
    summary: str
    fit_score: int = Field(ge=0, le=100)
    fit_score_hint: int = Field(ge=0, le=100)
    matched_count: int = Field(ge=0)
    total_requirements: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    gap_penalty: float = Field(ge=0)
    coverage_raw: float = Field(ge=0)
    recommendations: list[FitRecommendation] = Field(default_factory=list)


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_LOW_SIGNAL_SENTENCE = re.compile(
    r"^\s*(copyright|all rights reserved|equal opportunity|eoe|privacy policy|"
    r"click here|lorem ipsum|page\s+\d+\s+of\s+\d+)\b",
    re.I,
)
_JD_RESUME_SIGNALS = re.compile(
    r"\b(must|required|requirement|qualifications?|years?\s+of|experience|bachelor|master|ph\.?d|degree|"
    r"proficien|skilled?|skill|ability|familiar|knowledge|responsibilit|deliver|include|preferred|"
    r"nice\s*to\s*have|hands[- ]on|expertise|certif|stack|framework|tooling|kubernetes|docker|aws|gcp|azure|"
    r"sql|python|java|react|node|agile|stakeholder|led|built|develop|design|implement|manag|architect|engineer|"
    r"deploy|scal|team|project|role|scope)\b",
    re.I,
)
_TECH_OR_METRIC = re.compile(
    r"\b\d+\+?\s*years?\b|\b[A-Z]{2,6}\b(?:\s*/\s*[A-Z]{2,6})?\b|\b[\w.-]+\+?\b@\b[\w.-]+\b",
)


def _split_sentences(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    parts = [p.strip() for p in _SENTENCE_BOUNDARY.split(t) if p.strip()]
    if len(parts) <= 1 and ("\n" in (text or "")):
        parts = [
            re.sub(r"\s+", " ", line.strip())
            for line in re.split(r"[\n\r•·]+", text or "")
            if line.strip() and len(line.strip()) > 12
        ]
    return parts[:24]


def _sentence_score(sentence: str, skills_hint: list[Any]) -> float:
    if not sentence or _LOW_SIGNAL_SENTENCE.match(sentence):
        return -100.0
    s = sentence.strip()
    n = len(s)
    if n < 18:
        return -50.0
    score = 0.0
    score += 3.0 * len(_JD_RESUME_SIGNALS.findall(s))
    score += 2.0 * len(_TECH_OR_METRIC.findall(s))
    if 45 <= n <= 280:
        score += 2.5
    elif n > 360:
        score -= 1.5
    if skills_hint:
        low = s.lower()
        for sk in skills_hint:
            if not sk:
                continue
            t = str(sk).strip().lower()
            if len(t) >= 2 and t in low:
                score += 4.0
    return score


def _jaccard_words(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"[a-z0-9]{2,}", a.lower())}
    wb = {w for w in re.findall(r"[a-z0-9]{2,}", b.lower())}
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def _pick_top_sentences(text: str, skills_hint: list[Any], max_sentences: int) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    scored = [(s, _sentence_score(s, skills_hint)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    picked: list[str] = []
    for s, sc in scored:
        if sc < -40 and picked:
            continue
        if any(_jaccard_words(s, p) > 0.82 for p in picked):
            continue
        picked.append(s)
        if len(picked) >= max_sentences:
            break
    if not picked:
        best = max(scored, key=lambda x: x[1])
        picked = [best[0]] if best[1] > -80 else [sentences[0][: _MAX_CHARS_PER_COMPRESSED_CHUNK]]
    order = {id(s): i for i, s in enumerate(sentences)}
    picked.sort(key=lambda x: order.get(id(x), 0))
    out = " ".join(picked).strip()
    if len(out) > _MAX_CHARS_PER_COMPRESSED_CHUNK:
        out = out[: _MAX_CHARS_PER_COMPRESSED_CHUNK].rsplit(" ", 1)[0] + "..."
    return out


def _compress_chunk_body(raw: str, chunk: dict) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    skills = chunk.get("skills_detected") if isinstance(chunk.get("skills_detected"), list) else []
    if chunk.get("is_boilerplate") or chunk.get("is_low_signal"):
        snippet = _pick_top_sentences(raw[:_MAX_CHARS_PER_CHUNK], skills, _MIN_COMPRESS_SENTENCES)
        return snippet or raw[:200]
    return _pick_top_sentences(raw[:_MAX_CHARS_PER_CHUNK], skills, _MAX_COMPRESS_SENTENCES)


def compress_chunks(
    chunks: list[dict],
    *,
    max_combined_estimated_tokens: int = _ANALYZE_FIT_USER_EXCERPT_TOKEN_BUDGET,
) -> list[dict]:
    """
    Shrink chunk text for analyze-fit prompts: keep 1-3 highest-signal sentences per chunk
    (skills, tools, experience, requirements), drop boilerplate/low-signal lines, dedupe
    near-duplicate sentences, then trim least-relevant whole chunks (retrieval tail) until
    the formatted JOB+RESUME excerpt block is under ``max_combined_estimated_tokens``.
    """
    if not chunks:
        return []
    compressed: list[dict] = []
    for c in chunks:
        nc = dict(c)
        body = _compress_chunk_body(_chunk_excerpt_text(c), c)
        nc["snippet"] = body
        if "text" in nc:
            nc["text"] = body
        compressed.append(nc)

    jd, rs = split_chunks_for_fit(compressed)
    jd = jd[:_MAX_CHUNKS_PER_SIDE]
    rs = rs[:_MAX_CHUNKS_PER_SIDE]
    excerpt_budget = max(400, int(max_combined_estimated_tokens) - _ANALYZE_FIT_USER_FRAME_TOKEN_EST)

    def _excerpt_tokens(jd_: list[dict], rs_: list[dict]) -> int:
        jt = _format_side_for_prompt("JOB_EXCERPTS (requirements / role)", jd_)
        rt = _format_side_for_prompt("RESUME_EXCERPTS (candidate)", rs_)
        return estimate_tokens(jt) + estimate_tokens(rt)

    while _excerpt_tokens(jd, rs) > excerpt_budget and (len(jd) + len(rs) > 2):
        jd_tok = sum(estimate_tokens(_chunk_excerpt_text(c)) for c in jd)
        rs_tok = sum(estimate_tokens(_chunk_excerpt_text(c)) for c in rs)
        if len(rs) > 1 and (rs_tok >= jd_tok or len(jd) <= 1):
            rs.pop()
        elif len(jd) > 1:
            jd.pop()
        else:
            rs.pop()

    _shrink_excerpts_until_budget(jd, rs, excerpt_budget)

    keep = {id(x) for x in jd} | {id(x) for x in rs}
    return [c for c in compressed if id(c) in keep]


def _shrink_excerpts_until_budget(jd: list[dict], rs: list[dict], excerpt_budget: int) -> None:
    """Trim longest chunk bodies in place until formatted JOB+RESUME excerpts fit the token budget."""
    guard = 0

    def _tok(jd_: list[dict], rs_: list[dict]) -> int:
        jt = _format_side_for_prompt("JOB_EXCERPTS (requirements / role)", jd_)
        rt = _format_side_for_prompt("RESUME_EXCERPTS (candidate)", rs_)
        return estimate_tokens(jt) + estimate_tokens(rt)

    while _tok(jd, rs) > excerpt_budget and guard < 600:
        guard += 1
        pool = [(c, len(_chunk_excerpt_text(c))) for c in jd + rs if _chunk_excerpt_text(c)]
        if not pool:
            break
        longest, ln = max(pool, key=lambda x: x[1])
        if ln < 40:
            break
        txt = _chunk_excerpt_text(longest)
        new_len = max(32, ln - max(16, int(ln * 0.12)))
        shortened = txt[:new_len].rsplit(" ", 1)[0].strip()
        if len(shortened) < 16:
            shortened = txt[:32].strip()
        suffix = "..." if not shortened.endswith("...") else ""
        longest["snippet"] = shortened + suffix
        longest["text"] = longest["snippet"]


def split_chunks_for_fit(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Partition retrieved chunks into job-description vs resume buckets.

    Uses ``source_type`` when present (``JD`` / ``RESUME``). Chunks with
    ``OTHER`` or missing ``source_type`` are treated as JD context so
    requirement extraction still has material.
    """
    jd_chunks: list[dict] = []
    resume_chunks: list[dict] = []
    for c in chunks:
        st = str(c.get("source_type") or "").strip().upper()
        if st == "RESUME":
            resume_chunks.append(c)
        else:
            jd_chunks.append(c)
    return jd_chunks, resume_chunks


def _chunk_excerpt_text(c: dict) -> str:
    return (c.get("snippet") or c.get("text") or "").strip()


def _format_side_for_prompt(title: str, chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks[:_MAX_CHUNKS_PER_SIDE]):
        body = _chunk_excerpt_text(c)[:_MAX_CHARS_PER_CHUNK]
        if not body:
            continue
        meta_parts: list[str] = [f"idx={i}"]
        if c.get("chunk_id") or c.get("chunkId"):
            meta_parts.append(f"id={c.get('chunk_id') or c.get('chunkId')}")
        if c.get("page_number") is not None or c.get("page") is not None:
            meta_parts.append(f"page={c.get('page_number') if c.get('page_number') is not None else c.get('page')}")
        lines.append(f"[{' '.join(meta_parts)}]\n{body}")
    if not lines:
        return f"({title}: no excerpt text available.)"
    return f"## {title}\n\n" + "\n\n---\n\n".join(lines)


def _extract_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    return m.group(0) if m else raw


_SYSTEM_PROMPT = """You are an expert recruiter. In a single pass over the excerpts below, do all of the following in one structured response (no follow-up turns):

1) Requirement extraction — Derive concrete requirements and qualifications from JOB_EXCERPTS only.
2) Matching — For each requirement with adequate resume support, output a row in "matches" with resume_evidence from RESUME_EXCERPTS.
3) Gap detection — Requirements missing or clearly weak in the resume go in "gaps" with a short reason.
4) Scoring hint — Set "fit_score_hint" to your rough 0–100 subjective estimate; the server IGNORES it and computes the final fit_score from matches/gaps.
5) Recommendations — One recommendation object per gap, same order as "gaps"; if gaps is empty, recommendations must be [].

Rules:
- Use ONLY information present in JOB_EXCERPTS and RESUME_EXCERPTS. If the resume does not mention something, do not invent experience.
- Each "requirement" in matches and gaps must be traceable to JOB_EXCERPTS (paraphrase is OK; do not fabricate JD content).
- resume_evidence must quote or tightly paraphrase RESUME_EXCERPTS; use an empty string if there is no supporting resume text.
- confidence: 0.0–1.0 for how strongly the resume supports the requirement (use low values if evidence is thin).
- importance (each match and gap): "low" (nice-to-have), "medium" (standard), "high" (must-have / critical skill or qualification).
- gaps: JD requirements that are missing or clearly unsupported in the resume; mark importance "high" when the gap is critical.
- summary: 2–5 sentences, neutral and specific.
- recommendations: each item's "gap" must echo that gap's requirement text; "suggestion" must be specific—name an exact skill or keyword from the JD to add, propose how to reframe a real project or role already visible in RESUME_EXCERPTS (do not invent employers or jobs not in the excerpts), and say what to change on the resume. Ban vague phrases like "highlight your strengths", "tailor your resume", "develop skills", or "improve communication".
  "example_resume_line": one plausible bullet or phrase the candidate could use, consistent with their resume when possible.

Output must conform exactly to the JSON schema you are given (no markdown, no extra keys)."""


def _coerce_llm_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy / json_object fallback shapes before Pydantic validation."""
    if "fit_score_hint" not in data and "fit_score" in data:
        data = {**data, "fit_score_hint": data["fit_score"]}
    for m in data.get("matches") or []:
        if isinstance(m, dict) and "importance" not in m:
            m["importance"] = "medium"
    for g in data.get("gaps") or []:
        if isinstance(g, dict) and "importance" not in g:
            g["importance"] = "medium"
    if "recommendations" not in data or data["recommendations"] is None:
        data["recommendations"] = []
    return data


def _parse_llm_json(content: str) -> AnalyzeFitLLMResult:
    """Parse model output; strips accidental markdown; validates strictly (no extra fields)."""
    cleaned = _extract_json_object(content)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise TypeError("root JSON must be an object")
    return AnalyzeFitLLMResult.model_validate(_coerce_llm_payload(data))


def _align_recommendations_to_gaps(
    gaps: list[FitGap],
    recommendations: list[FitRecommendation],
) -> list[FitRecommendation]:
    """
    Bind recommendations to gaps by index; ``gap`` text is taken from the gap
    record (source of truth). Drops extras; missing indices yield no row.
    """
    out: list[FitRecommendation] = []
    for i, g in enumerate(gaps):
        if i >= len(recommendations):
            break
        r = recommendations[i]
        out.append(
            FitRecommendation(
                gap=g.requirement.strip(),
                suggestion=(r.suggestion or "").strip(),
                example_resume_line=(r.example_resume_line or "").strip(),
            )
        )
    if gaps and len(out) < len(gaps):
        logger.warning(
            "analyze_fit: %s gap(s) missing recommendations (model returned %s)",
            len(gaps) - len(out),
            len(recommendations),
        )
    return out


def _finalize_from_llm(llm: AnalyzeFitLLMResult) -> AnalyzeFitResult:
    m_dicts = [x.model_dump() for x in llm.matches]
    g_dicts = [x.model_dump() for x in llm.gaps]
    scores = compute_fit_score(m_dicts, g_dicts)
    recs = (
        []
        if not llm.gaps
        else _align_recommendations_to_gaps(llm.gaps, llm.recommendations)
    )
    return AnalyzeFitResult(
        matches=llm.matches,
        gaps=llm.gaps,
        summary=llm.summary,
        fit_score=scores["fit_score"],
        fit_score_hint=llm.fit_score_hint,
        matched_count=scores["matched_count"],
        total_requirements=scores["total_requirements"],
        gap_count=scores["gap_count"],
        gap_penalty=scores["gap_penalty"],
        coverage_raw=scores["coverage_raw"],
        recommendations=recs,
    )


def _empty_result(summary: str) -> dict[str, Any]:
    return AnalyzeFitResult(
        matches=[],
        gaps=[],
        summary=summary,
        fit_score=0,
        fit_score_hint=0,
        matched_count=0,
        total_requirements=0,
        gap_count=0,
        gap_penalty=0.0,
        coverage_raw=0.0,
        recommendations=[],
    ).model_dump()


def analyze_fit(
    query: str,
    retrieved_chunks: list[dict],
    user_id: uuid.UUID,
) -> dict[str, Any]:
    """
    Run fit analysis over multi-document retrieved chunks.

    Uses a single chat-completions call (structured JSON schema) for extraction,
    matching, gaps, scoring hint, summary, and recommendations.

    Args:
        query: User question or focus (e.g. "How well do I fit this role?").
        retrieved_chunks: Chunk dicts from retrieval (must include text/snippet; ``source_type`` preferred).
            Passed through :func:`compress_chunks` before prompting the model.
        user_id: Caller user id (logged for audit; does not load DB state).

    Returns:
        Dict with ``matches``, ``gaps``, ``summary``, deterministic ``fit_score``,
        ``fit_score_hint`` (model suggestion), ``matched_count``, ``total_requirements``,
        ``gap_count``, ``gap_penalty``, ``coverage_raw``, and ``recommendations``
        (actionable items aligned to gaps).

    Raises:
        ValueError: if OpenAI is not configured.
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    want_completion = min(4500, max(1100, settings.max_completion_tokens * 7))
    excerpt_cap = max(
        400,
        min(
            _ANALYZE_FIT_USER_EXCERPT_TOKEN_BUDGET,
            settings.max_llm_budget_tokens
            - want_completion
            - estimate_tokens(_SYSTEM_PROMPT)
            - TOKEN_BUDGET_SAFETY_SLACK
            - 120,
        ),
    )
    prepped_chunks = compress_chunks(
        retrieved_chunks,
        max_combined_estimated_tokens=excerpt_cap,
    )
    jd_chunks, resume_chunks = split_chunks_for_fit(prepped_chunks)
    jd_text = _format_side_for_prompt("JOB_EXCERPTS (requirements / role)", jd_chunks)
    resume_text = _format_side_for_prompt("RESUME_EXCERPTS (candidate)", resume_chunks)

    jd_has_text = any(bool(_chunk_excerpt_text(c)) for c in jd_chunks)
    resume_has_text = any(bool(_chunk_excerpt_text(c)) for c in resume_chunks)

    if not jd_has_text and not resume_has_text:
        logger.info("analyze_fit: no excerpt text user_id=%s", user_id)
        return _empty_result(
            "There is no job description or resume text in the retrieved excerpts to analyze."
        )

    user_content = f"""User focus / question:
{query.strip() or "(general fit)"}

Requesting user_id (for traceability only): {user_id}

{jd_text}

{resume_text}

In one response, output the JSON schema: extract requirements from job excerpts, match and gap-analyze against resume excerpts, set fit_score_hint (advisory only), write summary, and recommendations (one per gap, same order as gaps). The server computes the final fit_score from matches and gaps."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    max_tokens = want_completion
    input_est = estimate_tokens(_SYSTEM_PROMPT) + estimate_tokens(user_content)
    max_tokens = min(
        max_tokens,
        settings.max_llm_budget_tokens - input_est - TOKEN_BUDGET_SAFETY_SLACK,
    )
    max_tokens = max(400, max_tokens)
    client = OpenAI(api_key=settings.openai_api_key)

    llm_call_count = 0

    def _chat_completions_create_once(**kwargs: Any) -> Any:
        nonlocal llm_call_count
        llm_call_count += 1
        if llm_call_count > 1:
            logger.error(
                "analyze_fit: invariant violated — %s LLM calls in one request (expected 1); "
                "check for regressions that add extra completions",
                llm_call_count,
            )
        assert llm_call_count <= 1, "analyze_fit: at most one LLM call per request"
        return client.chat.completions.create(**kwargs)

    create_kwargs: dict[str, Any] = {
        "model": settings.chat_model_fit_analysis(),
        "messages": messages,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": _ANALYZE_FIT_JSON_SCHEMA,
        },
    }

    try:
        resp = _chat_completions_create_once(**create_kwargs)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception(
            "analyze_fit: chat completion failed user_id=%s: %s",
            user_id,
            e,
        )
        return _empty_result(
            "The fit analysis could not be completed because the model request failed. Try again."
        )

    try:
        llm = _parse_llm_json(raw)
    except Exception as e:
        logger.exception("analyze_fit: validate failed user_id=%s: %s", user_id, e)
        return _empty_result(
            "The fit analysis could not be parsed. Try again with clearer retrieved excerpts."
        )

    finalized = _finalize_from_llm(llm)

    logger.info(
        "analyze_fit: ok user_id=%s matched=%s total=%s gaps=%s fit_score=%s hint=%s",
        user_id,
        finalized.matched_count,
        finalized.total_requirements,
        finalized.gap_count,
        finalized.fit_score,
        finalized.fit_score_hint,
    )
    return finalized.model_dump()
