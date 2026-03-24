"""Deterministic, structured interview answer scoring (0–100 per dimension).

Used for measurable progress; narrative feedback may still come from the LLM separately.
"""

from __future__ import annotations

import re
from typing import Any

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "that",
        "this",
        "with",
        "have",
        "from",
        "they",
        "been",
        "into",
        "more",
        "some",
        "than",
        "then",
        "them",
        "these",
        "when",
        "what",
        "your",
        "will",
        "about",
        "after",
        "also",
        "back",
        "because",
        "could",
        "first",
        "just",
        "like",
        "make",
        "most",
        "only",
        "other",
        "over",
        "such",
        "their",
        "there",
        "very",
        "well",
        "were",
        "would",
    }
)

_WEIGHTS = {
    "relevance_to_context": 0.28,
    "completeness": 0.28,
    "clarity": 0.18,
    "jd_alignment": 0.26,
}


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


def _tokens(s: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", _normalize(s))
        if len(t) > 2 and t not in _STOPWORDS
    ]


def _token_set(s: str) -> set[str]:
    return set(_tokens(s))


def _word_count(s: str) -> int:
    return len(re.findall(r"[a-zA-Z0-9']+", s or ""))


def _sentence_count(s: str) -> int:
    if not (s or "").strip():
        return 0
    parts = re.split(r"[.!?]+", s)
    return max(1, len([p for p in parts if p.strip()]))


def score_relevance_to_context(answer: str, evidence: list[dict]) -> int:
    """Overlap between answer tokens and retrieved JD/auxiliary evidence text."""
    a = _token_set(answer)
    if not a:
        return 0
    ev_text = " ".join(str(e.get("snippet") or e.get("text") or "") for e in evidence or [])
    e = _token_set(ev_text)
    if not e:
        return 35
    inter = len(a & e)
    union = len(a | e)
    jaccard = inter / union if union else 0.0
    coverage = inter / len(e) if e else 0.0
    raw = 55.0 * jaccard + 45.0 * min(1.0, 2.5 * coverage)
    return int(round(min(100.0, max(0.0, raw))))


def score_completeness(
    answer: str,
    what_good_looks_like: list[str],
    must_mention: list[str],
) -> int:
    """Coverage of rubric bullets and required phrases (deterministic checks)."""
    a_norm = _normalize(answer)
    a_toks = _token_set(answer)

    bullet_hits = 0
    bullets = [str(b).strip() for b in (what_good_looks_like or []) if str(b).strip()]
    for b in bullets:
        bt = _token_set(b)
        if not bt:
            continue
        need = max(1, min(3, (len(bt) + 1) // 2))
        if len(a_toks & bt) >= need:
            bullet_hits += 1
    bullet_ratio = bullet_hits / len(bullets) if bullets else 1.0

    must = [str(m).strip() for m in (must_mention or []) if str(m).strip()]
    must_hits = 0
    for m in must:
        if m.lower() in a_norm or all(
            w in a_norm for w in re.findall(r"[a-z0-9]+", m.lower()) if len(w) > 2
        ):
            must_hits += 1
    must_ratio = must_hits / len(must) if must else 1.0

    if bullets and must:
        raw = 0.55 * bullet_ratio + 0.45 * must_ratio
    elif bullets:
        raw = bullet_ratio
    elif must:
        raw = must_ratio
    else:
        raw = 0.65
    return int(round(min(100.0, max(0.0, raw * 100.0))))


def score_clarity(answer: str) -> int:
    """Lightweight length and structure heuristics (no ML)."""
    wc = _word_count(answer)
    sc = _sentence_count(answer)
    if wc <= 0:
        return 0
    avg_len = wc / max(1, sc)

    if wc < 12:
        base = 28 + min(52, wc * 4)
    elif wc < 30:
        base = 55 + int((wc - 12) * 2.2)
    elif wc <= 220:
        base = 100
    elif wc <= 380:
        base = int(100 - (wc - 220) * 0.22)
    else:
        base = max(38, int(64 - (wc - 380) * 0.06))

    if avg_len > 52:
        base = int(max(0, base - min(22, int((avg_len - 52) * 0.8))))
    if sc == 1 and wc > 120:
        base = int(max(0, base - 8))

    return int(round(min(100.0, max(0.0, float(base)))))


def score_jd_alignment(
    answer: str,
    role_profile: dict,
    competency_label: str | None,
    what_good_looks_like: list[str],
    evidence: list[dict],
) -> int:
    """Alignment with role focus areas, competency, rubric, and evidence themes."""
    parts: list[str] = []
    rp = role_profile or {}
    for fa in rp.get("focusAreas") or []:
        parts.append(str(fa))
    if competency_label:
        parts.append(str(competency_label))
    for b in what_good_looks_like or []:
        parts.append(str(b))
    parts.append(" ".join(str(e.get("snippet") or "")[:400] for e in (evidence or [])[:6]))
    jd_text = " ".join(parts)
    jd_toks = _token_set(jd_text)
    a_toks = _token_set(answer)
    if not a_toks:
        return 0
    if not jd_toks:
        return 45
    inter = len(a_toks & jd_toks)
    union = len(a_toks | jd_toks)
    jaccard = inter / union if union else 0.0
    coverage = inter / len(jd_toks) if jd_toks else 0.0
    raw = 50.0 * jaccard + 50.0 * min(1.0, 2.2 * coverage)
    return int(round(min(100.0, max(0.0, raw))))


def compute_score_breakdown(
    user_answer: str,
    evidence: list[dict],
    what_good_looks_like: list[str],
    must_mention: list[str],
    role_profile: dict,
    competency_label: str | None,
) -> dict[str, Any]:
    """
    Returns a deterministic breakdown; overall is a fixed weighted sum of dimensions.
    All values are integers 0–100 for stable storage and APIs.
    """
    r = score_relevance_to_context(user_answer, evidence)
    c = score_completeness(user_answer, what_good_looks_like, must_mention)
    cl = score_clarity(user_answer)
    j = score_jd_alignment(
        user_answer,
        role_profile,
        competency_label,
        what_good_looks_like,
        evidence,
    )
    overall = int(
        round(
            _WEIGHTS["relevance_to_context"] * r
            + _WEIGHTS["completeness"] * c
            + _WEIGHTS["clarity"] * cl
            + _WEIGHTS["jd_alignment"] * j
        )
    )
    overall = max(0, min(100, overall))
    return {
        "relevance_to_context": r,
        "completeness": c,
        "clarity": cl,
        "jd_alignment": j,
        "overall": overall,
        "weights": dict(_WEIGHTS),
    }


def build_feedback_summary(breakdown: dict[str, Any]) -> str:
    """Short deterministic summary line derived only from numeric breakdown."""
    r = breakdown.get("relevance_to_context", 0)
    c = breakdown.get("completeness", 0)
    cl = breakdown.get("clarity", 0)
    j = breakdown.get("jd_alignment", 0)
    o = breakdown.get("overall", 0)
    dims = [
        ("relevance to context", r),
        ("completeness", c),
        ("clarity", cl),
        ("JD alignment", j),
    ]
    dims_sorted = sorted(dims, key=lambda x: x[1], reverse=True)
    strongest = dims_sorted[0][0]
    weakest = dims_sorted[-1][0]
    return (
        f"Overall score {o}/100 (relevance {r}, completeness {c}, clarity {cl}, JD alignment {j}). "
        f"Strongest dimension: {strongest}. Focus next: {weakest}."
    )
