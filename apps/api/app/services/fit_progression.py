"""Derive score progression messages from stored fit_analyses rows."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any


def build_fit_progression(
    rows: list[Any],
    *,
    max_pairs: int = 200,
) -> list[dict[str, Any]]:
    """
    Group ORM rows (or any objects with job_description_id, resume_id, fit_score, created_at)
    by job/resume pair and compute first vs latest scores.

    ``improved`` is ``True``/``False`` when there are at least two runs; ``None`` for a single run.
    """
    groups: dict[tuple[uuid.UUID, uuid.UUID], list[Any]] = defaultdict(list)
    for r in rows:
        key = (r.job_description_id, r.resume_id)
        groups[key].append(r)

    out: list[dict[str, Any]] = []
    for (jd, rs), series in groups.items():
        series = sorted(series, key=lambda x: x.created_at)
        first = int(series[0].fit_score)
        latest = int(series[-1].fit_score)
        delta = latest - first
        n = len(series)
        if n >= 2 and delta > 0:
            message = f"Your fit improved from {first} → {latest}."
        elif n >= 2 and delta < 0:
            message = f"Your fit score went from {first} → {latest}."
        elif n >= 2:
            message = f"Your fit score stayed at {latest} across {n} runs."
        else:
            message = "Only one analysis so far for this job description and resume."

        out.append(
            {
                "job_description_id": str(jd),
                "resume_id": str(rs),
                "first_score": first,
                "latest_score": latest,
                "delta": delta,
                "run_count": n,
                "improved": (delta > 0) if n >= 2 else None,
                "message": message,
            }
        )

    out.sort(key=lambda x: (-x["run_count"], x["job_description_id"], x["resume_id"]))
    return out[:max_pairs]
