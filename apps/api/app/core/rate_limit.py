"""In-memory rate limiter. Swap to Redis for multi-instance scaling."""

from collections import defaultdict
from time import time
from typing import Literal

RouteKey = Literal[
    "ask",
    "analyze-fit",
    "study-plan",
    "fit-history",
    "retrieve",
    "documents/ingest",
    "documents/presign",
    "documents/confirm",
]

# (limit, window_seconds)
RATE_LIMITS: dict[str, tuple[int, int]] = {
    "ask": (10, 3600),  # 10 per hour
    "analyze-fit": (10, 3600),  # 10 per hour (LLM + retrieval)
    "study-plan": (10, 3600),  # 10 per hour (LLM plan generation)
    "fit-history": (120, 3600),  # read-heavy; generous cap
    "retrieve": (60, 3600),  # 60 per hour
    "documents/ingest": (3, 86400),  # 3 per day
    "documents/presign": (10, 86400),  # 10 per day
    "documents/confirm": (20, 86400),  # 20 per day
}

def _path_to_route(path: str) -> RouteKey | None:
    """Map path to route key. Supports /documents/{id}/ingest pattern."""
    path = path.rstrip("/") or "/"
    if path == "/ask":
        return "ask"
    if path == "/user/resume/ask":
        return "ask"
    if path == "/analyze-fit":
        return "analyze-fit"
    if path == "/analyze-fit/latest":
        return "fit-history"
    if path == "/fit-history":
        return "fit-history"
    if path == "/retrieve":
        return "retrieve"
    if path == "/documents/presign":
        return "documents/presign"
    if path == "/documents/confirm":
        return "documents/confirm"
    # /documents/{uuid}/ingest
    if path.startswith("/documents/") and path.endswith("/ingest"):
        return "documents/ingest"
    if path.startswith("/documents/") and path.endswith("/study-plan"):
        return "study-plan"
    return None

# in-memory: {key: [timestamp, ...]}
_store: defaultdict[str, list[float]] = defaultdict(list)


def clear_store() -> None:
    """Clear rate limit store (for testing only)."""
    _store.clear()


def _make_key(ip: str, user_id: str | None, route: RouteKey) -> str:
    if user_id:
        return f"{user_id}:{route}"
    return f"{ip}:{route}"


def _prune(timestamps: list[float], window_seconds: int) -> list[float]:
    cutoff = time() - window_seconds
    return [t for t in timestamps if t > cutoff]


def check_rate_limit(
    ip: str,
    path: str,
    user_id: str | None = None,
) -> tuple[bool, int | None]:
    """
    Check if request is within rate limit.
    Returns (allowed, retry_after_seconds or None if allowed).
    """
    path = path.rstrip("/") or "/"
    route = _path_to_route(path)
    if not route:
        return True, None

    limit, window = RATE_LIMITS[route]
    key = _make_key(ip, user_id, route)

    now = time()
    timestamps = _store[key]
    timestamps = _prune(timestamps, window)
    _store[key] = timestamps  # already pruned

    if len(timestamps) >= limit:
        oldest = min(timestamps) if timestamps else now
        retry_after = int(window - (now - oldest))
        retry_after = max(1, retry_after)
        return False, retry_after

    timestamps.append(now)
    return True, None
