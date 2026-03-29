"""Heuristic token estimation and budgeting before LLM calls (cost guard).

Uses a character-based estimate (~4 chars/token) consistent with analyze-fit
compression. Actual tokenizer counts can differ; :data:`TOKEN_BUDGET_SAFETY_SLACK`
keeps requests under the configured cap.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 4
TOKEN_BUDGET_SAFETY_SLACK = 64
MIN_QA_COMPLETION_TOKENS = 256


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE)


def retrieval_score(chunk: dict) -> float:
    for key in ("final_score", "score", "semantic_score"):
        v = chunk.get(key)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def clamp_completion_to_budget(
    *,
    estimated_input_tokens: int,
    requested_completion_tokens: int,
    total_budget: int,
    slack: int = TOKEN_BUDGET_SAFETY_SLACK,
) -> int:
    """Cap completion ``max_tokens`` so estimated input + completion stays under ``total_budget``."""
    room = total_budget - estimated_input_tokens - slack
    if room < 1:
        return 1
    return min(max(1, requested_completion_tokens), room)


def budget_grounded_qa_prompt(
    *,
    question: str,
    chunks: list[dict],
    system_prompt: str,
    build_user_content: Callable[[list[dict], list[dict]], str],
    split_chunks: Callable[[list[dict]], tuple[list[dict], list[dict]]],
    requested_completion_tokens: int,
    total_budget: int,
    min_completion_tokens: int = MIN_QA_COMPLETION_TOKENS,
    slack: int = TOKEN_BUDGET_SAFETY_SLACK,
) -> tuple[list[dict], str, int]:
    """
    Drop lowest-scoring tail chunks (JD vs resume tails) and shorten snippets until
    ``estimate_tokens(system) + estimate_tokens(user) + requested_completion`` fits
    ``total_budget`` (with slack). Then set completion to
    ``min(requested, budget - input - slack)``.

    Returns ``(chunks_used_jd_then_resume, user_content, effective_max_completion_tokens)``.
    """
    jd, rs = split_chunks([{**c} for c in chunks])
    jd.sort(key=retrieval_score, reverse=True)
    rs.sort(key=retrieval_score, reverse=True)

    def _shorten_longest_snippet() -> bool:
        pool = [(c, len((c.get("snippet") or c.get("text") or ""))) for c in jd + rs]
        pool = [(c, ln) for c, ln in pool if ln > 48]
        if not pool:
            return False
        target, _ = max(pool, key=lambda x: x[1])
        raw = (target.get("snippet") or target.get("text") or "").strip()
        if len(raw) < 48:
            return False
        new_len = max(32, int(len(raw) * 0.82))
        shortened = raw[:new_len].rsplit(" ", 1)[0].strip() or raw[:32]
        suffix = "..." if not shortened.endswith("...") else ""
        target["snippet"] = shortened + suffix
        if "text" in target:
            target["text"] = target["snippet"]
        return True

    def _drop_lowest_tail() -> bool:
        if len(jd) + len(rs) <= 1:
            return False
        sides: list[tuple[list[dict], float]] = []
        if jd:
            sides.append((jd, retrieval_score(jd[-1])))
        if rs:
            sides.append((rs, retrieval_score(rs[-1])))
        sides.sort(key=lambda x: x[1])
        for lst, _ in sides:
            if len(lst) > 1:
                lst.pop()
                return True
        return False

    guard = 0
    while guard < 512:
        guard += 1
        user_content = build_user_content(jd, rs)
        input_est = estimate_tokens(system_prompt) + estimate_tokens(user_content)
        completion_cap = clamp_completion_to_budget(
            estimated_input_tokens=input_est,
            requested_completion_tokens=requested_completion_tokens,
            total_budget=total_budget,
            slack=slack,
        )
        if completion_cap >= min_completion_tokens:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "token_budget qa: input_est=%s completion_cap=%s jd=%s rs=%s budget=%s",
                    input_est,
                    completion_cap,
                    len(jd),
                    len(rs),
                    total_budget,
                )
            used = jd + rs
            return used, user_content, completion_cap
        if _drop_lowest_tail():
            continue
        if _shorten_longest_snippet():
            continue
        user_content = build_user_content(jd, rs)
        input_est = estimate_tokens(system_prompt) + estimate_tokens(user_content)
        completion_cap = clamp_completion_to_budget(
            estimated_input_tokens=input_est,
            requested_completion_tokens=requested_completion_tokens,
            total_budget=total_budget,
            slack=slack,
        )
        if logger.isEnabledFor(logging.WARNING):
            logger.warning(
                "token_budget qa: hard cap hit input_est=%s completion_cap=%s budget=%s",
                input_est,
                completion_cap,
                total_budget,
            )
        return jd + rs, user_content, max(1, completion_cap)

    user_content = build_user_content(jd, rs)
    input_est = estimate_tokens(system_prompt) + estimate_tokens(user_content)
    completion_cap = clamp_completion_to_budget(
        estimated_input_tokens=input_est,
        requested_completion_tokens=requested_completion_tokens,
        total_budget=total_budget,
        slack=slack,
    )
    return jd + rs, user_content, max(1, completion_cap)
