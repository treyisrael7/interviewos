"""JD-aware chunking. Chunk by semantic section, keep bullets intact."""

import hashlib
import logging
import re
from dataclasses import dataclass

from app.core.config import settings
from app.services.chunking import (
    _compute_quality_metrics,
    _content_hash,
    _is_low_signal,
    _quality_score,
)
from app.services.jd_extraction import _extract_skills_from_text
from app.services.jd_sections import normalize_jd_text, sectionize_jd_text

logger = logging.getLogger(__name__)

JD_DOMAIN = "job_description"
MAX_CHARS_PER_CHUNK = 500  # Split larger sections for more granular retrieval


@dataclass
class JDChunkResult:
    page_number: int
    content: str
    chunk_index: int
    quality_score: float
    is_low_signal: bool
    content_hash: str
    section_type: str
    skills_detected: list[str]
    doc_domain: str = JD_DOMAIN


def _split_section_into_chunks(
    section_type: str, content: str, page_num: int, max_chars: int
) -> list[tuple[str, str]]:
    """
    Split a section into chunks. Keep bullet lists intact.
    Returns list of (chunk_content, section_type).
    """
    if not content.strip():
        return []
    if len(content) <= max_chars:
        return [(content.strip(), section_type)]

    # Split by bullet blocks (groups of bullets)
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append("\n".join(current))
                current = []
                current_len = 0
            continue
        line_len = len(stripped) + (1 if current else 0)
        # If adding would exceed limit and we have content, start new block
        if current_len + line_len > max_chars and current:
            blocks.append("\n".join(current))
            current = [stripped]
            current_len = len(stripped)
        else:
            current.append(stripped)
            current_len += line_len
    if current:
        blocks.append("\n".join(current))

    return [(b, section_type) for b in blocks if len(b) >= 25]


def chunk_jd_pages(
    page_texts: list[tuple[int, str]],
    min_chars: int = 25,
    max_chunks: int = 300,
) -> list[JDChunkResult]:
    """
    Chunk JD text by semantic section. Keeps bullet lists intact.
    Tags each chunk with section_type, skills_detected, doc_domain=job_description.
    """
    full_text = "\n\n".join(t for _, t in page_texts)
    norm_text = normalize_jd_text(full_text)
    sections = sectionize_jd_text(norm_text)

    results: list[JDChunkResult] = []
    chunk_idx = 0

    for section_type, content in sections:
        if chunk_idx >= max_chunks:
            break
        sub_chunks = _split_section_into_chunks(
            section_type, content, page_num=1, max_chars=MAX_CHARS_PER_CHUNK
        )
        for chunk_content, sec in sub_chunks:
            if chunk_idx >= max_chunks or len(chunk_content) < min_chars:
                continue
            metrics = _compute_quality_metrics(chunk_content)
            qs = _quality_score(metrics)
            low = _is_low_signal(metrics)
            chash = _content_hash(chunk_content)
            skills = _extract_skills_from_text(chunk_content)

            results.append(
                JDChunkResult(
                    page_number=1,
                    content=chunk_content,
                    chunk_index=chunk_idx,
                    quality_score=round(qs, 4),
                    is_low_signal=low,
                    content_hash=chash,
                    section_type=sec,
                    skills_detected=skills,
                    doc_domain=JD_DOMAIN,
                )
            )
            chunk_idx += 1

    logger.info(
        "chunk_jd_pages done: sections=%s chunks=%s (max_chars=%s)",
        len(sections),
        len(results),
        MAX_CHARS_PER_CHUNK,
    )
    return results
