"""Paragraph-based chunking with quality scoring. Document-agnostic."""

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+")

# Bullet/heading markers for line-based paragraph splitting
BULLET_RE = re.compile(r"^\s*[•\-*]\s+")
# Short line that looks like a heading (no trailing period, < 80 chars, mostly alpha)
HEADING_LIKE_RE = re.compile(r"^[A-Za-z][^.]{0,78}$")
# Common section titles (document-agnostic patterns)
SECTION_TITLES = frozenset(
    "experience education summary skills contact qualifications projects certifications".split()
)


def normalize_text(text: str) -> str:
    """Replace non-breaking spaces, remove Â artifacts, collapse whitespace."""
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = t.replace("Â", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _looks_like_heading(line: str) -> bool:
    """True if line appears to be a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    # Bullet
    if BULLET_RE.match(line):
        return True
    # Short, no period, first word could be section title
    if "." not in stripped and " " in stripped:
        first = stripped.split()[0].lower()
        if first in SECTION_TITLES:
            return True
    # Very short line (< 40 chars) with no period
    if len(stripped) < 40 and "." not in stripped:
        return True
    return False


def _paragraphs_line_fallback(text: str, max_para_chars: int = 500) -> list[str]:
    """
    Fallback for PDFs with no blank lines: build paragraphs by grouping lines.
    Start new paragraph on: empty line, bullet/heading line, or when current exceeds max_para_chars.
    """
    lines = text.split("\n")
    result: list[str] = []
    acc: list[str] = []
    size = 0

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            if acc:
                result.append("\n".join(acc))
                acc = []
                size = 0
            continue
        add = len(stripped) + (1 if acc else 0)
        # New paragraph: bullet/heading, or would exceed limit
        if _looks_like_heading(ln) and acc:
            result.append("\n".join(acc))
            acc = [stripped]
            size = len(stripped)
        elif size + add > max_para_chars and acc:
            result.append("\n".join(acc))
            acc = [stripped]
            size = len(stripped)
        else:
            acc.append(stripped)
            size += add
    if acc:
        result.append("\n".join(acc))
    return result


def _paragraphs(text: str, max_para_chars: int = 500) -> list[str]:
    """
    Split text into paragraphs. Primary: blank lines. Fallback: line-based grouping
    for single-block PDF extractions (no blank lines).
    """
    blocks = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in blocks if p.strip()]
    # If we got very few paragraphs and text is substantial, use line-based fallback
    total_chars = len(text.strip())
    if len(paras) <= 2 and total_chars > 100:
        paras = _paragraphs_line_fallback(text, max_para_chars)
    return paras


def _split_oversized_paragraph(para: str, max_chars: int) -> list[str]:
    """
    Split a paragraph that exceeds max_chars into smaller pieces.
    Splits by single newlines first (preserves logical lines).
    Only splits when the whole paragraph exceeds max_chars; does not split
    individual long lines (avoids tiny fragments that break tests).
    Document-agnostic fallback for PDFs with minimal blank-line structure.
    """
    if len(para) <= max_chars:
        return [para]
    lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
    if not lines:
        return [para[i : i + max_chars] for i in range(0, len(para), max_chars)]
    result: list[str] = []
    acc: list[str] = []
    size = 0
    for ln in lines:
        add = 1 + len(ln) if acc else len(ln)
        if size + add > max_chars and acc:
            result.append("\n".join(acc))
            acc = [ln]
            size = len(ln)
        else:
            acc.append(ln)
            size += add
    if acc:
        result.append("\n".join(acc))
    return result


def _chunk_by_paragraphs(
    page_num: int,
    paragraphs: list[str],
    max_chars: int,
    overlap_paragraphs: int,
    min_chars: int,
) -> list[tuple[int, str]]:
    """Build chunks by accumulating paragraphs. Overlap by N paragraphs."""
    chunks: list[tuple[int, str]] = []
    overlap = max(0, overlap_paragraphs)
    i = 0
    while i < len(paragraphs):
        acc: list[str] = []
        size = 0
        j = i
        while j < len(paragraphs) and (size + len(paragraphs[j]) + (1 if acc else 0) <= max_chars or not acc):
            p = paragraphs[j]
            if acc:
                size += 1  # newline
            size += len(p)
            acc.append(p)
            j += 1
        chunk = "\n\n".join(acc).strip()
        if chunk and len(chunk) >= min_chars:
            chunks.append((page_num, chunk))
        step = max(1, len(acc) - overlap)
        i += step
    return chunks


def _compute_quality_metrics(text: str) -> dict:
    """Compute document-agnostic quality metrics for a chunk."""
    non_ws = [c for c in text if not c.isspace()]
    total = len(non_ws)
    alpha = sum(1 for c in non_ws if c.isalpha())
    digit = sum(1 for c in non_ws if c.isdigit())
    alpha_ratio = alpha / total if total else 0.0
    digit_ratio = digit / total if total else 0.0
    url_count = len(URL_RE.findall(text))
    email_count = len(EMAIL_RE.findall(text))
    phone_count = len(PHONE_RE.findall(text))
    words = re.findall(r"\b\w+\b", text.lower())
    unique = len(set(words))
    unique_word_ratio = unique / len(words) if words else 0.0
    length_chars = len(text)
    return {
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "url_count": url_count,
        "email_count": email_count,
        "phone_count": phone_count,
        "unique_word_ratio": unique_word_ratio,
        "length_chars": length_chars,
    }


def _content_hash(normalized_text: str) -> str:
    """SHA1 hash of normalized content for duplicate detection."""
    return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()


def _quality_score(metrics: dict) -> float:
    """Weighted heuristic quality score 0..1."""
    a = metrics["alpha_ratio"]
    u = metrics["unique_word_ratio"]
    L = metrics["length_chars"]
    contact = metrics["email_count"] + metrics["url_count"] + metrics["phone_count"]
    score = 0.0
    score += 0.35 * min(1.0, a / 0.6)
    score += 0.35 * min(1.0, u / 0.5)
    score += 0.20 * min(1.0, L / 400)
    score -= 0.15 * min(1.0, contact * 0.5)
    return max(0.0, min(1.0, score))


def _is_low_signal(metrics: dict) -> bool:
    """
    Mark chunk as low-signal (boilerplate/nav/low-info). Document-agnostic.
    True if ANY condition matches.
    """
    a = metrics["alpha_ratio"]
    u = metrics["unique_word_ratio"]
    L = metrics["length_chars"]
    d = metrics["digit_ratio"]
    contact = metrics["email_count"] + metrics["url_count"] + metrics["phone_count"]
    phone_count = metrics["phone_count"]
    email_count = metrics["email_count"]
    url_count = metrics["url_count"]

    if L < 80:
        return True
    if u < 0.25:
        return True
    if a < 0.35:
        return True
    if contact >= 1 and L < 500:
        return True
    if phone_count >= 1 and d > 0.08:
        return True
    if contact >= 2 and L < 400:
        return True
    # Email + URL together = contact block (letterheads, footers, any doc)
    if email_count >= 1 and url_count >= 1:
        return True
    return False


@dataclass
class ChunkResult:
    page_number: int
    content: str
    chunk_index: int
    quality_score: float
    is_low_signal: bool
    content_hash: str


def chunk_pages(
    page_texts: list[tuple[int, str]],
    chunk_size: int = 1400,
    overlap_paragraphs: int = 1,
    min_chars: int = 25,
    max_chunks: int = 300,
    stats: dict | None = None,
) -> list[ChunkResult]:
    """
    Paragraph-based chunking. Document-agnostic.
    Uses blank-line paragraphs; fallback line-based grouping for single-block PDFs.
    Marks is_low_signal including repeated-content (hash appears >= 2x).
    Returns ChunkResult with quality_score, is_low_signal, content_hash.
    """
    results: list[ChunkResult] = []
    chunk_idx = 0
    total_paras = 0

    for page_num, text in page_texts:
        text = normalize_text(text)
        if not text:
            continue
        paras = _paragraphs(text)
        # Split oversized paragraphs (blank-line path can still produce huge blocks)
        expanded: list[str] = []
        for p in paras:
            expanded.extend(_split_oversized_paragraph(p, chunk_size))
        paras = expanded
        total_paras += len(paras)
        logger.debug(
            "chunk_pages page=%s text_len=%s paragraphs=%s",
            page_num,
            len(text),
            len(paras),
        )
        if not paras:
            continue
        raw_chunks = _chunk_by_paragraphs(
            page_num, paras, chunk_size, overlap_paragraphs, min_chars
        )
        for _, content in raw_chunks:
            if chunk_idx >= max_chunks:
                break
            metrics = _compute_quality_metrics(content)
            qs = _quality_score(metrics)
            low = _is_low_signal(metrics)
            chash = _content_hash(content)
            results.append(ChunkResult(
                page_number=page_num,
                content=content,
                chunk_index=chunk_idx,
                quality_score=round(qs, 4),
                is_low_signal=low,
                content_hash=chash,
            ))
            chunk_idx += 1

    # Repeated-content: only mark SHORT duplicates (headers/footers) as low-signal.
    hash_counts = Counter(r.content_hash for r in results)
    for r in results:
        if len(r.content) <= 200 and hash_counts[r.content_hash] >= 3:
            r.is_low_signal = True

    low_count = sum(1 for r in results if r.is_low_signal)
    if stats is not None:
        stats["total_paragraphs"] = total_paras
        stats["chunks_produced"] = len(results)
        stats["low_signal"] = low_count
    logger.info(
        "chunk_pages done: pages=%s total_paragraphs=%s chunks_produced=%s low_signal=%s",
        len(page_texts),
        total_paras,
        len(results),
        low_count,
    )
    return results
