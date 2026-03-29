"""Competency Map extraction: RAG-based extraction of 8–12 competencies from JD chunks."""

import json
import logging
import re
import uuid

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Document
from app.services.retrieval import embed_query, retrieve_chunks

logger = logging.getLogger(__name__)

COMPETENCY_QUERIES = [
    "responsibilities",
    "requirements",
    "qualifications",
    "skills",
    "preferred",
]

CHUNKS_PER_QUERY = 4
COMPETENCIES_MIN = 8
COMPETENCIES_MAX = 12


async def extract_competencies(db: AsyncSession, document_id: uuid.UUID) -> list[dict] | None:
    """
    Extract 8–12 competencies from JD using RAG (retrieved chunks only, no raw JD).
    Runs hybrid retrieval for responsibilities, requirements, qualifications, skills, preferred.
    Returns list of { id, label, description?, evidence: [{ chunkId, page?, sourceTitle }] }.
    Returns None if extraction fails or no chunks found.
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return None

    if doc.doc_domain != "job_description":
        logger.info("Competency extraction skipped: doc_domain=%s (not job_description)", doc.doc_domain)
        return None

    # Retrieve top chunks per query (RAG: no full JD)
    seen: dict[str, dict] = {}
    for query in COMPETENCY_QUERIES:
        try:
            q_embedding = embed_query(query)
            chunks = await retrieve_chunks(
                db=db,
                document_id=document_id,
                query_embedding=q_embedding,
                query_text=query,
                top_k=CHUNKS_PER_QUERY,
                include_low_signal=False,
                section_types=None,
                doc_domain="job_description",
            )
            for c in chunks:
                cid = c.get("chunkId") or c.get("chunk_id")
                if cid and cid not in seen:
                    seen[cid] = {
                        "chunkId": cid,
                        "page": c.get("page") or c.get("page_number"),
                        "sourceTitle": c.get("sourceTitle", ""),
                        "text": c.get("text") or c.get("snippet", ""),
                        "retrieval_source": c.get("retrieval_source"),
                        "semantic_score": c.get("semantic_score"),
                        "keyword_score": c.get("keyword_score"),
                        "final_score": c.get("final_score"),
                    }
        except Exception as e:
            logger.warning("Competency retrieval for query=%s failed: %s", query, e)

    if not seen:
        logger.warning("Competency extraction: no chunks retrieved for document_id=%s", document_id)
        return None

    chunks_list = list(seen.values())

    if not settings.openai_api_key:
        logger.warning("Competency extraction: OpenAI not configured")
        return None

    try:
        excerpt_lines = []
        for i, c in enumerate(chunks_list):
            ev = f"[{i}] chunkId={c['chunkId']} page={c.get('page') or '?'} sourceTitle={c.get('sourceTitle') or ''}"
            excerpt_lines.append(f"{ev}\n{c.get('text', '')[:800]}")

        excerpts_text = "\n\n---\n\n".join(excerpt_lines)

        system_prompt = """You extract competencies from job description excerpts ONLY. You must NOT use any knowledge outside the provided excerpts.

Output valid JSON only, no markdown:
{"competencies": [{"id": "uuid-like-id", "label": "short competency name", "description": "optional 1-2 sentence explanation", "evidence_indices": [0,1,...]}]}

Rules:
- Generate 8–12 distinct competencies.
- Each competency maps to skills, requirements, qualifications, responsibilities, or preferred traits mentioned in the excerpts.
- evidence_indices: 0-based indices into the excerpt list above. Each competency must cite at least 1 excerpt it is grounded in.
- id: short slug (e.g. "python-expertise", "leadership", "data-analysis").
- label: 2–5 word label.
- description: optional, 1–2 sentences. Can be null."""

        user_content = f"""Job description excerpts (indexed for citation):

{excerpts_text}

Extract 8–12 competencies. Output JSON only."""

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.model_fast,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=min(2000, settings.max_completion_tokens * 4),
        )

        raw = (response.choices[0].message.content or "").strip()
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            raw = json_match.group(0)

        data = json.loads(raw)
        raw_competencies = data.get("competencies", [])
        if not isinstance(raw_competencies, list):
            return None

        result_competencies = []
        for i, rc in enumerate(raw_competencies[:COMPETENCIES_MAX]):
            idx_list = rc.get("evidence_indices") or []
            if not isinstance(idx_list, list):
                idx_list = []
            evidence = []
            for idx in idx_list:
                if isinstance(idx, int) and 0 <= idx < len(chunks_list):
                    c = chunks_list[idx]
                    evidence.append({
                        "chunkId": c.get("chunkId", ""),
                        "page": c.get("page"),
                        "sourceTitle": c.get("sourceTitle", ""),
                        "retrieval_source": c.get("retrieval_source"),
                        "semantic_score": c.get("semantic_score"),
                        "keyword_score": c.get("keyword_score"),
                        "final_score": c.get("final_score"),
                    })
            if not evidence and chunks_list:
                evidence.append({
                    "chunkId": chunks_list[0].get("chunkId", ""),
                    "page": chunks_list[0].get("page"),
                    "sourceTitle": chunks_list[0].get("sourceTitle", ""),
                    "retrieval_source": chunks_list[0].get("retrieval_source"),
                    "semantic_score": chunks_list[0].get("semantic_score"),
                    "keyword_score": chunks_list[0].get("keyword_score"),
                    "final_score": chunks_list[0].get("final_score"),
                })

            comp_id = str(rc.get("id", "")) or f"comp-{i}"
            if not re.match(r"^[a-zA-Z0-9_-]+$", comp_id):
                comp_id = f"comp-{i}"

            result_competencies.append({
                "id": comp_id,
                "label": str(rc.get("label", "")).strip() or f"Competency {i+1}",
                "description": str(rc.get("description", "")).strip() or None,
                "evidence": evidence,
            })

        if len(result_competencies) < COMPETENCIES_MIN:
            logger.info(
                "Competency extraction produced %s competencies (min %s)",
                len(result_competencies),
                COMPETENCIES_MIN,
            )

        return result_competencies

    except Exception as e:
        logger.exception("Competency extraction failed: %s", e)
        return None
