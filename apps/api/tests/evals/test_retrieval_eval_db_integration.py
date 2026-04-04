"""Integration tests: built-in retrieval eval dataset against a real Postgres corpus.

Runs the same retrieval stack as production (semantic / hybrid / keyword) with a
deterministic query embedding so CI does not require OPENAI_API_KEY.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy import delete, select

from app.db.base import async_session_maker
from app.models import Document, DocumentChunk, InterviewSource, User
from app.models.document_chunk import EMBEDDING_DIM
from evals.retrieval.ci_constants import (
    CI_SHARED_EMBEDDING_VALUE,
    PLATFORM_ENGINEER_JD_DOCUMENT_ID,
    PLATFORM_ENGINEER_JD_FIXTURE_REF,
    PLATFORM_ENGINEER_JD_USER_ID,
)
from evals.retrieval.loader import load_builtin_dataset
from evals.retrieval.runner import run_eval_dataset_for_modes


def _shared_embedding() -> list[float]:
    return [CI_SHARED_EMBEDDING_VALUE] * EMBEDDING_DIM


# Chunk texts are aligned with evals/retrieval/cases/job_description_starter.json expectations.
_CI_CHUNKS: list[dict] = [
    {
        "chunk_index": 0,
        "page_number": 1,
        "section_type": "compensation",
        "content": (
            "The salary range for this senior role is $160,000 to $190,000 annually. "
            "We offer a competitive salary range and equity."
        ),
    },
    {
        "chunk_index": 1,
        "page_number": 1,
        "section_type": "qualifications",
        "content": (
            "Qualifications and required skills: Python and PostgreSQL expertise with 5+ years building "
            "distributed systems for production backends."
        ),
    },
    {
        "chunk_index": 2,
        "page_number": 2,
        "section_type": "tools",
        "content": (
            "Skills and qualifications for tooling are required: tools and technologies include AWS, "
            "Kubernetes, Terraform, and pgvector. The team uses Python services across the platform."
        ),
    },
    {
        "chunk_index": 3,
        "page_number": 2,
        "section_type": "responsibilities",
        "content": (
            "Main responsibilities: you will build and operate backend services and collaborate with "
            "product, data, and infrastructure partners in this role."
        ),
    },
]


async def _teardown_ci_retrieval_fixture() -> None:
    async with async_session_maker() as db:
        await db.execute(delete(Document).where(Document.id == PLATFORM_ENGINEER_JD_DOCUMENT_ID))
        await db.execute(delete(User).where(User.id == PLATFORM_ENGINEER_JD_USER_ID))
        await db.commit()


async def _seed_ci_fixture_document() -> None:
    emb = _shared_embedding()
    async with async_session_maker() as db:
        await db.execute(delete(Document).where(Document.id == PLATFORM_ENGINEER_JD_DOCUMENT_ID))
        result = await db.execute(select(User).where(User.id == PLATFORM_ENGINEER_JD_USER_ID))
        if result.scalar_one_or_none() is None:
            db.add(
                User(
                    id=PLATFORM_ENGINEER_JD_USER_ID,
                    email="retrieval-eval-ci@fixture.local",
                    clerk_id=None,
                )
            )
        await db.commit()

    async with async_session_maker() as db:
        doc = Document(
            id=PLATFORM_ENGINEER_JD_DOCUMENT_ID,
            user_id=PLATFORM_ENGINEER_JD_USER_ID,
            filename="platform_engineer_ci.pdf",
            s3_key="retrieval-eval/ci/platform_engineer_ci.pdf",
            status="ready",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()

        source = InterviewSource(
            document_id=doc.id,
            source_type="jd",
            title="Platform Engineer JD (CI)",
            original_file_name="platform_engineer_ci.pdf",
        )
        db.add(source)
        await db.flush()

        for spec in _CI_CHUNKS:
            db.add(
                DocumentChunk(
                    document_id=doc.id,
                    source_id=source.id,
                    chunk_index=spec["chunk_index"],
                    content=spec["content"],
                    page_number=spec["page_number"],
                    section_type=spec["section_type"],
                    doc_domain="job_description",
                    embedding=emb,
                )
            )
        await db.commit()


@pytest.mark.asyncio
async def test_job_description_starter_passes_all_modes(monkeypatch):
    """Built-in dataset should pass semantic, hybrid, and keyword against the seeded corpus."""
    await _seed_ci_fixture_document()
    try:
        monkeypatch.setattr("evals.retrieval.runner.embed_query", lambda _query: _shared_embedding())

        async def _resolve_fixture(name: str) -> UUID:
            assert name == PLATFORM_ENGINEER_JD_FIXTURE_REF
            return PLATFORM_ENGINEER_JD_DOCUMENT_ID

        dataset = load_builtin_dataset("job_description_starter")
        async with async_session_maker() as db:
            runs = await run_eval_dataset_for_modes(
                db=db,
                dataset=dataset,
                modes=("semantic", "hybrid", "keyword"),
                fixture_resolver=_resolve_fixture,
            )

        assert [r.mode for r in runs] == ["semantic", "hybrid", "keyword"]
        for run in runs:
            assert run.failed_cases == 0, (
                f"mode={run.mode} failures={run.failed_cases} "
                f"details={[r for r in run.results if not r.passed]}"
            )
            assert run.passed_cases == run.total_cases == len(dataset.cases)

        async with async_session_maker() as db:
            row = await db.execute(select(Document).where(Document.id == PLATFORM_ENGINEER_JD_DOCUMENT_ID))
            assert row.scalar_one_or_none() is not None
    finally:
        await _teardown_ci_retrieval_fixture()
