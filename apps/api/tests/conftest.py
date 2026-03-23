"""Pytest fixtures."""

import asyncio
import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.auth import get_current_user
from app.main import app
from app.models import User
from app.core.rate_limit import clear_store


@pytest.fixture(scope="session")
def event_loop():
    """Use a single event loop for all async tests to avoid SQLAlchemy 'attached to different loop' errors."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def force_auth():
    """Override auth dependency with a concrete current user for a test."""

    async def _force(
        *,
        user_id: uuid.UUID | None = None,
        email: str | None = None,
    ) -> User:
        resolved_id = user_id or uuid.uuid4()
        resolved_email = email or f"{resolved_id}@test.local"

        current_user = User(id=resolved_id, email=resolved_email)

        async def _override() -> User:
            return current_user

        app.dependency_overrides[get_current_user] = _override
        return current_user

    yield _force
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Clear rate limit store before each test."""
    clear_store()
    yield
    clear_store()


@pytest.fixture
def demo_key_off(monkeypatch):
    """Disable demo-key auth requirement for tests that don't send the header."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "demo_key", None)


@pytest.fixture
def seed_document_bundle():
    """Create a user, document, sources, and chunks for retrieval-style tests."""

    async def _seed(
        *,
        user_email: str = "seed-user@t.local",
        filename: str = "jd.pdf",
        status: str = "ready",
        doc_domain: str | None = "job_description",
        sources: list[dict] | None = None,
        chunks: list[dict] | None = None,
        document_kwargs: dict | None = None,
    ) -> dict:
        from app.db.base import async_session_maker
        from app.models import Document, DocumentChunk, InterviewSource, User

        user_id = uuid.uuid4()
        async with async_session_maker() as db:
            user = User(id=user_id, email=user_email)
            db.add(user)
            await db.commit()

        source_specs = sources or [{
            "key": "jd",
            "source_type": "jd",
            "title": filename,
            "original_file_name": filename,
        }]

        async with async_session_maker() as db:
            doc = Document(
                user_id=user_id,
                filename=filename,
                s3_key="x",
                status=status,
                doc_domain=doc_domain,
                **(document_kwargs or {}),
            )
            db.add(doc)
            await db.flush()

            source_ids: dict[str, uuid.UUID] = {}
            source_objects: dict[str, InterviewSource] = {}
            for idx, spec in enumerate(source_specs):
                source_key = spec.get("key") or f"source-{idx}"
                source = InterviewSource(
                    document_id=doc.id,
                    source_type=spec.get("source_type", "jd"),
                    title=spec.get("title", filename),
                    original_file_name=spec.get("original_file_name"),
                    url=spec.get("url"),
                )
                db.add(source)
                await db.flush()
                source_ids[source_key] = source.id
                source_objects[source_key] = source

            chunk_ids: list[uuid.UUID] = []
            for idx, spec in enumerate(chunks or []):
                source_key = spec.get("source_key", "jd")
                source = source_objects[source_key]
                chunk = DocumentChunk(
                    document_id=doc.id,
                    source_id=source.id,
                    chunk_index=spec.get("chunk_index", idx),
                    content=spec["content"],
                    page_number=spec.get("page_number", 1),
                    section=spec.get("section"),
                    is_boilerplate=spec.get("is_boilerplate", False),
                    quality_score=spec.get("quality_score"),
                    is_low_signal=spec.get("is_low_signal", False),
                    content_hash=spec.get("content_hash"),
                    section_type=spec.get("section_type"),
                    skills_detected=spec.get("skills_detected"),
                    doc_domain=spec.get("doc_domain", doc_domain),
                    embedding=spec["embedding"],
                )
                db.add(chunk)
                await db.flush()
                chunk_ids.append(chunk.id)

            await db.commit()

        return {
            "user_id": user_id,
            "document_id": doc.id,
            "source_ids": source_ids,
            "chunk_ids": chunk_ids,
        }

    return _seed
