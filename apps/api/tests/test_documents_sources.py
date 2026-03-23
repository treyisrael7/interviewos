"""Tests for document source routes: add-text, presign-resume, ingest-resume, add-from-url, list-sources."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings
from app.services.storage import LocalStorage


@pytest.fixture
def clerk_jwks_off(monkeypatch):
    monkeypatch.setattr(settings, "clerk_jwks_url", None)


@pytest.fixture
def use_local_storage(monkeypatch, tmp_path):
    storage = LocalStorage(base_path=str(tmp_path))

    def _get_storage():
        return storage

    monkeypatch.setattr("app.services.storage.get_storage", _get_storage)
    monkeypatch.setattr("app.routers.documents.get_storage", _get_storage)
    return storage


@pytest.mark.asyncio
async def test_add_text_source_returns_401_without_auth(client, demo_key_off, clerk_jwks_off):
    """POST add-text returns 401 when no user_id and no Bearer."""
    doc_id = str(uuid.uuid4())
    resp = await client.post(
        f"/documents/{doc_id}/sources/add-text",
        json={
            "source_type": "resume",
            "title": "My Resume",
            "content": "Experienced developer with Python.",
        },
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_add_text_source_success(client, demo_key_off, clerk_jwks_off, use_local_storage, force_auth):
    """POST add-text ingests text and returns source_id."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    user_id = uuid.uuid4()
    doc_id = None
    async with async_session_maker() as db:
        user = User(id=user_id, email="addtext@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="addtext@t.local")
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    with patch("app.services.source_ingestion.ingest_text_source", new_callable=AsyncMock, return_value=str(uuid.uuid4())):
        resp = await client.post(
                f"/documents/{doc_id}/sources/add-text",
                json={
                    "source_type": "resume",
                    "title": "Resume",
                    "content": "Python developer with 5 years experience.",
                },
            )
        assert resp.status_code == 200
    data = resp.json()
    assert "source_id" in data
    assert data["status"] == "ingested"


@pytest.mark.asyncio
async def test_list_sources_returns_401_without_auth(client, demo_key_off, clerk_jwks_off):
    """GET list-sources returns 401 when no user_id and no Bearer."""
    doc_id = str(uuid.uuid4())
    resp = await client.get(f"/documents/{doc_id}/sources")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_sources_success(client, demo_key_off, clerk_jwks_off, force_auth):
    """GET list-sources returns sources for document."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewSource, User

    user_id = uuid.uuid4()
    doc_id = None
    async with async_session_maker() as db:
        user = User(id=user_id, email="listsrc@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="listsrc@t.local")
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        src = InterviewSource(
            document_id=doc_id,
            source_type="resume",
            title="Resume.pdf",
            original_file_name="resume.pdf",
        )
        db.add(src)
        await db.commit()

    resp = await client.get(f"/documents/{doc_id}/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["source_type"] == "resume"
    assert "Resume" in data[0]["title"] or "resume" in data[0]["title"].lower()


@pytest.mark.asyncio
async def test_presign_resume_returns_401_without_auth(client, demo_key_off, clerk_jwks_off, use_local_storage):
    """POST presign-resume returns 401 when no user_id and no Bearer."""
    doc_id = str(uuid.uuid4())
    resp = await client.post(
        f"/documents/{doc_id}/sources/presign-resume",
        json={"filename": "resume.pdf", "file_size_bytes": 1024},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_add_from_url_returns_401_without_auth(client, demo_key_off, clerk_jwks_off):
    """POST add-from-url returns 401 when no user_id and no Bearer."""
    doc_id = str(uuid.uuid4())
    resp = await client.post(
        f"/documents/{doc_id}/sources/add-from-url",
        json={"url": "https://example.com/about", "title": "Company"},
    )
    assert resp.status_code == 401
