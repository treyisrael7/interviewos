"""Tests for POST /retrieve endpoint."""

import uuid

import pytest

from app.core.config import settings


@pytest.fixture
def demo_key_off(monkeypatch):
    monkeypatch.setattr(settings, "demo_key", None)


@pytest.mark.asyncio
async def test_retrieve_requires_valid_input(client, demo_key_off):
    """Retrieve returns 422 for missing or invalid body."""
    resp = await client.post("/retrieve", json={})
    assert resp.status_code == 422

    resp = await client.post(
        "/retrieve",
        json={
            "user_id": "11111111-1111-1111-1111-111111111111",
            "document_id": "11111111-1111-1111-1111-111111111111",
            "query": "",  # min_length=1
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_retrieve_document_not_found(client, demo_key_off):
    """Retrieve returns 404 for unknown document."""
    resp = await client.post(
        "/retrieve",
        json={
            "user_id": "11111111-1111-1111-1111-111111111111",
            "document_id": "11111111-1111-1111-1111-111111111111",
            "query": "test query",
            "top_k": 3,
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_retrieve_rejects_top_k_exceeds_max(client, demo_key_off, monkeypatch):
    """Retrieve returns 400 when top_k > TOP_K_MAX."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    monkeypatch.setattr(settings, "top_k_max", 5)

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="retrieve-test@t.local")
        db.add(user)
        await db.commit()
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="x.pdf",
            s3_key="x",
            status="ready",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "test",
            "top_k": 10,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "top_k exceeds limit"
    assert resp.json()["detail"]["max"] == 5


@pytest.mark.asyncio
async def test_retrieve_rejects_document_not_ready(client, demo_key_off):
    """Retrieve returns 400 when document status is not ready."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="retrieve-test2@t.local")
        db.add(user)
        await db.commit()
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="x.pdf",
            s3_key="x",
            status="uploaded",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "test",
            "top_k": 3,
        },
    )
    assert resp.status_code == 400
    assert "ready" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_retrieve_success_returns_chunks(client, demo_key_off, monkeypatch):
    """Retrieve returns chunks with chunk_id, page_number, snippet, score when document has chunks."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, User

    # Deterministic embedding: same vector for query and chunk -> cosine sim 1.0
    dim = 1536
    mock_vec = [0.1] * dim

    def _mock_embed(q: str):
        return mock_vec

    monkeypatch.setattr("app.services.retrieval.embed_query", _mock_embed)
    monkeypatch.setattr("app.routers.retrieve.embed_query", _mock_embed)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="retrieve-success@t.local")
        db.add(user)
        await db.commit()
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="x.pdf",
            s3_key="x",
            status="ready",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            content="Machine learning skills include Python and TensorFlow.",
            page_number=1,
            embedding=mock_vec,
        )
        db.add(chunk)
        await db.flush()
        chunk_id = chunk.id
        await db.commit()

    resp = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "machine learning",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "chunks" in data
    assert len(data["chunks"]) >= 1
    c = data["chunks"][0]
    assert c["chunk_id"] == str(chunk_id)
    assert c["page_number"] == 1
    assert "Machine learning" in c["snippet"]
    assert c["score"] == pytest.approx(1.0, abs=1e-4)
    assert c["is_low_signal"] is False


@pytest.mark.asyncio
async def test_retrieve_returns_section_type_for_jd_doc(client, demo_key_off, monkeypatch):
    """Retrieve returns section_type in chunks when doc has jd_extraction_json."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, User

    dim = 1536
    mock_vec = [0.1] * dim

    def _mock_embed(q: str):
        return mock_vec

    monkeypatch.setattr("app.services.retrieval.embed_query", _mock_embed)
    monkeypatch.setattr("app.routers.retrieve.embed_query", _mock_embed)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="jd-retrieve@t.local")
        db.add(user)
        await db.commit()
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            jd_extraction_json={"role_title": "AI Engineer", "company": "Acme"},
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_index=0,
            content="Python, TensorFlow, AWS required. 5+ years experience.",
            page_number=1,
            section_type="qualifications",
            doc_domain="job_description",
            embedding=mock_vec,
        )
        db.add(chunk)
        await db.flush()
        chunk_id = chunk.id
        await db.commit()

    resp = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "what skills are required?",
            "top_k": 3,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["chunks"]) >= 1
    c = data["chunks"][0]
    assert c["section_type"] == "qualifications"
