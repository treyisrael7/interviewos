"""Tests for rate limit middleware."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_ask_rate_limit(client, monkeypatch, force_auth):
    """POST /ask is rate limited to 10 per hour."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    monkeypatch.setattr(settings, "demo_key", None)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="ratelimit@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="ratelimit@t.local")
    async with async_session_maker() as db:
        doc = Document(user_id=user_id, filename="x.pdf", s3_key="x", status="ready")
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    body = {"document_id": str(doc_id), "question": "test?"}
    with patch("app.routers.ask.retrieve_chunks", new_callable=AsyncMock, return_value=[]):
        with patch("app.routers.ask.embed_query", return_value=[0.1] * 1536):
            for i in range(10):
                resp = await client.post("/ask", json=body)
                assert resp.status_code == 200, f"Request {i+1} should succeed"

            resp = await client.post("/ask", json=body)
    data = resp.json()
    assert data["detail"] == "Rate limit exceeded"
    assert "retry_after_seconds" in data
    assert data["limit"] == 10
    assert data["window"] == "hour"
    assert "Retry-After" in resp.headers


@pytest.mark.asyncio
async def test_ingest_rate_limit(client, monkeypatch, force_auth):
    """POST /documents/{id}/ingest is rate limited to 3 per day."""
    monkeypatch.setattr(settings, "demo_key", None)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "s3_bucket", None)

    user_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    await force_auth(user_id=user_id, email="ingest-limit@t.local")
    presign_body = {
        "filename": "test.pdf",
        "content_type": "application/pdf",
        "file_size_bytes": 1024,
    }
    # Minimal valid PDF bytes for upload
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    doc_ids = []
    for _ in range(4):
        pr = await client.post("/documents/presign", json=presign_body)
        assert pr.status_code == 200
        data = pr.json()
        doc_id = data["document_id"]
        doc_ids.append(doc_id)
        key = data["s3_key"]
        await client.put(f"/documents/upload-local?key={key}", content=pdf_bytes)
        await client.post(
            "/documents/confirm",
            json={"document_id": doc_id, "s3_key": key},
        )

    for i in range(3):
        resp = await client.post(
            f"/documents/{doc_ids[i]}/ingest",
            json={},
        )
        assert resp.status_code == 200, f"Request {i+1}"

    resp = await client.post(
        f"/documents/{doc_ids[3]}/ingest",
        json={},
    )
    assert resp.status_code == 429
    assert resp.json()["limit"] == 3
    assert resp.json()["window"] == "day"


@pytest.mark.asyncio
async def test_presign_rate_limit(client, monkeypatch, force_auth):
    """POST /documents/presign is rate limited to 10 per day."""
    monkeypatch.setattr(settings, "demo_key", None)
    monkeypatch.setattr(settings, "s3_bucket", None)  # Use LocalStorage
    await force_auth()
    presign_body = {
        "filename": "test.pdf",
        "content_type": "application/pdf",
        "file_size_bytes": 1024,
    }
    for _ in range(10):
        resp = await client.post("/documents/presign", json=presign_body)
        assert resp.status_code == 200

    resp = await client.post("/documents/presign", json=presign_body)
    assert resp.status_code == 429
    assert resp.json()["limit"] == 10
    assert resp.json()["window"] == "day"


@pytest.mark.asyncio
async def test_non_rate_limited_paths(client, monkeypatch):
    """GET / and /health are not rate limited."""
    monkeypatch.setattr(settings, "demo_key", None)
    for _ in range(15):
        resp = await client.get("/")
        assert resp.status_code == 200
        resp = await client.get("/health")
        assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_client_supplied_user_header_does_not_bypass_limit(client, monkeypatch, force_auth):
    """Changing x-user-id does not create a new rate-limit bucket."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    monkeypatch.setattr(settings, "demo_key", None)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="limit-users@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="limit-users@t.local")
    async with async_session_maker() as db:
        doc = Document(user_id=user_id, filename="x.pdf", s3_key="x", status="ready")
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    body = {"document_id": str(doc_id), "question": "test?"}
    with patch("app.routers.ask.retrieve_chunks", new_callable=AsyncMock, return_value=[]):
        with patch("app.routers.ask.embed_query", return_value=[0.1] * 1536):
            for _ in range(10):
                resp = await client.post(
                    "/ask", json=body, headers={"x-user-id": "user-a"}
                )
                assert resp.status_code == 200

            # Changing the header should not bypass the shared IP-based limit.
            resp = await client.post("/ask", json=body, headers={"x-user-id": "user-a"})
            assert resp.status_code == 429

            resp = await client.post("/ask", json=body, headers={"x-user-id": "user-b"})
            assert resp.status_code == 429
