"""Tests for POST /ask endpoint."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_ask_requires_valid_input(client, demo_key_off, monkeypatch, force_auth):
    """Ask returns 422 for missing or invalid body."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    await force_auth()

    resp = await client.post("/ask", json={})
    assert resp.status_code == 422

    resp = await client.post(
        "/ask",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "",  # min_length=1
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ask_document_not_found(client, demo_key_off, monkeypatch, force_auth):
    """Ask returns 404 for unknown document."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    await force_auth()

    resp = await client.post(
        "/ask",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question": "What is the salary range?",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ask_rejects_document_not_ready(client, demo_key_off, monkeypatch, force_auth):
    """Ask returns 400 when document status is not ready."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="ask-test@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="ask-test@t.local")
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="x.pdf",
            s3_key="x",
            status="pending",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(
        "/ask",
        json={
            "document_id": str(doc_id),
            "question": "What is the salary?",
        },
    )
    assert resp.status_code == 400
    assert "ready" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_ask_no_chunks_returns_fallback(client, demo_key_off, monkeypatch, force_auth):
    """Ask returns fallback answer when no relevant chunks found."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="ask-nochunks@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="ask-nochunks@t.local")
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

    # Mock retrieval to return empty chunks; generate_grounded_answer returns fallback
    with patch("app.routers.ask.retrieve_chunks", new_callable=AsyncMock, return_value=[]):
        with patch("app.routers.ask.embed_query", return_value=[0.1] * 1536):
            resp = await client.post(
                "/ask",
                json={
                    "document_id": str(doc_id),
                    "question": "What is the salary?",
                },
            )

    # With no chunks, generate_grounded_answer returns fallback without calling OpenAI
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "I don't have enough information" in data["answer"]
    assert data["citations"] == []
