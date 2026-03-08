"""Tests for POST /retrieve endpoint."""

import uuid

import pytest

from app.core.config import settings


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
            "top_k": 6,  # > top_k_max (5), but <= Pydantic le=8 so we hit the handler
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
    """Retrieve returns metadata-rich chunks: text, score, sourceType, sourceTitle, page, chunkId."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

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
        source = InterviewSource(
            document_id=doc_id,
            source_type="jd",
            title="x.pdf",
            original_file_name="x.pdf",
        )
        db.add(source)
        await db.flush()
        chunk = DocumentChunk(
            document_id=doc_id,
            source_id=source.id,
            chunk_index=0,
            content="Machine learning skills include Python and TensorFlow.",
            page_number=1,
            section_type="other",
            doc_domain="general",
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
    assert c["chunkId"] == str(chunk_id)
    assert c["page"] == 1
    assert "Machine learning" in c["text"]
    assert c["sourceType"] == "jd"
    assert c["sourceTitle"] == "x.pdf"
    assert c["score"] == pytest.approx(1.0, abs=1e-4)
    assert c["is_low_signal"] is False


@pytest.mark.asyncio
async def test_retrieve_returns_section_type_for_jd_doc(client, demo_key_off, monkeypatch):
    """Retrieve returns section_type in chunks when doc has doc_domain=job_description."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

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
            doc_domain="job_description",
            jd_extraction_json={"role_title": "AI Engineer", "company": "Acme"},
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        source = InterviewSource(
            document_id=doc_id,
            source_type="jd",
            title="jd.pdf",
            original_file_name="jd.pdf",
        )
        db.add(source)
        await db.flush()
        chunk = DocumentChunk(
            document_id=doc_id,
            source_id=source.id,
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


@pytest.mark.asyncio
async def test_retrieve_section_types_filter(client, demo_key_off, monkeypatch):
    """Retrieve respects optional section_types filter."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

    dim = 1536
    mock_vec = [0.1] * dim

    def _mock_embed(q: str):
        return mock_vec

    monkeypatch.setattr("app.services.retrieval.embed_query", _mock_embed)
    monkeypatch.setattr("app.routers.retrieve.embed_query", _mock_embed)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="filter-test@t.local")
        db.add(user)
        await db.commit()
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="x.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        source = InterviewSource(
            document_id=doc_id,
            source_type="jd",
            title="x.pdf",
            original_file_name="x.pdf",
        )
        db.add(source)
        await db.flush()
        for idx, st in enumerate(["qualifications", "responsibilities"]):
            chunk = DocumentChunk(
                document_id=doc_id,
                source_id=source.id,
                chunk_index=idx,
                content=f"Content for {st} section.",
                page_number=1,
                section_type=st,
                doc_domain="job_description",
                embedding=mock_vec,
            )
            db.add(chunk)
        await db.flush()
        await db.commit()

    # Filter by qualifications only - should exclude responsibilities chunk
    resp = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "skills",
            "top_k": 5,
            "section_types": ["qualifications"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    for c in data["chunks"]:
        assert c["section_type"] == "qualifications"


@pytest.mark.asyncio
async def test_retrieve_source_types_filter(client, demo_key_off, monkeypatch):
    """Retrieve respects optional source_types filter; returns only chunks from matching sources."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

    dim = 1536
    mock_vec = [0.1] * dim

    def _mock_embed(q: str):
        return mock_vec

    monkeypatch.setattr("app.services.retrieval.embed_query", _mock_embed)
    monkeypatch.setattr("app.routers.retrieve.embed_query", _mock_embed)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="source-filter@t.local")
        db.add(user)
        await db.commit()
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="kit.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        # JD source
        src_jd = InterviewSource(
            document_id=doc_id,
            source_type="jd",
            title="Job Description",
            original_file_name="jd.pdf",
        )
        db.add(src_jd)
        await db.flush()
        # Notes source
        src_notes = InterviewSource(
            document_id=doc_id,
            source_type="notes",
            title="Interview Notes",
            original_file_name=None,
        )
        db.add(src_notes)
        await db.flush()
        chunk_jd = DocumentChunk(
            document_id=doc_id,
            source_id=src_jd.id,
            chunk_index=0,
            content="Python and AWS required.",
            page_number=1,
            section_type="qualifications",
            doc_domain="job_description",
            embedding=mock_vec,
        )
        chunk_notes = DocumentChunk(
            document_id=doc_id,
            source_id=src_notes.id,
            chunk_index=0,
            content="Candidate mentioned Kubernetes experience.",
            page_number=1,
            section_type="other",
            doc_domain="general",
            embedding=mock_vec,
        )
        db.add(chunk_jd)
        db.add(chunk_notes)
        await db.commit()

    # Filter by jd only - should exclude notes chunk
    resp = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "skills",
            "top_k": 5,
            "source_types": ["jd"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["chunks"]) >= 1
    for c in data["chunks"]:
        assert c["sourceType"] == "jd"
        assert c["sourceTitle"] == "Job Description"

    # No filter - returns chunks from all sources
    resp2 = await client.post(
        "/retrieve",
        json={
            "user_id": str(user_id),
            "document_id": str(doc_id),
            "query": "skills",
            "top_k": 5,
        },
    )
    assert resp2.status_code == 200
    types_seen = {c["sourceType"] for c in resp2.json()["chunks"]}
    assert "jd" in types_seen or "notes" in types_seen


@pytest.mark.asyncio
async def test_retrieve_chunks_keyword_returns_ranked_chunks(demo_key_off, seed_document_bundle):
    """Keyword retrieval returns chunk metadata in the shared retrieval shape."""
    from app.services.retrieval import retrieve_chunks_keyword

    dim = 1536
    mock_vec = [0.1] * dim

    seeded = await seed_document_bundle(
        user_email="keyword-retrieve@t.local",
        filename="jd.pdf",
        doc_domain="job_description",
        sources=[{
            "key": "jd",
            "source_type": "jd",
            "title": "Job Description",
            "original_file_name": "jd.pdf",
        }],
        chunks=[
            {
                "source_key": "jd",
                "content": "Python and AWS are required for this backend platform role.",
                "page_number": 1,
                "section_type": "qualifications",
                "doc_domain": "job_description",
                "embedding": mock_vec,
            },
            {
                "source_key": "jd",
                "content": "Benefits include health insurance and PTO.",
                "page_number": 2,
                "section_type": "compensation",
                "doc_domain": "job_description",
                "embedding": mock_vec,
            },
        ],
    )
    doc_id = seeded["document_id"]

    from app.db.base import async_session_maker
    async with async_session_maker() as db:
        chunks = await retrieve_chunks_keyword(
            db=db,
            document_id=doc_id,
            query_text="python aws backend",
            top_k=3,
            include_low_signal=False,
            section_types=None,
            doc_domain="job_description",
            source_types=["jd"],
        )

    assert len(chunks) == 1
    c = chunks[0]
    assert c["page"] == 1
    assert c["sourceType"] == "jd"
    assert c["sourceTitle"] == "Job Description"
    assert c["section_type"] == "qualifications"
    assert "Python and AWS" in c["text"]
    assert c["score"] > 0


@pytest.mark.asyncio
async def test_retrieve_chunks_keyword_respects_filters(demo_key_off, seed_document_bundle):
    """Keyword retrieval reuses source and section filters from semantic retrieval."""
    from app.services.retrieval import retrieve_chunks_keyword

    dim = 1536
    mock_vec = [0.1] * dim

    seeded = await seed_document_bundle(
        user_email="keyword-filters@t.local",
        filename="kit.pdf",
        doc_domain="job_description",
        sources=[
            {
                "key": "jd",
                "source_type": "jd",
                "title": "Job Description",
                "original_file_name": "jd.pdf",
            },
            {
                "key": "notes",
                "source_type": "notes",
                "title": "Notes",
                "original_file_name": None,
            },
        ],
        chunks=[
            {
                "source_key": "jd",
                "content": "Python experience is required for this role.",
                "page_number": 1,
                "section_type": "qualifications",
                "doc_domain": "job_description",
                "embedding": mock_vec,
            },
            {
                "source_key": "notes",
                "content": "Python experience came up in recruiter notes.",
                "page_number": 1,
                "section_type": "other",
                "doc_domain": "general",
                "embedding": mock_vec,
            },
        ],
    )
    doc_id = seeded["document_id"]

    from app.db.base import async_session_maker
    async with async_session_maker() as db:
        chunks = await retrieve_chunks_keyword(
            db=db,
            document_id=doc_id,
            query_text="python experience",
            top_k=5,
            source_types=["jd"],
            section_types=["qualifications"],
            doc_domain="job_description",
        )

    assert len(chunks) == 1
    assert chunks[0]["sourceType"] == "jd"
    assert chunks[0]["section_type"] == "qualifications"


def test_normalize_keyword_query_text_preserves_technical_tokens():
    """Keyword preprocessing keeps high-value technical terms retrieval-friendly."""
    from app.services.retrieval import _normalize_keyword_query_text

    normalized = _normalize_keyword_query_text(" Need C++, C#, .NET, Node.js, Next.js, React.js, PostgreSQL, pgvector, AWS!! ")

    assert '("c++" OR cpp)' in normalized
    assert '("c#" OR csharp)' in normalized
    assert '(".net" OR dotnet)' in normalized
    assert '("node.js" OR nodejs)' in normalized
    assert '("next.js" OR nextjs)' in normalized
    assert '("react.js" OR reactjs)' in normalized
    assert "(postgresql OR postgres)" in normalized
    assert '(pgvector OR "pg vector")' in normalized
    assert '(aws OR "amazon web services")' in normalized


def test_normalize_keyword_query_text_expands_jd_variants():
    """Keyword preprocessing adds simple format variants common in job descriptions."""
    from app.services.retrieval import _normalize_keyword_query_text

    normalized = _normalize_keyword_query_text("frontend backend full stack engineer")

    assert '(frontend OR "front-end")' in normalized
    assert '(backend OR "back-end")' in normalized
    assert '("full stack" OR "full-stack")' in normalized


def test_normalize_keyword_query_text_cleans_noise_without_aggressive_rewrites():
    """Keyword preprocessing removes punctuation noise but keeps the query readable."""
    from app.services.retrieval import _normalize_keyword_query_text

    normalized = _normalize_keyword_query_text(" senior backend?? role;;;   with pgvector + PostgreSQL ")

    assert "?" not in normalized
    assert ";" not in normalized
    assert '(backend OR "back-end")' in normalized
    assert '(pgvector OR "pg vector")' in normalized
    assert "(postgresql OR postgres)" in normalized
