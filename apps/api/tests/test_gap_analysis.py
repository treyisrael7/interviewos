"""Tests for resume-to-JD gap analysis."""

import uuid

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_gap_analysis_requires_resume_source(client, demo_key_off, force_auth):
    """Gap analysis returns 400 when no attached or account-level resume exists."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewSource, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=user_id, email="gap-no-resume@t.local"))
        await db.commit()
    await force_auth(user_id=user_id, email="gap-no-resume@t.local")

    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
            jd_extraction_json={"required_skills": ["python"]},
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        db.add(InterviewSource(
            document_id=doc_id,
            source_type="jd",
            title="JD",
            original_file_name="jd.pdf",
        ))
        await db.commit()

    resp = await client.post(
        f"/documents/{doc_id}/gap-analysis",
        json={},
    )
    assert resp.status_code == 400
    assert "resume source" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_gap_analysis_uses_attached_resume_source(client, demo_key_off, monkeypatch, force_auth):
    """Gap analysis classifies attached resume evidence while preserving citations."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr("app.services.gap_analysis_retrieval.embed_query", lambda _: mock_vec)
    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", False)

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=user_id, email="gap-attached@t.local"))
        await db.commit()
    await force_auth(user_id=user_id, email="gap-attached@t.local")

    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
            jd_extraction_json={
                "required_skills": ["python", "aws", "kubernetes"],
                "tools": ["docker"],
            },
            role_profile={"focusAreas": ["backend engineering"]},
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id

        jd_source = InterviewSource(
            document_id=doc_id,
            source_type="jd",
            title="Job Description",
            original_file_name="jd.pdf",
        )
        resume_source = InterviewSource(
            document_id=doc_id,
            source_type="resume",
            title="Resume",
            original_file_name="resume.pdf",
            profile_json={
                "cachedFromChunks": True,
                "skills": [{"label": "python", "normalized": "python", "evidence": []}],
                "tools": [],
                "cloudPlatforms": [{"label": "aws", "normalized": "aws", "evidence": []}],
                "experienceClaims": [],
                "education": [],
                "certifications": [],
            },
        )
        db.add(jd_source)
        db.add(resume_source)
        await db.flush()

        db.add(DocumentChunk(
            document_id=doc_id,
            source_id=jd_source.id,
            chunk_index=0,
            content="Required skills include Python, AWS, Kubernetes, and Docker.",
            page_number=1,
            section_type="qualifications",
            doc_domain="job_description",
            embedding=mock_vec,
        ))
        db.add(DocumentChunk(
            document_id=doc_id,
            source_id=resume_source.id,
            chunk_index=0,
            content="Python developer with AWS experience building APIs and Docker-based services.",
            page_number=1,
            section_type="other",
            doc_domain="general",
            embedding=mock_vec,
        ))
        await db.commit()

    resp = await client.post(
        f"/documents/{doc_id}/gap-analysis",
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["resume_sources_considered"]
    matched_labels = {item["label"].lower() for item in data["matched_requirements"]}
    partial_labels = {item["label"].lower() for item in data["partial_requirements"]}
    gap_labels = {item["label"].lower() for item in data["gap_requirements"]}
    assert "python" in matched_labels
    assert "aws" in matched_labels
    assert "kubernetes" in gap_labels or "kubernetes" in partial_labels
    assert any(item["resume_evidence"] for item in data["matched_requirements"])
    assert any(citation["sourceType"] == "resume" for item in data["strengths_cited"] for citation in item["citations"])


@pytest.mark.asyncio
async def test_gap_analysis_uses_account_level_resume(client, demo_key_off, monkeypatch, force_auth):
    """Gap analysis falls back to the account-level resume document when no attached resume exists."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

    dim = 1536
    mock_vec = [0.1] * dim

    monkeypatch.setattr("app.services.gap_analysis_retrieval.embed_query", lambda _: mock_vec)
    monkeypatch.setattr(settings, "hybrid_retrieval_enabled", False)

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        db.add(User(id=user_id, email="gap-account@t.local"))
        await db.commit()
    await force_auth(user_id=user_id, email="gap-account@t.local")

    async with async_session_maker() as db:
        jd_doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
            jd_extraction_json={"required_skills": ["sql"]},
        )
        resume_doc = Document(
            user_id=user_id,
            filename="Resume.pdf",
            s3_key="resume",
            status="ready",
            doc_domain="user_resume",
        )
        db.add(jd_doc)
        db.add(resume_doc)
        await db.flush()

        jd_source = InterviewSource(
            document_id=jd_doc.id,
            source_type="jd",
            title="Job Description",
            original_file_name="jd.pdf",
        )
        resume_source = InterviewSource(
            document_id=resume_doc.id,
            source_type="resume",
            title="Resume",
            original_file_name="Resume.pdf",
            profile_json={"cachedFromChunks": True, "skills": [{"label": "sql", "normalized": "sql", "evidence": []}]},
        )
        db.add(jd_source)
        db.add(resume_source)
        await db.flush()

        db.add(DocumentChunk(
            document_id=jd_doc.id,
            source_id=jd_source.id,
            chunk_index=0,
            content="The role requires strong SQL and analytics skills.",
            page_number=1,
            section_type="qualifications",
            doc_domain="job_description",
            embedding=mock_vec,
        ))
        db.add(DocumentChunk(
            document_id=resume_doc.id,
            source_id=resume_source.id,
            chunk_index=0,
            content="Built analytics dashboards and wrote SQL for reporting pipelines.",
            page_number=1,
            section_type="other",
            doc_domain="general",
            embedding=mock_vec,
        ))
        await db.commit()

    resp = await client.post(
        f"/documents/{jd_doc.id}/gap-analysis",
        json={},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert any(item["label"].lower() == "sql" for item in data["matched_requirements"])
    assert any(source["documentId"] == str(resume_doc.id) for source in data["resume_sources_considered"])
