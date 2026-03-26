"""Tests for POST /interview/generate endpoint."""

import uuid

import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_interview_generate_requires_valid_input(client, demo_key_off, force_auth):
    """Generate returns 422 for invalid body."""
    await force_auth()
    resp = await client.post("/interview/generate", json={})
    assert resp.status_code == 422

    resp = await client.post(
        "/interview/generate",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "difficulty": "invalid",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_interview_generate_document_not_found(client, demo_key_off, force_auth):
    """Generate returns 404 for unknown document."""
    await force_auth()
    resp = await client.post(
        "/interview/generate",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "difficulty": "junior",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_interview_generate_rejects_non_jd_document(client, demo_key_off, force_auth):
    """Generate returns 400 when doc_domain is not job_description."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="interview-nonjd@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="interview-nonjd@t.local")
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="general.pdf",
            s3_key="x",
            status="ready",
            doc_domain="general",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(
        "/interview/generate",
        json={
            "document_id": str(doc_id),
            "mode": "technical",
            "difficulty": "junior",
        },
    )
    assert resp.status_code == 400
    assert "job_description" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_interview_generate_rejects_document_not_ready(client, demo_key_off, force_auth):
    """Generate returns 400 when document status is not ready."""
    from app.db.base import async_session_maker
    from app.models import Document, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="interview-pending@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="interview-pending@t.local")
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="processing",
            doc_domain="job_description",
        )
        db.add(doc)
        await db.flush()
        doc_id = doc.id
        await db.commit()

    resp = await client.post(
        "/interview/generate",
        json={
            "document_id": str(doc_id),
            "difficulty": "junior",
        },
    )
    assert resp.status_code == 400
    assert "ready" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_interview_generate_no_evidence_returns_400(client, demo_key_off, monkeypatch, force_auth):
    """Generate returns 400 when no evidence chunks found for mode."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

    dim = 1536
    mock_vec = [0.1] * dim

    async def _mock_generate_empty(*args, **kwargs):
        return []

    monkeypatch.setattr("app.routers.interview.runtime.generate_questions", _mock_generate_empty)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="interview-noev@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="interview-noev@t.local")
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
            content="Minimal content.",
            page_number=1,
            section_type="other",
            doc_domain="job_description",
            embedding=mock_vec,
        )
        db.add(chunk)
        await db.commit()

    resp = await client.post(
        "/interview/generate",
        json={
            "document_id": str(doc_id),
            "difficulty": "junior",
        },
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "evidence" in str(detail).lower() or "section" in str(detail).lower() or "questions" in str(detail).lower()


@pytest.mark.asyncio
async def test_interview_generate_success_creates_session(client, demo_key_off, monkeypatch, force_auth):
    """Generate creates a session and questions."""
    from app.db.base import async_session_maker
    from app.models import Document, DocumentChunk, InterviewSource, User

    dim = 1536
    mock_vec = [0.1] * dim

    def _mock_embed(q: str):
        return mock_vec

    mock_evidence = [
        {"chunk_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "page_number": 1, "snippet": "Python, AWS required."},
        {"chunk_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "page_number": 2, "snippet": "ML pipelines, NLP."},
    ]

    mock_questions = [
        {
            "type": "role_specific",
            "competencyId": "python-expertise",
            "competencyLabel": "Python",
            "questionText": "Describe your experience with Python and AWS.",
            "whatGoodLooksLike": [
                "Concrete project examples",
                "Scalability considerations",
                "Collaboration experience",
            ],
            "mustMention": [],
            "evidence": mock_evidence[:1],
            "evidenceChunkIds": ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"],
        },
        {
            "type": "behavioral",
            "competencyId": "leadership",
            "competencyLabel": "leadership",
            "questionText": "Tell me about a time you led a technical initiative.",
            "whatGoodLooksLike": ["STAR format", "Measurable outcomes", "Learning takeaways"],
            "mustMention": [],
            "evidence": mock_evidence,
            "evidenceChunkIds": ["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
        },
    ]

    async def _mock_generate_questions(*args, **kwargs):
        return mock_questions[: kwargs.get("num_questions", 2)]

    monkeypatch.setattr("app.services.retrieval.embed_query", _mock_embed)
    monkeypatch.setattr("app.services.interview.embed_query", _mock_embed)
    monkeypatch.setattr("app.routers.interview.runtime.generate_questions", _mock_generate_questions)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="interview-gen@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="interview-gen@t.local")
    async with async_session_maker() as db:
        doc = Document(
            user_id=user_id,
            filename="jd.pdf",
            s3_key="x",
            status="ready",
            doc_domain="job_description",
            role_profile={
                "domain": "technical",
                "seniority": "entry",
                "focusAreas": ["Python", "AWS", "cloud"],
                "questionMix": {"behavioral": 40, "roleSpecific": 30, "scenario": 30},
            },
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
            content="Python, AWS, ML pipelines. 5+ years.",
            page_number=1,
            section_type="qualifications",
            doc_domain="job_description",
            embedding=mock_vec,
        )
        db.add(chunk)
        await db.commit()

    resp = await client.post(
        "/interview/generate",
        json={
            "document_id": str(doc_id),
            "difficulty": "junior",
            "num_questions": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "questions" in data
    assert len(data["questions"]) == 2
    q0 = data["questions"][0]
    assert q0["type"] == "role_specific"
    assert "Python" in q0["question"]
    assert q0.get("competency_label") == "Python" or "Python" in (q0.get("key_topics") or []) or q0.get("focus_area") == "Python"
    assert len(q0["evidence"]) >= 1
    assert "chunk_id" in q0["evidence"][0]
    assert "page_number" in q0["evidence"][0]
    assert "snippet" in q0["evidence"][0]
    assert len(q0["rubric_bullets"]) == 3


# --- Evaluate endpoint tests ---


@pytest.mark.asyncio
async def test_interview_evaluate_requires_valid_input(client, demo_key_off, force_auth):
    """Evaluate returns 422 for invalid body."""
    await force_auth()
    resp = await client.post("/interview/evaluate", json={})
    assert resp.status_code == 422

    resp = await client.post(
        "/interview/evaluate",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question_id": "11111111-1111-1111-1111-111111111111",
            "answer_text": "",  # min_length=1
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_interview_evaluate_question_not_found(client, demo_key_off, monkeypatch, force_auth):
    """Evaluate returns 404 for unknown question."""
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    await force_auth()
    resp = await client.post(
        "/interview/evaluate",
        json={
            "document_id": "11111111-1111-1111-1111-111111111111",
            "question_id": "11111111-1111-1111-1111-111111111111",
            "answer_text": "I have 5 years of Python experience.",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_interview_evaluate_success(client, demo_key_off, monkeypatch, force_auth):
    """Evaluate returns score, strengths, gaps, improved_answer, follow_up_questions, citations."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewQuestion, InterviewSession, User

    async def _mock_evaluate(*args, **kwargs):
        return {
            "score": 7.0,
            "strengths": ["Mentioned Python experience"],
            "gaps": ["No AWS specifics"],
            "improved_answer": "I built ML pipelines on AWS...",
            "follow_up_questions": ["Which AWS services?", "Scale of data?"],
            "evidence_used": [
                {"quote": "Python, AWS", "sourceId": "aa", "page": 1, "chunkId": "aa"}
            ],
            "evidence_for_scoring": [
                {"snippet": "Python, AWS, scalability, collaboration, concrete examples"},
            ],
        }

    monkeypatch.setattr("app.routers.interview.runtime.evaluate_answer_with_retrieval", _mock_evaluate)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")

    user_id = uuid.uuid4()
    doc_id = None
    question_id = None
    session_id = None
    async with async_session_maker() as db:
        user = User(id=user_id, email="eval-test@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="eval-test@t.local")
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
        session = InterviewSession(
            user_id=user_id,
            document_id=doc_id,
            mode="technical",
            difficulty="junior",
        )
        db.add(session)
        await db.flush()
        session_id = session.id
        q = InterviewQuestion(
            session_id=session.id,
            type="technical",
            question="Describe your Python and AWS experience.",
            rubric_json={
                "bullets": ["Concrete examples", "Scale"],
                "evidence": [{"chunk_id": "aa", "page_number": 1, "snippet": "Python, AWS"}],
                "key_topics": ["Python", "AWS"],
            },
        )
        db.add(q)
        await db.flush()
        question_id = q.id
        await db.commit()

    resp = await client.post(
        "/interview/evaluate",
        json={
            "document_id": str(doc_id),
            "question_id": str(question_id),
            "answer_text": "I have 5 years of Python experience.",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer_id" in data
    assert 0 <= data["score"] <= 100
    assert "score_breakdown" in data
    for k in ("relevance_to_context", "completeness", "clarity", "jd_alignment", "overall"):
        assert k in data["score_breakdown"]
    assert "feedback_summary" in data and data["feedback_summary"]
    assert "strengths" in data
    assert "gaps" in data
    assert "improved_answer" in data
    assert "follow_up_questions" in data
    assert "evidence_used" in data
    assert len(data["evidence_used"]) == 1
    assert data["evidence_used"][0]["sourceId"] == "aa"
    assert data["evidence_used"][0]["quote"]

    async with async_session_maker() as db:
        from sqlalchemy import select

        from app.models import InterviewSession

        r = await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
        sess = r.scalar_one()
        assert sess.performance_profile is not None
        for k in ("technical", "behavioral", "communication", "overall"):
            assert k in sess.performance_profile
            assert isinstance(sess.performance_profile[k], (int, float))


# --- GET endpoint tests ---


@pytest.mark.asyncio
async def test_list_sessions(client, demo_key_off, force_auth):
    """GET /interview/sessions returns sessions for user."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="sessions@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="sessions@t.local")
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
        session = InterviewSession(
            user_id=user_id,
            document_id=doc_id,
            mode="technical",
            difficulty="junior",
        )
        db.add(session)
        await db.flush()
        q = InterviewQuestion(
            session_id=session.id,
            type="technical",
            question="Test question?",
            rubric_json={"bullets": [], "evidence": [], "key_topics": []},
        )
        db.add(q)
        await db.commit()

    resp = await client.get("/interview/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    s = data[0]
    assert s["document_id"] == str(doc_id)
    assert s["mode"] == "technical"
    assert s["question_count"] >= 1


@pytest.mark.asyncio
async def test_get_session(client, demo_key_off, force_auth):
    """GET /interview/sessions/{id} returns session with questions."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="get-session@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="get-session@t.local")
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
        session = InterviewSession(
            user_id=user_id,
            document_id=doc.id,
            mode="behavioral",
            difficulty="mid",
        )
        db.add(session)
        await db.flush()
        session_id = session.id
        q = InterviewQuestion(
            session_id=session.id,
            type="behavioral",
            question="Tell me about a challenge.",
            rubric_json={
                "bullets": ["STAR format"],
                "evidence": [{"chunk_id": "x", "page_number": 1, "snippet": "challenge"}],
                "key_topics": ["resilience"],
            },
        )
        db.add(q)
        await db.commit()

    resp = await client.get(f"/interview/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(session_id)
    assert data["mode"] == "behavioral"
    assert len(data["questions"]) >= 1
    assert data["questions"][0]["question"] == "Tell me about a challenge."


@pytest.mark.asyncio
async def test_get_question(client, demo_key_off, force_auth):
    """GET /interview/questions/{id} returns question."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="get-q@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="get-q@t.local")
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
        session = InterviewSession(
            user_id=user_id,
            document_id=doc.id,
            mode="technical",
            difficulty="junior",
        )
        db.add(session)
        await db.flush()
        q = InterviewQuestion(
            session_id=session.id,
            type="technical",
            question="What is Python?",
            rubric_json={
                "bullets": ["Explain clearly"],
                "evidence": [{"chunk_id": "e1", "page_number": 2, "snippet": "Python"}],
                "key_topics": ["Python"],
            },
        )
        db.add(q)
        await db.flush()
        question_id = q.id
        session_id = session.id
        await db.commit()

    resp = await client.get(f"/interview/questions/{question_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(question_id)
    assert data["question"] == "What is Python?"
    assert data["session_id"] == str(session_id)
    assert "Python" in data["key_topics"]


@pytest.mark.asyncio
async def test_interview_session_analytics(client, demo_key_off, force_auth):
    """GET /interview/{session_id}/analytics aggregates scores and competency stats."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewAnswer, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="analytics@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="analytics@t.local")
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
        session = InterviewSession(
            user_id=user_id,
            document_id=doc.id,
            mode="role_driven",
            difficulty="mid",
        )
        db.add(session)
        await db.flush()
        q_a = InterviewQuestion(
            session_id=session.id,
            type="behavioral",
            question="Q1",
            rubric_json={
                "bullets": [],
                "evidence": [],
                "key_topics": [],
                "competency_id": "c1",
                "competency_label": "Communication",
            },
        )
        q_b = InterviewQuestion(
            session_id=session.id,
            type="scenario",
            question="Q2",
            rubric_json={
                "bullets": [],
                "evidence": [],
                "key_topics": [],
                "competency_id": "c2",
                "competency_label": "System design",
            },
        )
        db.add(q_a)
        db.add(q_b)
        await db.flush()
        for q, sc in [(q_a, 40.0), (q_b, 50.0), (q_a, 70.0), (q_b, 80.0)]:
            db.add(
                InterviewAnswer(
                    question_id=q.id,
                    answer_text="answer",
                    score=sc,
                    feedback_summary="s",
                    strengths=[],
                    weaknesses=[],
                    feedback_json={"score_breakdown": {}},
                )
            )
        await db.commit()
        session_id = session.id

    resp = await client.get(f"/interview/{session_id}/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == str(session_id)
    assert data["answer_count"] == 4
    assert data["average_score"] == 60.0
    assert len(data["score_trend"]) == 4
    assert data["score_trend"][0]["score"] == 40.0
    assert data["score_trend"][-1]["score"] == 80.0
    assert data["improvement"]["improvement_delta"] is not None
    assert data["improvement"]["improvement_delta"] > 0
    labels = {c["competency_label"] for c in data["strongest_competencies"]}
    assert "Communication" in labels or "System design" in labels


@pytest.mark.asyncio
async def test_interview_analytics_overview(client, demo_key_off, force_auth):
    """GET /interview/analytics/overview aggregates across sessions for the user."""
    from app.db.base import async_session_maker
    from app.models import Document, InterviewAnswer, InterviewQuestion, InterviewSession, User

    user_id = uuid.uuid4()
    async with async_session_maker() as db:
        user = User(id=user_id, email="overview@t.local")
        db.add(user)
        await db.commit()
    await force_auth(user_id=user_id, email="overview@t.local")
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

        # Older session: lower scores
        s_old = InterviewSession(
            user_id=user_id,
            document_id=doc.id,
            mode="role_driven",
            difficulty="mid",
        )
        db.add(s_old)
        await db.flush()
        q_old = InterviewQuestion(
            session_id=s_old.id,
            type="behavioral",
            question="Old Q",
            rubric_json={
                "bullets": [],
                "evidence": [],
                "key_topics": [],
                "competency_label": "Leadership",
            },
        )
        db.add(q_old)
        await db.flush()
        db.add(
            InterviewAnswer(
                question_id=q_old.id,
                answer_text="a",
                score=50.0,
                feedback_summary="s",
                strengths=[],
                weaknesses=[],
                feedback_json={"score_breakdown": {}},
            )
        )

        # Newer session: higher scores
        s_new = InterviewSession(
            user_id=user_id,
            document_id=doc.id,
            mode="role_driven",
            difficulty="mid",
        )
        db.add(s_new)
        await db.flush()
        q_new = InterviewQuestion(
            session_id=s_new.id,
            type="scenario",
            question="New Q",
            rubric_json={
                "bullets": [],
                "evidence": [],
                "key_topics": [],
                "competency_label": "System design",
            },
        )
        db.add(q_new)
        await db.flush()
        db.add(
            InterviewAnswer(
                question_id=q_new.id,
                answer_text="b",
                score=80.0,
                feedback_summary="s",
                strengths=[],
                weaknesses=[],
                feedback_json={"score_breakdown": {}},
            )
        )
        await db.commit()

    resp = await client.get("/interview/analytics/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_session_count"] == 2
    assert data["total_answer_count"] == 2
    assert data["overall_average_score"] == 65.0
    assert len(data["score_trend"]) == 2
    assert data["last_session_vs_prior_percent_change"] is not None
    assert data["last_session_vs_prior_percent_change"] > 0
    assert data["focus_area_hint"] == "Leadership"
    assert len(data["recent_sessions"]) >= 2


def test_interview_scoring_is_deterministic():
    from app.services.interview_scoring import compute_score_breakdown

    ev = [{"snippet": "Python microservices AWS monitoring"}]
    bullets = ["Use concrete metrics", "Mention reliability"]
    must = ["Python"]
    rp = {"focusAreas": ["backend", "cloud"]}
    a = "We used Python on AWS with Prometheus monitoring and on-call rotations."
    b1 = compute_score_breakdown(a, ev, bullets, must, rp, "Backend")
    b2 = compute_score_breakdown(a, ev, bullets, must, rp, "Backend")
    assert b1 == b2
    assert b1["overall"] == b2["overall"]
