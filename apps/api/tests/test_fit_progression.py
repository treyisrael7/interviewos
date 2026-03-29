"""Tests for fit progression helper."""

import uuid
from datetime import datetime, timezone

from app.services.fit_progression import build_fit_progression


class _Row:
    def __init__(self, jd, rs, score, created_at):
        self.job_description_id = jd
        self.resume_id = rs
        self.fit_score = score
        self.created_at = created_at


def test_progression_improved_message():
    jd = uuid.uuid4()
    rs = uuid.uuid4()
    rows = [
        _Row(jd, rs, 40, datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _Row(jd, rs, 72, datetime(2026, 2, 1, tzinfo=timezone.utc)),
    ]
    out = build_fit_progression(rows)
    assert len(out) == 1
    assert out[0]["first_score"] == 40
    assert out[0]["latest_score"] == 72
    assert out[0]["delta"] == 32
    assert out[0]["improved"] is True
    assert "improved from 40 → 72" in out[0]["message"]


def test_progression_single_run():
    jd = uuid.uuid4()
    rs = uuid.uuid4()
    rows = [_Row(jd, rs, 55, datetime(2026, 1, 1, tzinfo=timezone.utc))]
    out = build_fit_progression(rows)
    assert out[0]["improved"] is None
    assert out[0]["first_score"] == out[0]["latest_score"] == 55
