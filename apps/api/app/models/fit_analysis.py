from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FitAnalysis(Base):
    """Persisted result of POST /analyze-fit for history and score progression."""

    __tablename__ = "fit_analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_description_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    fit_score: Mapped[int] = mapped_column(sa.Integer(), nullable=False)
    matches_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    gaps_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    recommendations_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    summary: Mapped[str] = mapped_column(sa.Text(), nullable=False, server_default="")
    query_fingerprint: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    jd_chunk_fingerprint: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    resume_chunk_fingerprint: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )

    __table_args__ = (
        sa.Index("ix_fit_analyses_user_created", "user_id", "created_at"),
        sa.Index("ix_fit_analyses_user_jd_resume", "user_id", "job_description_id", "resume_id"),
        sa.Index(
            "ix_fit_analyses_cache_lookup",
            "user_id",
            "job_description_id",
            "resume_id",
            "query_fingerprint",
            "jd_chunk_fingerprint",
            "resume_chunk_fingerprint",
        ),
    )
