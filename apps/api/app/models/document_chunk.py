import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# OpenAI text-embedding-ada-002 dimension; adjust if using a different model
EMBEDDING_DIM = 1536


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    page_number: Mapped[int] = mapped_column(nullable=False)
    section: Mapped[str | None] = mapped_column(nullable=True)
    is_boilerplate: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    quality_score: Mapped[float | None] = mapped_column(nullable=True, server_default="1.0")
    is_low_signal: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    content_hash: Mapped[str | None] = mapped_column(nullable=True)
    section_type: Mapped[str | None] = mapped_column(nullable=True)
    skills_detected: Mapped[list | None] = mapped_column(JSONB(), nullable=True)
    doc_domain: Mapped[str | None] = mapped_column(nullable=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("now()"),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_doc_low_signal", "document_id", "is_low_signal"),
        Index("ix_document_chunks_section_type", "document_id", "section_type"),
        Index("ix_document_chunks_doc_domain", "doc_domain"),
    )
