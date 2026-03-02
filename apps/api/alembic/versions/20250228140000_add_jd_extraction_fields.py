"""Add JD extraction fields: documents.jd_extraction_json, chunks.section_type, skills_detected, doc_domain."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20250228140000"
down_revision: Union[str, None] = "20250228130000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("jd_extraction_json", sa.dialects.postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("section_type", sa.String(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("skills_detected", sa.dialects.postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("doc_domain", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_document_chunks_section_type",
        "document_chunks",
        ["document_id", "section_type"],
        unique=False,
    )
    op.create_index(
        "ix_document_chunks_doc_domain",
        "document_chunks",
        ["doc_domain"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_doc_domain", table_name="document_chunks")
    op.drop_index("ix_document_chunks_section_type", table_name="document_chunks")
    op.drop_column("document_chunks", "doc_domain")
    op.drop_column("document_chunks", "skills_detected")
    op.drop_column("document_chunks", "section_type")
    op.drop_column("documents", "jd_extraction_json")
