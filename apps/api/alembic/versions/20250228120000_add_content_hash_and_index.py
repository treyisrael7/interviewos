"""Add content_hash and index on (document_id, is_low_signal)

Revision ID: 20250228120000
Revises: 20250228110000
Create Date: 2025-02-28

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20250228120000"
down_revision: Union[str, None] = "20250228110000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("content_hash", sa.String(), nullable=True),
    )
    op.execute("ALTER TABLE document_chunks ALTER COLUMN quality_score SET DEFAULT 1.0")
    op.create_index(
        "ix_document_chunks_doc_low_signal",
        "document_chunks",
        ["document_id", "is_low_signal"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_doc_low_signal", table_name="document_chunks")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN quality_score DROP DEFAULT")
    op.drop_column("document_chunks", "content_hash")
