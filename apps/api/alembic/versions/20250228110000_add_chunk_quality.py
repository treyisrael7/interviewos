"""Add quality_score and is_low_signal to document_chunks

Revision ID: 20250228110000
Revises: 20250228100000
Create Date: 2025-02-28

Low-signal chunks are excluded from retrieval (document-agnostic boilerplate/nav).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20250228110000"
down_revision: Union[str, None] = "20250228100000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("quality_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("is_low_signal", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "is_low_signal")
    op.drop_column("document_chunks", "quality_score")
