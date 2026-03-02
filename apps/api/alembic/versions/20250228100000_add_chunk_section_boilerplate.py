"""Add section and is_boilerplate to document_chunks

Revision ID: 20250228100000
Revises: 20250228000000
Create Date: 2025-02-28

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20250228100000"
down_revision: Union[str, None] = "20250228000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("section", sa.String(), nullable=True),
    )
    op.add_column(
        "document_chunks",
        sa.Column("is_boilerplate", sa.Boolean(), nullable=False, server_default="false"),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "is_boilerplate")
    op.drop_column("document_chunks", "section")
