"""Add generated tsvector column and GIN index for document chunk content."""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TSVECTOR

revision: str = "20250307000000"
down_revision: Union[str, None] = "20250306000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column(
            "search_vector",
            TSVECTOR(),
            sa.Computed(
                "to_tsvector('english'::regconfig, COALESCE(content, ''::text))",
                persisted=True,
            ),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_document_chunks_search_vector",
        "document_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_search_vector", table_name="document_chunks")
    op.drop_column("document_chunks", "search_vector")
