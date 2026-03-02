"""Add UNIQUE(document_id, chunk_index) to document_chunks.

Ensures one chunk per index per document. No other UNIQUE should limit rows globally.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "20250228130000"
down_revision: Union[str, None] = "20250228120000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_document_chunks_document_id_chunk_index",
        "document_chunks",
        ["document_id", "chunk_index"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_document_chunks_document_id_chunk_index",
        "document_chunks",
        type_="unique",
    )
