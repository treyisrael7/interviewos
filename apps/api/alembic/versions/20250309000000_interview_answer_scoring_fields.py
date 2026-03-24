"""Add interview answer scoring summary and strengths/weaknesses; scale legacy scores to 0–100."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "20250309000000"
down_revision = "20250308000000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "interview_answers",
        sa.Column("feedback_summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "interview_answers",
        sa.Column(
            "strengths",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "interview_answers",
        sa.Column(
            "weaknesses",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.execute(
        """
        UPDATE interview_answers
        SET score = score * 10.0
        WHERE score IS NOT NULL AND score <= 10.5
        """
    )


def downgrade() -> None:
    op.drop_column("interview_answers", "weaknesses")
    op.drop_column("interview_answers", "strengths")
    op.drop_column("interview_answers", "feedback_summary")
