"""tranquis: quota, password_reset, app_source

Revision ID: a3f8c1d4e9b2
Revises: 5c1e7a9d3b42
Create Date: 2026-10-09 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'a3f8c1d4e9b2'
down_revision = '966a3aa0830c'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. app_source column on users
    op.add_column(
        'users',
        sa.Column('app_source', sa.String(), nullable=False, server_default='careermind'),
    )

    # 2. translation_history table
    op.create_table(
        'translation_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_text', sa.Text(), nullable=False),
        sa.Column('translated_text', sa.Text(), nullable=False),
        sa.Column('source_lang', sa.String(), nullable=False, server_default='auto'),
        sa.Column('target_lang', sa.String(), nullable=False),
        sa.Column('mode', sa.String(), nullable=False, server_default='text'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_translation_history_user_id', 'translation_history', ['user_id'])

    # 3. phrasebook_entries table
    op.create_table(
        'phrasebook_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('translation_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('translation_history.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index('ix_phrasebook_entries_user_id', 'phrasebook_entries', ['user_id'])

    # 4. translation_quotas table
    op.create_table(
        'translation_quotas',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('quota_date', sa.Date(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False, default=0),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), onupdate=sa.text('now()')),
    )
    op.create_index('ix_translation_quotas_user_id', 'translation_quotas', ['user_id'])
    op.create_unique_constraint(
        'uq_translation_quota_user_date', 'translation_quotas', ['user_id', 'quota_date']
    )

    # 3. password_reset_tokens table
    op.create_table(
        'password_reset_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('token_hash', sa.String(), nullable=False, unique=True),
        sa.Column('used', sa.Boolean(), nullable=False, default=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )


def downgrade() -> None:
    op.drop_table('password_reset_tokens')
    op.drop_table('translation_quotas')
    op.drop_table('phrasebook_entries')
    op.drop_table('translation_history')
    op.drop_column('users', 'app_source')
