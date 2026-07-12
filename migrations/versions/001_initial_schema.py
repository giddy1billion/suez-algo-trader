"""Initial schema - all tables for fresh PostgreSQL deployment.

Revision ID: 001
Revises: None
Create Date: 2026-07-12

Creates all tables from the four ORM bases:
  - TradingBase: trades, signal_logs, portfolio_snapshots, trade_journal, market_data, sector_cache
  - EventBase: events
  - SnapshotBase: snapshots
  - ConfigBase: system_configuration, configuration_audit_log
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '001'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all application tables."""

    # --- Trading tables ---

    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('side', sa.String(length=10), nullable=False),
        sa.Column('qty', sa.Float(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('order_type', sa.String(length=20), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('order_id', sa.String(length=64), nullable=True),
        sa.Column('strategy', sa.String(length=50), nullable=True),
        sa.Column('signal_confidence', sa.Float(), nullable=True),
        sa.Column('stop_loss', sa.Float(), nullable=True),
        sa.Column('take_profit', sa.Float(), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('pnl_pct', sa.Float(), nullable=True),
        sa.Column('fees', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('prediction_id', sa.String(length=64), nullable=True),
        sa.Column('model_version', sa.String(length=64), nullable=True),
        sa.Column('contract_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column('filled_at', sa.DateTime(), nullable=True),
        sa.Column('closed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_id'),
    )
    op.create_index('ix_trades_symbol', 'trades', ['symbol'])
    op.create_index('ix_trades_strategy', 'trades', ['strategy'])
    op.create_index('ix_trades_created_at', 'trades', ['created_at'])
    op.create_index('ix_trades_prediction_id', 'trades', ['prediction_id'])
    op.create_index('ix_trades_contract_id', 'trades', ['contract_id'])

    op.create_table(
        'signal_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('strategy', sa.String(length=50), nullable=False),
        sa.Column('signal', sa.String(length=20), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('price_at_signal', sa.Float(), nullable=True),
        sa.Column('indicators', sa.Text(), nullable=True),
        sa.Column('was_executed', sa.Boolean(), nullable=True, server_default=sa.text('false')),
        sa.Column('outcome', sa.String(length=20), nullable=True),
        sa.Column('outcome_pnl', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_signal_logs_symbol', 'signal_logs', ['symbol'])
    op.create_index('ix_signal_logs_created_at', 'signal_logs', ['created_at'])

    op.create_table(
        'portfolio_snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('total_equity', sa.Float(), nullable=False),
        sa.Column('cash', sa.Float(), nullable=True),
        sa.Column('positions_value', sa.Float(), nullable=True),
        sa.Column('unrealized_pnl', sa.Float(), nullable=True),
        sa.Column('daily_pnl', sa.Float(), nullable=True),
        sa.Column('open_positions', sa.Integer(), nullable=True),
        sa.Column('win_rate', sa.Float(), nullable=True),
        sa.Column('sharpe_ratio', sa.Float(), nullable=True),
        sa.Column('max_drawdown', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_portfolio_snapshots_created_at', 'portfolio_snapshots', ['created_at'])

    op.create_table(
        'trade_journal',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('trade_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('side', sa.String(length=10), nullable=False),
        sa.Column('entry_price', sa.Float(), nullable=True),
        sa.Column('entry_time', sa.DateTime(), nullable=True),
        sa.Column('qty', sa.Float(), nullable=True),
        sa.Column('strategy_name', sa.String(length=50), nullable=True),
        sa.Column('model_version', sa.String(length=20), nullable=True),
        sa.Column('prediction', sa.String(length=20), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('features_snapshot', sa.Text(), nullable=True),
        sa.Column('contract_id', sa.String(length=64), nullable=True),
        sa.Column('exit_price', sa.Float(), nullable=True),
        sa.Column('exit_time', sa.DateTime(), nullable=True),
        sa.Column('exit_reason', sa.String(length=50), nullable=True),
        sa.Column('pnl', sa.Float(), nullable=True),
        sa.Column('pnl_pct', sa.Float(), nullable=True),
        sa.Column('holding_bars', sa.Integer(), nullable=True),
        sa.Column('market_regime', sa.String(length=20), nullable=True),
        sa.Column('volatility_at_entry', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_trade_journal_trade_id', 'trade_journal', ['trade_id'])
    op.create_index('ix_trade_journal_symbol', 'trade_journal', ['symbol'])
    op.create_index('ix_trade_journal_contract_id', 'trade_journal', ['contract_id'])
    op.create_index('ix_trade_journal_created_at', 'trade_journal', ['created_at'])

    op.create_table(
        'market_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('open', sa.Float(), nullable=True),
        sa.Column('high', sa.Float(), nullable=True),
        sa.Column('low', sa.Float(), nullable=True),
        sa.Column('close', sa.Float(), nullable=True),
        sa.Column('volume', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_market_data_symbol_ts', 'market_data', ['symbol', 'timestamp'])

    op.create_table(
        'sector_cache',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('sector', sa.String(length=50), nullable=False),
        sa.Column('source', sa.String(length=20), nullable=True, server_default='manual'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_sector_cache_symbol', 'sector_cache', ['symbol'], unique=True)

    # --- Event store ---

    op.create_table(
        'events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('event_id', sa.String(length=64), nullable=False),
        sa.Column('timestamp', sa.String(length=64), nullable=False),
        sa.Column('source', sa.String(length=100), nullable=True, server_default=''),
        sa.Column('payload', sa.Text(), nullable=False),
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('event_id'),
    )
    op.create_index('idx_events_event_id', 'events', ['event_id'], unique=True)
    op.create_index('idx_events_session', 'events', ['session_id'])
    op.create_index('idx_events_timestamp', 'events', ['timestamp'])
    op.create_index('idx_events_type', 'events', ['event_type'])

    # --- Snapshot store ---

    op.create_table(
        'snapshots',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.Column('timestamp', sa.String(length=64), nullable=False),
        sa.Column('last_event_id', sa.Integer(), nullable=False),
        sa.Column('state', sa.Text(), nullable=False),
        sa.Column('schema_version', sa.String(length=20), nullable=False, server_default='1'),
        sa.Column('engine_version', sa.String(length=20), nullable=False, server_default='1.0.0'),
        sa.Column('config_hash', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('created_at', sa.String(length=64), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_snapshots_session', 'snapshots', ['session_id'])

    # --- Configuration tables ---

    op.create_table(
        'system_configuration',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('value_type', sa.String(length=20), nullable=False, server_default='str'),
        sa.Column('description', sa.Text(), nullable=True, server_default=''),
        sa.Column('is_secret', sa.Boolean(), nullable=True, server_default=sa.text('false')),
        sa.Column('is_editable', sa.Boolean(), nullable=True, server_default=sa.text('true')),
        sa.Column('validation_rule', sa.String(length=200), nullable=True, server_default=''),
        sa.Column('updated_by', sa.String(length=100), nullable=True, server_default='system'),
        sa.Column('version', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('category', 'key', name='uq_category_key'),
    )
    op.create_index('ix_system_config_category', 'system_configuration', ['category'])

    op.create_table(
        'configuration_audit_log',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('old_value', sa.Text(), nullable=True),
        sa.Column('new_value', sa.Text(), nullable=False),
        sa.Column('old_version', sa.Integer(), nullable=True),
        sa.Column('new_version', sa.Integer(), nullable=False),
        sa.Column('changed_by', sa.String(length=100), nullable=False, server_default='system'),
        sa.Column('change_reason', sa.Text(), nullable=True, server_default=''),
        sa.Column('changed_at', sa.DateTime(), nullable=True, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_config_audit_category_key', 'configuration_audit_log', ['category', 'key'])
    op.create_index('ix_config_audit_timestamp', 'configuration_audit_log', ['changed_at'])


def downgrade() -> None:
    """Drop all application tables."""
    op.drop_table('configuration_audit_log')
    op.drop_table('system_configuration')
    op.drop_table('snapshots')
    op.drop_table('events')
    op.drop_table('sector_cache')
    op.drop_table('market_data')
    op.drop_table('trade_journal')
    op.drop_table('portfolio_snapshots')
    op.drop_table('signal_logs')
    op.drop_table('trades')
