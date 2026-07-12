# PostgreSQL Integration Guide

## Overview

The algo-trader supports both **SQLite** (default, zero-config) and **PostgreSQL** (production) as its persistence backend. All ORM models, event store, snapshot store, and configuration service work transparently with either backend.

## Quick Start

### 1. Set the Database URL

In your `.env` file or environment:

```bash
# PostgreSQL (production)
DATABASE_URL=postgresql://user:password@host:5432/algo_trader?sslmode=require

# SQLite (development default - no configuration needed)
# DATABASE_URL=sqlite:///data_cache/trading.db
```

### 2. Run Migrations

For PostgreSQL, Alembic migrations run automatically on startup. For manual execution:

```bash
# Run pending migrations
DATABASE_URL="postgresql://..." python -c "from src.utils.database import run_migrations; run_migrations()"

# Or via Alembic CLI directly
DATABASE_URL="postgresql://..." python -m alembic -c alembic.ini upgrade head
```

For SQLite, tables are created automatically via `create_all()` - no migration step is needed.

### 3. Migrate Existing Data (Optional)

To migrate from SQLite to PostgreSQL:

```bash
python scripts/migrate_sqlite_to_pg.py \
  --trading-db data_cache/trading.db \
  --events-db data_cache/events.db \
  --snapshots-db data_cache/snapshots.db \
  --target-url "postgresql://user:password@host:5432/algo_trader" \
  --dry-run  # Remove for actual migration
```

---

## Architecture

### Database Schemas

All tables are managed by four SQLAlchemy `DeclarativeBase` classes:

| Base | Tables | Purpose |
|------|--------|---------|
| `TradingBase` | trades, signal_logs, portfolio_snapshots, trade_journal, market_data, sector_cache | Core trading OLTP data |
| `EventBase` | events | Durable event log for replay and auditing |
| `SnapshotBase` | snapshots | Periodic state snapshots for fast recovery |
| `ConfigBase` | system_configuration, configuration_audit_log | Runtime config with audit trail |

### Connection Pool Configuration (PostgreSQL)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pool_size` | 5 | Steady-state connections |
| `max_overflow` | 3 | Burst capacity (total max: 8) |
| `pool_timeout` | 30s | Wait time for available connection |
| `pool_recycle` | 1800s | Max connection age before refresh |
| `pool_pre_ping` | True | Verify connection before checkout |

### Lifecycle Integration

The database pool participates in the application's deterministic shutdown sequence:

```
RuntimeManager.shutdown()
  -> Scheduler.shutdown()
    -> Market/Trade streams stop
      -> EventStore.close()        # Dispose event engine pool
        -> SnapshotStore.close()   # Dispose snapshot engine pool
          -> reset_engine()        # Dispose shared engine pool
            -> Telegram bot stop
```

---

## Alembic Migrations

### Directory Structure

```
alembic.ini                    # Configuration (script_location = migrations/)
migrations/
  env.py                       # Imports all 4 ORM bases
  script.py.mako               # Template for new migrations
  versions/
    001_initial_schema.py      # All tables for fresh PG deployment
```

### Creating a New Migration

```bash
# Auto-generate from ORM model changes
DATABASE_URL="postgresql://..." python -c "
from alembic.config import Config
from alembic import command
cfg = Config('alembic.ini')
command.revision(cfg, autogenerate=True, message='description_of_change')
"
```

### Migration Best Practices

- Always test migrations against a disposable database before production
- Migrations are forward-only by default; test `downgrade()` paths
- The `run_migrations()` function in `src/utils/database.py` runs on startup for PostgreSQL only
- SQLite bypasses migrations entirely (uses `create_all()` for simplicity)

---

## Telemetry Events

The PostgreSQL integration emits structured log events following the project's event taxonomy:

### Pool Lifecycle

| Event | Level | Fields | Trigger |
|-------|-------|--------|---------|
| `db.pool.initialized` | INFO | backend, pool_size, max_overflow, pool_timeout, pool_recycle | Engine creation (PG only) |
| `db.pool.shutdown` | INFO | backend | `dispose_engine()` or `reset_engine()` |
| `db.pool.connection_invalidated` | WARNING | error | Connection dropped/invalid |
| `db.pool.new_connection` | DEBUG | - | New physical connection created |
| `db.pool.dispose_with_active_connections` | WARNING | checked_out | Dispose while connections in use |
| `db.connection.slow_return` | WARNING | hold_seconds, threshold | Connection held longer than threshold |

### Query Performance

| Event | Level | Fields | Trigger |
|-------|-------|--------|---------|
| `db.query.slow` | WARNING | elapsed_seconds, threshold, statement_preview | Query exceeds `SLOW_QUERY_THRESHOLD_SECONDS` (default: 1.0s) |

### Migrations

| Event | Level | Fields | Trigger |
|-------|-------|--------|---------|
| `db.migration.starting` | INFO | target | Before running Alembic upgrade |
| `db.migration.completed` | INFO | target | Migration successful |
| `db.migration.failed` | ERROR | error | Migration error |
| `db.migration.skipped` | DEBUG | reason | SQLite or missing alembic.ini |

### Health Check

| Event | Level | Fields | Trigger |
|-------|-------|--------|---------|
| `db.health_check.ok` | INFO | backend, healthy, pool_size, checked_out, overflow, checked_in | Successful health check |
| `db.health_check.failed` | ERROR | error | Connection test failed |

---

## Health Check API

```python
from src.utils.database import db_health_check

status = db_health_check()
# Returns:
# {
#   "backend": "postgresql",
#   "healthy": True,
#   "pool_size": 5,
#   "checked_out": 1,
#   "overflow": 0,
#   "checked_in": 4,
#   "pool_timeout": 30
# }
```

---

## Testing

### Run All Database Tests

```bash
# Unit tests (no PG server needed)
python -m pytest tests/test_pg_integration.py tests/test_pg_lifecycle.py -v

# Live PostgreSQL tests (requires running PG)
DATABASE_URL_TEST="postgresql://test:test@localhost:5432/test_algo" \
  python -m pytest tests/test_pg_lifecycle.py::TestLivePostgreSQL -v
```

### Test Coverage

- **Dialect compatibility**: Verifies all ORM models generate valid PostgreSQL DDL
- **Engine factory**: Pool configuration, URL routing, singleton behavior
- **Transactional rollback**: Unique constraint violations roll back cleanly
- **Connection lifecycle**: Dispose, reconnect, health check
- **Observability**: Structured log events emitted at correct times
- **Concurrent writes**: Thread-safe access (validated on PG)
- **Migration logic**: Dry-run, fresh target, skip behavior

---

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DATABASE_URL` | `sqlite:///data_cache/trading.db` | Primary database URL |
| `DATABASE_URL_TEST` | - | Test database URL (for integration tests) |

The `database_url` setting in `config/settings.py` accepts any SQLAlchemy-compatible URL:

```python
# PostgreSQL
database_url = "postgresql://user:pass@host:5432/dbname?sslmode=require"

# SQLite (relative path)
database_url = "sqlite:///data_cache/trading.db"

# SQLite (absolute path)
database_url = "sqlite:////var/data/trading.db"
```

---

## Backward Compatibility

- SQLite remains the default - no configuration change is required for existing deployments
- All existing SQLite databases continue to work without modification
- The migration script (`scripts/migrate_sqlite_to_pg.py`) handles data transfer when upgrading
- Both backends use identical ORM models - application code is backend-agnostic
