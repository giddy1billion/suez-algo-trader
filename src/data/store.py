"""
Data Storage — SQLAlchemy models for trade history, signals, and portfolio tracking.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Text, Index
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """Record of executed trades."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # buy / sell
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    order_type = Column(String(20))  # market / limit / bracket
    status = Column(String(20))  # filled / cancelled / rejected
    order_id = Column(String(64), unique=True)
    strategy = Column(String(50), index=True)
    signal_confidence = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float)  # Realized P&L (set when position closed)
    pnl_pct = Column(Float)
    fees = Column(Float, default=0.0)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    filled_at = Column(DateTime)
    closed_at = Column(DateTime)


class SignalLog(Base):
    """Log of all generated signals (for analysis and retraining)."""
    __tablename__ = 'signal_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    strategy = Column(String(50), nullable=False)
    signal = Column(String(20), nullable=False)  # BUY / SELL / HOLD
    confidence = Column(Float)
    price_at_signal = Column(Float)
    indicators = Column(Text)  # JSON blob
    was_executed = Column(Boolean, default=False)
    outcome = Column(String(20))  # win / loss / pending
    outcome_pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class PortfolioSnapshot(Base):
    """Periodic snapshots of portfolio state for performance tracking."""
    __tablename__ = 'portfolio_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    total_equity = Column(Float, nullable=False)
    cash = Column(Float)
    positions_value = Column(Float)
    unrealized_pnl = Column(Float)
    daily_pnl = Column(Float)
    open_positions = Column(Integer)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class JournalEntry(Base):
    """Trade journal entry for analysis and model retraining."""
    __tablename__ = 'trade_journal'

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, index=True)  # links to Trade.id
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)

    # Entry context
    entry_price = Column(Float)
    entry_time = Column(DateTime)
    qty = Column(Float)

    # ML/Strategy context at time of entry
    strategy_name = Column(String(50))
    model_version = Column(String(20))
    prediction = Column(String(20))  # BUY/SELL/HOLD
    confidence = Column(Float)
    features_snapshot = Column(Text)  # JSON: all feature values at entry time

    # Exit context
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_reason = Column(String(50))  # signal, stop_loss, take_profit, manual, timeout

    # Outcome
    pnl = Column(Float)
    pnl_pct = Column(Float)
    holding_bars = Column(Integer)

    # Market context
    market_regime = Column(String(20))  # trending, ranging, volatile
    volatility_at_entry = Column(Float)  # ATR% at entry

    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class MarketData(Base):
    """Cached market data for backtesting and feature calculation."""
    __tablename__ = 'market_data'
    __table_args__ = (
        Index('ix_market_data_symbol_ts', 'symbol', 'timestamp'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


class SectorCache(Base):
    """Cached sector classifications for symbols (manual or auto-resolved)."""
    __tablename__ = 'sector_cache'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    sector = Column(String(50), nullable=False)
    source = Column(String(20), default="manual")  # manual / auto
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────
# Database Manager
# ──────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """Manages database connections and provides CRUD operations."""

    def __init__(self, database_url: str = "sqlite:///data_cache/trading.db"):
        # Thread-safe SQLite configuration
        engine_kwargs = {"echo": False}
        if "sqlite" in database_url:
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30,  # Wait up to 30s on locked DB
            }
            # Use StaticPool for single-connection thread safety with SQLite
            from sqlalchemy.pool import StaticPool
            engine_kwargs["poolclass"] = StaticPool

        self.engine = create_engine(database_url, **engine_kwargs)

        # SQLite pragmas for production robustness
        if "sqlite" in database_url:
            from sqlalchemy import event as sa_event

            @sa_event.listens_for(self.engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")      # Concurrent reads during writes
                cursor.execute("PRAGMA busy_timeout=30000")    # 30s retry on lock
                cursor.execute("PRAGMA synchronous=NORMAL")    # Good balance of safety/speed
                cursor.execute("PRAGMA foreign_keys=ON")       # Enforce FK constraints
                cursor.execute("PRAGMA cache_size=-64000")     # 64MB cache
                cursor.close()

        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionFactory()

    # --- Trades ---

    def record_trade(self, trade_data: dict) -> Trade:
        """Insert a new trade record."""
        with self.get_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade

    def update_trade(self, order_id: str, updates: dict):
        """Update a trade (e.g., set PnL on close)."""
        with self.get_session() as session:
            trade = session.query(Trade).filter_by(order_id=order_id).first()
            if trade:
                for k, v in updates.items():
                    setattr(trade, k, v)
                session.commit()

    def get_trades(self, symbol: str = None, strategy: str = None, limit: int = 100) -> list[dict]:
        """Query trades with optional filters. Returns dicts to avoid detached session issues."""
        with self.get_session() as session:
            q = session.query(Trade)
            if symbol:
                q = q.filter_by(symbol=symbol)
            if strategy:
                q = q.filter_by(strategy=strategy)
            trades = q.order_by(Trade.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": t.id, "symbol": t.symbol, "side": t.side,
                    "qty": t.qty, "price": t.price, "order_type": t.order_type,
                    "status": t.status, "order_id": t.order_id, "strategy": t.strategy,
                    "signal_confidence": t.signal_confidence, "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit, "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                    "fees": t.fees, "created_at": str(t.created_at),
                    "filled_at": str(t.filled_at) if t.filled_at else None,
                }
                for t in trades
            ]

    # --- Signals ---

    def log_signal(self, signal_data: dict) -> SignalLog:
        """Log a generated signal."""
        with self.get_session() as session:
            sig = SignalLog(**signal_data)
            session.add(sig)
            session.commit()
            return sig

    # --- Portfolio ---

    def snapshot_portfolio(self, snapshot_data: dict) -> PortfolioSnapshot:
        """Record a portfolio snapshot."""
        with self.get_session() as session:
            snap = PortfolioSnapshot(**snapshot_data)
            session.add(snap)
            session.commit()
            return snap

    def get_portfolio_history(self, days: int = 30) -> list[dict]:
        """Get recent portfolio snapshots as dicts."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.get_session() as session:
            snapshots = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.created_at >= cutoff
            ).order_by(PortfolioSnapshot.created_at).all()
            return [
                {
                    "total_equity": s.total_equity, "cash": s.cash,
                    "positions_value": s.positions_value, "unrealized_pnl": s.unrealized_pnl,
                    "daily_pnl": s.daily_pnl, "open_positions": s.open_positions,
                    "created_at": str(s.created_at),
                }
                for s in snapshots
            ]

    # --- Journal ---

    def get_journal_session(self):
        """Get session for journal operations."""
        return self.get_session()

    # --- Performance Metrics ---

    def get_performance_summary(self, days: int = 30) -> dict:
        """Calculate performance metrics from trade history."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        with self.get_session() as session:
            trades = session.query(Trade).filter(
                Trade.created_at >= cutoff,
                Trade.pnl.isnot(None),
            ).all()

            if not trades:
                return {"total_trades": 0, "message": "No closed trades in period"}

            pnls = [t.pnl for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            return {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) if trades else 0,
                "total_pnl": sum(pnls),
                "avg_win": sum(wins) / len(wins) if wins else 0,
                "avg_loss": sum(losses) / len(losses) if losses else 0,
                "profit_factor": abs(sum(wins) / sum(losses)) if losses else float('inf'),
                "largest_win": max(pnls) if pnls else 0,
                "largest_loss": min(pnls) if pnls else 0,
            }

    # --- Sector Cache ---

    def get_cached_sector(self, symbol: str) -> Optional[str]:
        """Look up a cached sector classification for a symbol."""
        with self.get_session() as session:
            entry = session.query(SectorCache).filter_by(symbol=symbol.upper()).first()
            return entry.sector if entry else None

    def set_cached_sector(self, symbol: str, sector: str, source: str = "manual"):
        """Insert or update a sector classification for a symbol."""
        with self.get_session() as session:
            entry = session.query(SectorCache).filter_by(symbol=symbol.upper()).first()
            if entry:
                entry.sector = sector
                entry.source = source
                entry.updated_at = datetime.utcnow()
            else:
                entry = SectorCache(symbol=symbol.upper(), sector=sector, source=source)
                session.add(entry)
            session.commit()

    def get_all_cached_sectors(self) -> dict[str, str]:
        """Return all cached sector classifications as {symbol: sector}."""
        with self.get_session() as session:
            entries = session.query(SectorCache).all()
            return {e.symbol: e.sector for e in entries}
