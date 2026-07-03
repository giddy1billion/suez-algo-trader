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


# ──────────────────────────────────────────────────────────────────────────
# Database Manager
# ──────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """Manages database connections and provides CRUD operations."""

    def __init__(self, database_url: str = "sqlite:///data_cache/trading.db"):
        self.engine = create_engine(database_url, echo=False)
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

    def get_trades(self, symbol: str = None, strategy: str = None, limit: int = 100) -> list[Trade]:
        """Query trades with optional filters."""
        with self.get_session() as session:
            q = session.query(Trade)
            if symbol:
                q = q.filter_by(symbol=symbol)
            if strategy:
                q = q.filter_by(strategy=strategy)
            return q.order_by(Trade.created_at.desc()).limit(limit).all()

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

    def get_portfolio_history(self, days: int = 30) -> list[PortfolioSnapshot]:
        """Get recent portfolio snapshots."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.get_session() as session:
            return session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.created_at >= cutoff
            ).order_by(PortfolioSnapshot.created_at).all()

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
