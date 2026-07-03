"""
Execution Engine — Orchestrates strategy signals → risk check → order placement.
The central coordinator between strategies, risk manager, and broker.
"""

import threading
import time
from datetime import datetime
from typing import Optional

from src.broker.alpaca_client import AlpacaBroker
from src.risk.manager import RiskManager, RiskLimits
from src.risk.engine import RiskEngine
from src.risk.models import TradeRequest, RiskDecision
from src.strategy.base import BaseStrategy, TradeSignal, Signal
from src.data.store import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialized trade journal (double-checked locking)
_journal = None
_journal_lock = threading.Lock()


def _get_journal(db: DatabaseManager):
    """Lazy-load the trade journal to avoid circular imports. Thread-safe."""
    global _journal
    if _journal is None:
        with _journal_lock:
            if _journal is None:
                try:
                    from src.data.journal import TradeJournal
                    _journal = TradeJournal(db)
                except Exception as e:
                    logger.debug("journal.init_skipped", error=str(e))
    return _journal


class ExecutionEngine:
    """
    Core trading engine:
    1. Receives signals from strategies
    2. Validates through risk manager
    3. Executes via broker
    4. Records everything to database
    """

    def __init__(
        self,
        broker: AlpacaBroker,
        risk_manager: RiskManager,
        db: DatabaseManager,
        dry_run: bool = False,
        risk_engine: Optional[RiskEngine] = None,
    ):
        self.broker = broker
        self.risk = risk_manager
        self.db = db
        self.dry_run = dry_run
        self.risk_engine = risk_engine or RiskEngine()
        self._last_cycle_time: Optional[datetime] = None
        self._current_strategy_name: str = "unknown"

    # ──────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────

    def run_cycle(self, strategy: BaseStrategy) -> list[dict]:
        """
        Execute one complete trading cycle:
        1. Fetch data for all symbols
        2. Generate signals
        3. Filter through risk manager
        4. Execute orders
        5. Log everything
        """
        self._last_cycle_time = datetime.now()
        self._current_strategy_name = strategy.name
        results = []

        # 1. Check if trading is allowed
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            logger.warning("engine.halted", reason=reason)
            return []

        # 2. Fetch market data
        data = {}
        for symbol in strategy.symbols:
            try:
                df = self.broker.get_bars_df(symbol, strategy.timeframe, strategy.lookback)
                if df is not None and len(df) >= 50:
                    data[symbol] = df
            except Exception as e:
                logger.error("engine.data_error", symbol=symbol, error=str(e))

        if not data:
            logger.info("engine.no_data")
            return []

        # 3. Generate signals
        signals = strategy.generate_signals(data)
        logger.info("engine.signals", count=len(signals),
                   actionable=len([s for s in signals if s.is_actionable]))

        # 4. Process each signal
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        portfolio_value = account['portfolio_value']

        # Update risk manager with current equity
        self.risk.update_equity(portfolio_value)

        for signal in signals:
            if not signal.is_actionable:
                continue

            result = self._process_signal(signal, portfolio_value, positions)
            if result:
                results.append(result)

        # 5. Check existing positions for exit signals
        exit_results = self._check_exits(strategy, positions, data)
        results.extend(exit_results)

        # 6. Snapshot portfolio
        self._snapshot_portfolio(account, positions)

        return results

    def _process_signal(self, signal: TradeSignal, portfolio_value: float, positions: list[dict]) -> Optional[dict]:
        """Process a single trade signal through risk engine and execution."""

        # Determine side
        if signal.signal in (Signal.BUY, Signal.STRONG_BUY):
            side = "buy"
        elif signal.signal in (Signal.SELL, Signal.STRONG_SELL):
            side = "sell"
        else:
            return None

        # Check if we already have a position in this symbol
        existing = next((p for p in positions if p['symbol'] == signal.symbol), None)
        if existing and side == "buy" and existing.get('side') == 'long':
            logger.debug("engine.skip_existing_long", symbol=signal.symbol)
            return None
        if existing and side == "sell" and existing.get('side') == 'short':
            logger.debug("engine.skip_existing_short", symbol=signal.symbol)
            return None

        # Calculate position size
        if signal.stop_loss:
            qty = self.risk.calculate_position_size(
                price=signal.price,
                stop_loss_price=signal.stop_loss,
                portfolio_value=portfolio_value,
            )
        else:
            # Fallback: use max single stock allocation
            max_value = portfolio_value * self.risk.limits.max_single_stock_pct
            qty = max_value / signal.price

        if qty <= 0:
            return None

        # Build TradeRequest for the new risk engine
        trade_request = TradeRequest(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=getattr(self, '_current_strategy_name', 'unknown'),
            confidence=signal.confidence,
        )

        # Get account cash for risk evaluation
        try:
            account = self.broker.get_account()
            cash = account.get('cash', 0.0)
        except Exception:
            cash = portfolio_value * 0.5  # Fallback estimate

        # Evaluate through the multi-layer risk engine
        decision: RiskDecision = self.risk_engine.evaluate(
            request=trade_request,
            portfolio_value=portfolio_value,
            cash=cash,
            positions=positions,
        )

        if not decision.approved:
            logger.info("engine.trade_rejected", symbol=signal.symbol, reasons=decision.reasons)
            self._log_signal(signal, executed=False)
            return None

        final_qty = decision.adjusted_qty

        # Log signal
        self._log_signal(signal, executed=True)

        # Execute (or dry run)
        if self.dry_run:
            logger.info("engine.dry_run", symbol=signal.symbol, side=side, qty=final_qty,
                       price=signal.price, confidence=signal.confidence)
            return {
                "symbol": signal.symbol,
                "side": side,
                "qty": final_qty,
                "price": signal.price,
                "dry_run": True,
            }

        # Place the order
        try:
            if signal.stop_loss and signal.take_profit:
                order = self.broker.bracket_order(
                    symbol=signal.symbol,
                    qty=final_qty,
                    side=side,
                    stop_loss_price=signal.stop_loss,
                    take_profit_price=signal.take_profit,
                )
            else:
                order = self.broker.market_order(
                    symbol=signal.symbol,
                    qty=final_qty,
                    side=side,
                )

            # Record trade
            self.db.record_trade({
                "symbol": signal.symbol,
                "side": side,
                "qty": final_qty,
                "price": signal.price,
                "order_type": "bracket" if signal.stop_loss else "market",
                "status": order.get("status", "submitted"),
                "order_id": order.get("id"),
                "strategy": signal.reason[:50],
                "signal_confidence": signal.confidence,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
            })

            # Journal trade entry for analysis
            journal = _get_journal(self.db)
            if journal:
                try:
                    journal.log_entry({
                        "trade_id": order.get("id"),
                        "symbol": signal.symbol,
                        "side": side,
                        "entry_price": signal.price,
                        "qty": final_qty,
                        "strategy_name": getattr(self, '_current_strategy_name', 'unknown'),
                        "model_version": getattr(signal, 'model_version', None),
                        "prediction": side,
                        "confidence": signal.confidence,
                        "features_snapshot": signal.indicators,
                    })
                except Exception as e:
                    logger.debug("journal.log_entry_error", error=str(e))

            logger.info("engine.order_placed",
                       symbol=signal.symbol, side=side, qty=final_qty,
                       confidence=f"{signal.confidence:.1%}")

            return order

        except Exception as e:
            logger.error("engine.order_failed", symbol=signal.symbol, error=str(e))
            return None

    def _check_exits(self, strategy: BaseStrategy, positions: list[dict], data: dict) -> list[dict]:
        """Check if any open positions should be closed based on strategy logic."""
        results = []
        for pos in positions:
            symbol = pos['symbol']
            current_price = pos.get('current_price', 0)

            exit_signal = strategy.should_exit(symbol, pos, current_price)
            if exit_signal and exit_signal.signal in (Signal.SELL, Signal.STRONG_SELL):
                if not self.dry_run:
                    try:
                        self.broker.close_position(symbol)
                        pnl = pos.get('unrealized_pl', 0)
                        self.risk.record_trade({"symbol": symbol, "pnl": pnl})
                        results.append({"action": "exit", "symbol": symbol, "pnl": pnl})

                        # Journal the exit
                        journal = _get_journal(self.db)
                        if journal:
                            try:
                                # Find the oldest OPEN journal entry for this symbol
                                entries = journal.get_journal(symbol=symbol, limit=20)
                                open_entries = [e for e in entries if e.get("exit_price") is None]
                                if open_entries:
                                    target = open_entries[-1]  # oldest open entry (list is newest-first)
                                    entry_price = target.get("entry_price", 0)
                                    pnl_pct = (current_price - entry_price) / entry_price if entry_price else 0
                                    journal.log_exit(target["id"], {
                                        "exit_price": current_price,
                                        "exit_reason": exit_signal.reason[:50] if exit_signal.reason else "strategy_exit",
                                        "pnl": pnl,
                                        "pnl_pct": pnl_pct,
                                    })
                            except Exception as e:
                                logger.debug("journal.log_exit_error", error=str(e))
                    except Exception as e:
                        logger.error("engine.exit_failed", symbol=symbol, error=str(e))
                else:
                    logger.info("engine.dry_run_exit", symbol=symbol)

        return results

    def _log_signal(self, signal: TradeSignal, executed: bool):
        """Log signal to database."""
        import json
        try:
            self.db.log_signal({
                "symbol": signal.symbol,
                "strategy": getattr(self, '_current_strategy_name', 'unknown'),
                "signal": signal.signal.name,
                "confidence": signal.confidence,
                "price_at_signal": signal.price,
                "indicators": json.dumps(signal.indicators),
                "was_executed": executed,
            })
        except Exception as e:
            logger.error("engine.signal_log_error", error=str(e))

    def _snapshot_portfolio(self, account: dict, positions: list[dict]):
        """Take a portfolio snapshot."""
        try:
            self.db.snapshot_portfolio({
                "total_equity": account['equity'],
                "cash": account['cash'],
                "positions_value": account['long_market_value'],
                "unrealized_pnl": sum(p.get('unrealized_pl', 0) for p in positions),
                "open_positions": len(positions),
            })
        except Exception as e:
            logger.error("engine.snapshot_error", error=str(e))

    # ──────────────────────────────────────────────────────────────────────
    # Emergency Controls
    # ──────────────────────────────────────────────────────────────────────

    def emergency_liquidate(self):
        """PANIC: Close all positions and cancel all orders immediately."""
        logger.warning("engine.EMERGENCY_LIQUIDATE")
        self.broker.cancel_all_orders()
        self.broker.close_all_positions()
        self.risk.daily_stats.is_halted = True
        self.risk.daily_stats.halt_reason = "Emergency liquidation triggered"
