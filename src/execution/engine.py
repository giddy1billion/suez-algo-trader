"""
Execution Engine — Orchestrates strategy signals → risk check → order placement.
The central coordinator between strategies, risk manager, and broker.

Integrates:
- EventBus for decoupled event publishing
- TradeManager for trade lifecycle tracking
- ExecutionSimulator for realistic fill modeling (opt-in)
"""

import threading
from datetime import datetime, timezone
from typing import Optional

from src.broker.alpaca_client import AlpacaBroker
from src.risk.manager import RiskManager, RiskLimits
from src.risk.engine import RiskEngine
from src.risk.models import TradeRequest, RiskDecision
from src.strategy.base import BaseStrategy, TradeSignal, Signal
from src.data.store import DatabaseManager
from src.utils.logger import get_logger

# Event types — imported at top for reliability
from src.core.events import (
    SignalGenerated, RiskEvaluated, OrderSubmitted, OrderFilled,
    OrderRejected, TradeOpened, TradeClosed, RiskHalt,
)
from src.core.state_machine import TradeState

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
    3. Executes via broker (with optional execution simulation)
    4. Publishes events to EventBus
    5. Tracks trade lifecycle via TradeManager
    6. Records everything to database
    """

    def __init__(
        self,
        broker: AlpacaBroker,
        risk_manager: RiskManager,
        db: DatabaseManager,
        dry_run: bool = False,
        risk_engine: Optional[RiskEngine] = None,
        event_bus=None,
        trade_manager=None,
        execution_simulator=None,
    ):
        self.broker = broker
        self.risk = risk_manager
        self.db = db
        self.dry_run = dry_run
        self.risk_engine = risk_engine or RiskEngine()
        self._last_cycle_time: Optional[datetime] = None
        self._current_strategy_name: str = "unknown"
        self._cycle_count: int = 0

        # Event-driven components (optional but recommended)
        self._event_bus = event_bus
        self._trade_manager = trade_manager
        self._simulator = execution_simulator  # None = no simulation (direct broker)

    # ──────────────────────────────────────────────────────────────────────
    # Event Publishing Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _publish(self, event) -> None:
        """Publish event to bus if available. Never crashes."""
        if self._event_bus:
            try:
                self._event_bus.publish(event)
            except Exception as e:
                logger.debug("event_bus.publish_error", error=str(e))

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
            self._publish(RiskHalt(reason=reason, level="WARNING", source="engine"))
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
        actionable_count = len([s for s in signals if s.is_actionable])
        logger.info("engine.signals", count=len(signals), actionable=actionable_count)

        # Publish signal events
        for sig in signals:
            if sig.is_actionable:
                self._publish(SignalGenerated(
                    symbol=sig.symbol,
                    signal=sig.signal.name,
                    confidence=sig.confidence,
                    strategy=self._current_strategy_name,
                    price=sig.price,
                    indicators=sig.indicators or {},
                    source="engine",
                ))

        # 4. Process each signal
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        portfolio_value = account['portfolio_value']

        # Update risk manager with current equity
        self.risk.update_equity(portfolio_value)

        for signal in signals:
            if not signal.is_actionable:
                continue

            result = self._process_signal(signal, portfolio_value, positions, data)
            if result:
                results.append(result)

        # 5. Check existing positions for exit signals
        exit_results = self._check_exits(strategy, positions, data)
        results.extend(exit_results)

        # 6. Snapshot portfolio
        self._snapshot_portfolio(account, positions)

        # 7. Periodic cleanup (every 50 cycles, remove terminal trades from memory)
        self._cycle_count += 1
        if self._trade_manager and self._cycle_count % 50 == 0:
            removed = self._trade_manager.remove_terminal()
            if removed > 0:
                logger.info("engine.cleanup_terminal", removed=removed)

        return results

    def _process_signal(self, signal: TradeSignal, portfolio_value: float,
                        positions: list[dict], market_data: dict = None) -> Optional[dict]:
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
            strategy=self._current_strategy_name,
            confidence=signal.confidence,
        )

        # Create trade lifecycle if TradeManager available
        trade_lifecycle = None
        if self._trade_manager:
            trade_lifecycle = self._trade_manager.create_trade(signal.symbol, side)
            if not trade_lifecycle.transition(TradeState.PENDING_RISK, "risk evaluation"):
                logger.warning("engine.invalid_transition", symbol=signal.symbol, state="PENDING_RISK")

        # Get account cash for risk evaluation
        try:
            account = self.broker.get_account()
            cash = account.get('cash', 0.0)
        except Exception:
            cash = portfolio_value * 0.5

        # Build market_data context for risk engine
        risk_market_data = {}
        if market_data and signal.symbol in market_data:
            df = market_data[signal.symbol]
            if len(df) > 0:
                risk_market_data = {
                    "volume": float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0,
                    "adv": float(df['volume'].rolling(20).mean().iloc[-1]) if 'volume' in df.columns else 0,
                    "daily_vol": float(df['close'].pct_change().rolling(20).std().iloc[-1]) if len(df) > 20 else 0,
                }

        # Evaluate through the multi-layer risk engine
        decision: RiskDecision = self.risk_engine.evaluate(
            request=trade_request,
            portfolio_value=portfolio_value,
            cash=cash,
            positions=positions,
            market_data=risk_market_data,
        )

        # Publish risk evaluation event
        self._publish(RiskEvaluated(
            symbol=signal.symbol,
            approved=decision.approved,
            reasons=decision.reasons,
            adjusted_qty=decision.adjusted_qty,
            risk_score=getattr(decision, 'risk_score', 0.0),
            source="risk_engine",
        ))

        if not decision.approved:
            logger.info("engine.trade_rejected", symbol=signal.symbol, reasons=decision.reasons)
            self._log_signal(signal, executed=False)
            if trade_lifecycle:
                trade_lifecycle.transition(TradeState.RISK_REJECTED, "; ".join(decision.reasons))
            return None

        # Risk approved — transition lifecycle
        if trade_lifecycle:
            trade_lifecycle.transition(TradeState.RISK_APPROVED, "all layers passed")

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

        # --- Execution Simulation (opt-in) ---
        sim_result = None
        if self._simulator:
            volume = risk_market_data.get("volume", 1_000_000)
            atr = None
            if market_data and signal.symbol in market_data:
                df = market_data[signal.symbol]
                if len(df) > 14:
                    # Quick ATR estimate
                    tr = (df['high'] - df['low']).rolling(14).mean()
                    atr = float(tr.iloc[-1]) if not tr.empty else None

            sim_result = self._simulator.simulate_execution(
                symbol=signal.symbol,
                side=side,
                qty=final_qty,
                price=signal.price,
                volume=volume,
                atr=atr,
                asset_type="crypto" if "/" in signal.symbol else "equity",
            )

            if not sim_result.get('executed', True):
                # Simulated rejection — publish event and abort
                logger.warning("engine.sim_rejected", symbol=signal.symbol,
                             reason=sim_result.get('rejection_reason'))
                self._publish(OrderRejected(
                    order_id="",
                    reason=f"sim_rejected: {sim_result.get('rejection_reason', 'no_fill')}",
                    source="simulator",
                ))
                if trade_lifecycle:
                    trade_lifecycle.transition(TradeState.CANCELLED, "sim_rejected")
                return None

            # Use simulated average price for logging
            logger.info("engine.sim_fill",
                       symbol=signal.symbol,
                       avg_price=sim_result.get('avg_price'),
                       slippage_bps=sim_result.get('slippage_bps', 0),
                       fees=sim_result.get('fees', 0))

        # Transition lifecycle: SUBMITTED
        if trade_lifecycle:
            trade_lifecycle.transition(TradeState.SUBMITTED, "order submitted to broker")

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

            # Publish OrderSubmitted event
            self._publish(OrderSubmitted(
                symbol=signal.symbol,
                side=side,
                qty=final_qty,
                price=signal.price,
                order_type="bracket" if signal.stop_loss else "market",
                order_id=order.get("id", ""),
                source="engine",
            ))

            # Transition lifecycle: ACCEPTED → FILLED → ACTIVE
            # Use trade lifecycle ID as the canonical trade_id for event correlation
            trade_id = trade_lifecycle.trade_id if trade_lifecycle else order.get("id", "")
            if trade_lifecycle:
                trade_lifecycle.transition(TradeState.ACCEPTED, "broker accepted")
                trade_lifecycle.transition(TradeState.FILLED, f"filled qty={final_qty}")
                trade_lifecycle.transition(TradeState.ACTIVE, "position active")
                # Store order_id in lifecycle metadata for exit correlation
                trade_lifecycle.metadata['order_id'] = order.get("id", "")
                trade_lifecycle.metadata['symbol'] = signal.symbol

            # Publish OrderFilled
            fill_price = sim_result['avg_price'] if sim_result else signal.price
            fill_fees = sim_result['fees'] if sim_result else 0.0
            self._publish(OrderFilled(
                order_id=order.get("id", ""),
                fill_price=fill_price,
                fill_qty=final_qty,
                fees=fill_fees,
                source="engine",
            ))

            # Publish TradeOpened with consistent trade_id
            self._publish(TradeOpened(
                trade_id=trade_id,
                symbol=signal.symbol,
                side=side,
                entry_price=fill_price,
                qty=final_qty,
                stop_loss=signal.stop_loss or 0.0,
                take_profit=signal.take_profit or 0.0,
                source="engine",
            ))

            # Record trade to database
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
                "sim_slippage_bps": sim_result.get("slippage_bps") if sim_result else None,
                "sim_fees": fill_fees if sim_result else None,
            })

            # Journal trade entry for analysis
            journal = _get_journal(self.db)
            if journal:
                try:
                    journal.log_entry({
                        "trade_id": order.get("id"),
                        "symbol": signal.symbol,
                        "side": side,
                        "entry_price": fill_price,
                        "qty": final_qty,
                        "strategy_name": self._current_strategy_name,
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
            self._publish(OrderRejected(
                order_id="",
                reason=str(e),
                source="engine",
            ))
            if trade_lifecycle:
                trade_lifecycle.transition(TradeState.ERROR, f"order_failed: {str(e)[:80]}")
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

                        # Compute consistent PnL%
                        entry_price = pos.get('avg_entry_price', pos.get('cost_basis', 0))
                        pnl_pct = 0.0
                        if entry_price and abs(entry_price) > 1e-8:
                            pnl_pct = (current_price - entry_price) / entry_price * 100

                        # Find and transition trade lifecycle to CLOSED
                        trade_id = pos.get('asset_id', '')
                        if self._trade_manager:
                            active_trades = self._trade_manager.get_trades_by_symbol(symbol)
                            for t in active_trades:
                                if not t.is_terminal:
                                    t.transition(TradeState.CLOSING, "strategy_exit")
                                    t.transition(TradeState.CLOSED, f"pnl={pnl:.2f}")
                                    trade_id = t.trade_id
                                    break

                        # Publish TradeClosed event
                        self._publish(TradeClosed(
                            trade_id=trade_id,
                            symbol=symbol,
                            exit_price=current_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=exit_signal.reason[:50] if exit_signal.reason else "strategy_exit",
                            source="engine",
                        ))

                        # Journal the exit
                        journal = _get_journal(self.db)
                        if journal:
                            try:
                                entries = journal.get_journal(symbol=symbol, limit=20)
                                open_entries = [e for e in entries if e.get("exit_price") is None]
                                if open_entries:
                                    target = open_entries[-1]
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
                "strategy": self._current_strategy_name,
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

        # Get positions BEFORE closing to publish per-position events
        try:
            positions = self.broker.get_positions()
        except Exception:
            positions = []

        # Close all and publish individual TradeClosed events
        self.broker.close_all_positions()

        for pos in positions:
            symbol = pos.get('symbol', '?')
            pnl = pos.get('unrealized_pl', 0)
            current_price = pos.get('current_price', 0)
            entry_price = pos.get('avg_entry_price', 0)
            pnl_pct = 0.0
            if entry_price and abs(entry_price) > 1e-8:
                pnl_pct = (current_price - entry_price) / entry_price * 100

            # Transition trade lifecycle
            trade_id = pos.get('asset_id', '')
            if self._trade_manager:
                active_trades = self._trade_manager.get_trades_by_symbol(symbol)
                for t in active_trades:
                    if not t.is_terminal:
                        t.transition(TradeState.CLOSING, "emergency_liquidation")
                        t.transition(TradeState.CLOSED, "emergency_liquidation")
                        trade_id = t.trade_id
                        break

            self._publish(TradeClosed(
                trade_id=trade_id,
                symbol=symbol,
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason="emergency_liquidation",
                source="engine",
            ))

        self.risk.daily_stats.is_halted = True
        self.risk.daily_stats.halt_reason = "Emergency liquidation triggered"

        self._publish(RiskHalt(
            reason="Emergency liquidation triggered",
            level="CRITICAL",
            source="engine",
        ))

        # Cleanup terminal trades from memory
        if self._trade_manager:
            removed = self._trade_manager.remove_terminal()
            logger.info("engine.cleanup_terminal_trades", removed=removed)
