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
from src.strategy.base import BaseStrategy, TradeSignal, LegacyTradeSignal, Signal, Side
from src.strategy.signal_adapter import adapt_signal, is_legacy_signal, is_actionable
from src.data.store import DatabaseManager
from src.intelligence.orchestrator import AdaptiveIntelligenceOrchestrator
from src.intelligence.confidence.gate import ConfidenceGate, ConfidenceGateConfig, SignalContext
from src.intelligence.confidence.models import ConfidenceScore, SignalIntegrity
from src.intelligence.confidence.orchestrator import DecisionOrchestrator
from src.intelligence.confidence.decision_contract import DecisionContract, Decision
from src.utils.logger import get_logger
from src.core.runtime_state import RuntimeState

# Event types — imported at top for reliability
from src.core.events import (
    SignalGenerated, DecisionContractCreated, RiskEvaluated, OrderSubmitted, OrderFilled,
    OrderRejected, TradeOpened, TradeClosed, RiskHalt,
)
from src.core.state_machine import TradeState
from src.strategy.signal_bridge import (
    SignalPackageBuilder,
    SignalBridgeConfig,
    ActiveSignalMonitor,
)
from src.strategy.signal_package import SignalValidationGate, SignalStatus

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
        intelligence_orchestrator: Optional[AdaptiveIntelligenceOrchestrator] = None,
        min_signal_confidence: float = 0.55,
        circuit_breaker=None,
        runtime_state: Optional[RuntimeState] = None,
        signal_gate: Optional[SignalValidationGate] = None,
        signal_bridge_config: Optional[SignalBridgeConfig] = None,
        confidence_gate: Optional[ConfidenceGate] = None,
        decision_orchestrator: Optional[DecisionOrchestrator] = None,
        contract_store=None,
    ):
        self.broker = broker
        self.risk = risk_manager
        self.db = db
        self.dry_run = dry_run
        self.risk_engine = risk_engine or RiskEngine()
        self.min_signal_confidence = min_signal_confidence
        self._last_cycle_time: Optional[datetime] = None
        self._current_strategy_name: str = "unknown"
        self._current_capital_weight: float = 1.0
        self._cycle_count: int = 0

        # Event-driven components (optional but recommended)
        self._event_bus = event_bus
        self._trade_manager = trade_manager
        self._simulator = execution_simulator  # None = no simulation (direct broker)
        self.intelligence_orchestrator = intelligence_orchestrator
        self._circuit_breaker = circuit_breaker
        self._confidence_gate = confidence_gate
        
        # Decision Contract infrastructure — the new central governance path.
        # When decision_orchestrator is provided, every signal produces an
        # immutable DecisionContract that flows through the entire pipeline.
        self._decision_orchestrator = decision_orchestrator
        self._contract_store = contract_store  # ContractStore for DuckDB persistence
        
        # Runtime state — allows pause/resume to suppress signals
        self._runtime_state = runtime_state or RuntimeState()

        # Signal package integration — professional-grade signal validation
        self._signal_gate = signal_gate
        self._signal_builder = SignalPackageBuilder(signal_bridge_config)
        self._signal_monitor = ActiveSignalMonitor()

        # Trade context tracking for closed-loop feedback
        # Maps trade_id → {signal_package_id, model_version, strategy, entry_time, side, entry_price}
        self._trade_context: dict[str, dict] = {}
        self._trade_context_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────
    # Signal Monitor (public API for main loop)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def signal_monitor(self) -> ActiveSignalMonitor:
        """Access the active signal monitor for expiry/decay checks."""
        return self._signal_monitor

    def check_signal_expiry(self) -> int:
        """Check active signals for expiry/decay. Returns count invalidated."""
        invalidated = self._signal_monitor.check_all()
        for pkg in invalidated:
            self._publish(OrderRejected(
                symbol=pkg.symbol,
                side=pkg.side,
                qty=0,
                reason=f"signal_expired: {pkg.signal_id}",
                source="signal_monitor",
            ))
        return len(invalidated)

    # ──────────────────────────────────────────────────────────────────────
    # Event Publishing Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _publish(self, event) -> None:
        """Publish event to bus if available. Never crashes."""
        if self._event_bus:
            try:
                self._event_bus.publish(event)
            except Exception as e:
                logger.warning("event_bus.publish_error", event_type=type(event).__name__, error=str(e))

    def _has_trade_stream(self) -> bool:
        """Check if broker has an active trade stream for fill confirmations."""
        return getattr(self.broker, '_trade_stream_thread', None) is not None and \
               getattr(self.broker, '_shutdown_flag', True) is False

    # ──────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────

    def run_cycle(self, strategy: BaseStrategy, capital_weight: float = 1.0) -> list[dict]:
        """
        Execute one complete trading cycle:
        1. Fetch data for all symbols
        2. Generate signals
        3. Filter through risk manager
        4. Execute orders
        5. Log everything

        Args:
            strategy: The strategy to run.
            capital_weight: Fraction of portfolio allocated to this strategy (0.0-1.0+).
                          Used for multi-strategy capital allocation.
        """
        self._last_cycle_time = datetime.now()
        self._current_strategy_name = strategy.name
        self._current_capital_weight = max(0.01, min(capital_weight, 5.0))  # Clamp to sane range
        results = []

        # 0. Check circuit breaker
        if self._circuit_breaker and not self._circuit_breaker.is_trading_allowed():
            cb_reasons = self._circuit_breaker.active_reasons
            logger.warning("engine.circuit_breaker_active", reasons=cb_reasons)
            return []

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

        # 3. Generate signals and adapt to clean format
        raw_signals = strategy.generate_signals(data)
        
        # Adapt all signals to new TradeSignal format (frozen, minimal)
        adapted_signals: list[TradeSignal] = []
        for raw_sig in raw_signals:
            if is_actionable(raw_sig):
                adapted = adapt_signal(raw_sig, strategy)
                adapted_signals.append(adapted)
        
        logger.info("engine.signals", count=len(raw_signals), actionable=len(adapted_signals))

        # Publish signal events (respecting pause state and operating mode)
        for sig in adapted_signals:
            # Skip signal publishing if bot is paused or mode prohibits entries
            if self._runtime_state.is_paused():
                logger.info("engine.signal_suppressed_paused", symbol=sig.symbol, signal=sig.side.value)
                continue
            if not self._runtime_state.can_open_positions:
                logger.info(
                    "engine.signal_suppressed_mode",
                    symbol=sig.symbol,
                    mode=self._runtime_state.operating_mode.value,
                )
                continue
            
            self._publish(SignalGenerated(
                signal_id=sig.signal_id,
                strategy=sig.strategy_id,
                strategy_version=sig.strategy_version,
                symbol=sig.symbol,
                timeframe=sig.timeframe,
                signal=sig.side.value,
                side=sig.side.value,
                signal_strength=sig.signal_strength,
                expected_direction=sig.expected_direction,
                confidence=sig.signal_strength,  # backward compat
                price=sig.features.get("observed_price", 0.0),
                reason=sig.reason,
                tags=sig.tags,
                indicators=sig.indicators,
                features=sig.features,
                source="engine",
            ))

        # 4. Process each signal through the decision pipeline
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        portfolio_value = account['portfolio_value']

        # Apply capital_weight for multi-strategy allocation
        effective_portfolio = portfolio_value * self._current_capital_weight

        # Update risk manager with current equity
        self.risk.update_equity(portfolio_value)

        # Track alternatives within this cycle (signals NOT taken become
        # alternatives for those that ARE taken — institutional audit trail)
        self._cycle_alternatives: list[dict] = []

        for signal in adapted_signals:
            result = self._process_signal(signal, effective_portfolio, positions, data)
            if result:
                results.append(result)
            else:
                # Signal was rejected — record as alternative for future signals this cycle
                self._cycle_alternatives.append({
                    "symbol": signal.symbol,
                    "direction": signal.side.value,
                    "raw_signal_strength": signal.signal_strength,
                    "rejection_reason": "Rejected by decision/risk pipeline",
                    "rejected_by_stage": "execution_engine",
                    "signal_id": signal.signal_id,
                })

        # Clear cycle alternatives after processing
        self._cycle_alternatives = []

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
        """
        Process a single trade signal through the clean decision pipeline.

        Pipeline:
            TradeSignal (proposal)
                → Signal strength gate
                → Position check
                → Intelligence layer (optional)
                → Signal package gate (optional)
                → DecisionOrchestrator → DecisionContract
                → Publish DecisionContractCreated event
                → Build TradeRequest(signal, contract)
                → Risk Engine evaluation
                → Broker execution
        """

        # Hard signal strength gate — reject signals below minimum threshold.
        if signal.signal_strength < self.min_signal_confidence:
            logger.info(
                "engine.low_strength_rejected",
                symbol=signal.symbol,
                signal_strength=round(signal.signal_strength, 3),
                threshold=self.min_signal_confidence,
                strategy=signal.strategy_id,
                signal_id=signal.signal_id,
            )
            self._log_signal_v2(signal, executed=False)
            return None

        # Determine side for broker (lowercase)
        side = signal.side.value.lower()

        # Check if we already have a position in this symbol
        existing = next((p for p in positions if p['symbol'] == signal.symbol), None)
        if existing and side == "buy" and existing.get('side') == 'long':
            logger.debug("engine.skip_existing_long", symbol=signal.symbol)
            return None
        if existing and side == "sell" and existing.get('side') == 'short':
            logger.debug("engine.skip_existing_short", symbol=signal.symbol)
            return None

        # ── Intelligence Layer (optional, adaptive) ─────────────────────────
        intelligence_decision = None
        if self.intelligence_orchestrator:
            symbol_df = market_data.get(signal.symbol) if market_data else None
            portfolio_context = self._build_portfolio_context(positions, portfolio_value, signal.symbol)
            intelligence_decision = self.intelligence_orchestrator.evaluate_signal(
                strategy_name=signal.strategy_id,
                signal_confidence=signal.signal_strength,
                indicators=signal.indicators,
                df=symbol_df,
                portfolio_context=portfolio_context,
            )
            if not intelligence_decision.accepted:
                logger.info(
                    "engine.intelligence_rejected",
                    symbol=signal.symbol,
                    score=round(intelligence_decision.final_score, 2),
                    reason=intelligence_decision.routing.reason,
                    signal_id=signal.signal_id,
                )
                self._log_signal_v2(signal, executed=False)
                return None

        # ── Signal Package Gate (optional) ──────────────────────────────────
        signal_package = None
        if self._signal_gate:
            symbol_df = market_data.get(signal.symbol) if market_data else None
            # Build legacy-compat signal for signal_bridge (which expects old format)
            legacy_compat = LegacyTradeSignal(
                symbol=signal.symbol,
                signal=Signal.BUY if signal.side == Side.BUY else Signal.SELL,
                confidence=signal.signal_strength,
                price=signal.features.get("observed_price", 0.0),
                stop_loss=signal.features.get("strategy_proposed_stop_loss"),
                take_profit=signal.features.get("strategy_proposed_take_profit"),
                reason=signal.reason,
                indicators=dict(signal.indicators),
            )
            signal_package = self._signal_builder.build(
                signal=legacy_compat,
                strategy_name=signal.strategy_id,
                market_data=symbol_df,
                intelligence_decision=intelligence_decision,
                portfolio_value=portfolio_value,
                position_size_pct=0.0,
            )

            approved, gate_errors = self._signal_gate.evaluate(signal_package)
            if not approved:
                logger.info(
                    "engine.signal_gate_rejected",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    errors=gate_errors,
                )
                self._log_signal_v2(signal, executed=False)
                return None

            self._signal_monitor.register(signal_package)
            logger.info(
                "engine.signal_package_approved",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                signal_strength=signal.signal_strength,
                risk_reward=signal_package.expected_risk_reward,
                regime=signal_package.market_regime.value,
            )

        # ── Position Sizing (preliminary, may be overridden by contract) ────
        observed_price = signal.features.get("observed_price", 0.0)
        strategy_sl = signal.features.get("strategy_proposed_stop_loss")
        
        if strategy_sl and observed_price > 0:
            qty = self.risk.calculate_position_size(
                price=observed_price,
                stop_loss_price=strategy_sl,
                portfolio_value=portfolio_value,
            )
        else:
            max_value = portfolio_value * self.risk.limits.max_single_stock_pct
            qty = max_value / observed_price if observed_price > 0 else 0

        if qty <= 0:
            return None

        if intelligence_decision:
            qty *= intelligence_decision.qty_multiplier
            if qty <= 0:
                logger.info("engine.intelligence_zero_qty", symbol=signal.symbol)
                self._log_signal_v2(signal, executed=False)
                return None

        # Apply operating mode size reduction
        mode_multiplier = self._runtime_state.position_size_multiplier
        if mode_multiplier < 1.0:
            qty *= mode_multiplier
            if qty <= 0:
                logger.info(
                    "engine.mode_zero_qty",
                    symbol=signal.symbol,
                    mode=self._runtime_state.operating_mode.value,
                )
                return None

        # ── Decision Contract (authoritative decision) ──────────────────────
        confidence_score = None
        decision_contract = None
        effective_confidence = intelligence_decision.adjusted_confidence if intelligence_decision else signal.signal_strength

        if self._decision_orchestrator:
            signal_ctx = SignalContext(
                symbol=signal.symbol,
                strategy=signal.strategy_id,
                raw_confidence=effective_confidence,
                signal_integrity=SignalIntegrity.REAL,
                signal_generated_at=signal.timestamp,
                bars_available=len(market_data.get(signal.symbol, [])) if market_data else 0,
                spread_available=True,
                model_version=signal.strategy_version,
                current_trend=getattr(intelligence_decision, 'market_trend', '') if intelligence_decision else '',
                current_volatility=getattr(intelligence_decision, 'market_volatility', '') if intelligence_decision else '',
                current_stress=getattr(intelligence_decision, 'market_stress', '') if intelligence_decision else '',
                fingerprint_confidence=getattr(intelligence_decision, 'fingerprint_confidence', 1.0) if intelligence_decision else 1.0,
            )
            decision_contract = self._decision_orchestrator.evaluate(
                context=signal_ctx,
                provenance_kwargs={
                    "model_version": signal.strategy_version,
                    "signal_id": signal.signal_id,
                },
                position_pct=qty / portfolio_value * 100.0 if portfolio_value > 0 else 0.0,
                kelly_fraction=0.0,
                risk_grade="",
                strategy_name=signal.strategy_id,
                signal_type=signal.side.value,
                alternatives=getattr(self, '_cycle_alternatives', None),
            )
            effective_confidence = decision_contract.final_confidence

            # Store contract for audit trail
            if self._contract_store:
                try:
                    self._contract_store.store(decision_contract)
                except Exception as e:
                    logger.warning("engine.contract_store_error", error=str(e))

            # ── Publish DecisionContractCreated event ──
            self._publish(DecisionContractCreated(
                contract_id=decision_contract.contract_id,
                signal_id=signal.signal_id,
                decision=decision_contract.decision.value,
                final_confidence=decision_contract.final_confidence,
                symbol=signal.symbol,
                side=signal.side.value,
                recommended_position_pct=decision_contract.recommended_position_pct,
                recommended_stop_loss=decision_contract.recommended_stop_loss,
                recommended_take_profit=decision_contract.recommended_take_profit,
                risk_grade=decision_contract.risk_grade,
                stage_scores=decision_contract.stage_scores,
                vetoed=decision_contract.vetoed,
                veto_reason=decision_contract.veto_reason,
                expires_at=decision_contract.valid_until.isoformat(),
                source="decision_orchestrator",
            ))

            # If contract says REJECT or is vetoed, stop immediately
            if not decision_contract.is_executable and decision_contract.decision == Decision.REJECT:
                logger.info(
                    "engine.contract_rejected",
                    contract_id=decision_contract.contract_id,
                    symbol=signal.symbol,
                    decision=decision_contract.decision.value,
                    reason=decision_contract.recommendation,
                    vetoed=decision_contract.vetoed,
                    signal_id=signal.signal_id,
                )
                self._record_contract_rejection(decision_contract, signal.symbol, "contract_rejected")
                self._log_signal_v2(signal, executed=False)
                return None

            # If contract says REDUCE, scale down position
            if decision_contract.decision == Decision.REDUCE:
                if decision_contract.recommended_position_pct > 0:
                    reduced_value = portfolio_value * (decision_contract.recommended_position_pct / 100.0)
                    qty = min(qty, reduced_value / observed_price) if observed_price > 0 else qty * 0.5
                else:
                    qty *= 0.5
                logger.info(
                    "engine.contract_reduced",
                    contract_id=decision_contract.contract_id,
                    symbol=signal.symbol,
                    new_qty=qty,
                )

        elif self._confidence_gate:
            # Fallback: Legacy confidence gate path (produces ConfidenceScore)
            signal_ctx = SignalContext(
                symbol=signal.symbol,
                strategy=signal.strategy_id,
                raw_confidence=effective_confidence,
                signal_integrity=SignalIntegrity.REAL,
                signal_generated_at=signal.timestamp,
                bars_available=len(market_data.get(signal.symbol, [])) if market_data else 0,
                spread_available=True,
                model_version=signal.strategy_version,
                current_trend=getattr(intelligence_decision, 'market_trend', '') if intelligence_decision else '',
                current_volatility=getattr(intelligence_decision, 'market_volatility', '') if intelligence_decision else '',
                current_stress=getattr(intelligence_decision, 'market_stress', '') if intelligence_decision else '',
                fingerprint_confidence=getattr(intelligence_decision, 'fingerprint_confidence', 1.0) if intelligence_decision else 1.0,
            )
            confidence_score = self._confidence_gate.evaluate(signal_ctx)
            effective_confidence = confidence_score.value

        # ── Build TradeRequest (signal + contract) ──────────────────────────
        # Use contract's recommended SL/TP if available, else strategy hints
        final_stop_loss = None
        final_take_profit = None
        if decision_contract:
            if decision_contract.recommended_stop_loss > 0:
                final_stop_loss = decision_contract.recommended_stop_loss
            if decision_contract.recommended_take_profit > 0:
                final_take_profit = decision_contract.recommended_take_profit
        if final_stop_loss is None:
            final_stop_loss = signal.features.get("strategy_proposed_stop_loss")
        if final_take_profit is None:
            final_take_profit = signal.features.get("strategy_proposed_take_profit")

        trade_request = TradeRequest(
            symbol=signal.symbol,
            side=side,
            qty=qty,
            price=observed_price,
            stop_loss=final_stop_loss,
            take_profit=final_take_profit,
            strategy=signal.strategy_id,
            confidence=effective_confidence,
            confidence_score=confidence_score,
            decision_contract=decision_contract,
            trade_signal=signal,
        )

        # Create trade lifecycle if TradeManager available
        trade_lifecycle = None
        if self._trade_manager:
            trade_lifecycle = self._trade_manager.create_trade(signal.symbol, side)
            if not trade_lifecycle.transition(TradeState.PENDING_RISK, "risk evaluation"):
                logger.warning("engine.invalid_transition", symbol=signal.symbol, state="PENDING_RISK")
                return None  # Abort: state machine inconsistency

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

        # Evaluate through the multi-layer risk engine (fail-closed: reject on exception)
        try:
            decision: RiskDecision = self.risk_engine.evaluate(
                request=trade_request,
                portfolio_value=portfolio_value,
                cash=cash,
                positions=positions,
                market_data=risk_market_data,
            )
        except Exception as e:
            logger.error("engine.risk_evaluation_exception", symbol=signal.symbol, error=str(e))
            self._log_signal_v2(signal, executed=False)
            if trade_lifecycle:
                trade_lifecycle.transition(TradeState.RISK_REJECTED, f"risk_exception: {str(e)[:80]}")
            return None

        # Publish risk evaluation event
        self._publish(RiskEvaluated(
            symbol=signal.symbol,
            approved=decision.approved,
            reasons=decision.reasons,
            adjusted_qty=decision.adjusted_qty,
            risk_score=getattr(decision, 'risk_score', 0.0),
            contract_id=decision_contract.contract_id if decision_contract else "",
            source="risk_engine",
        ))

        if not decision.approved:
            logger.info("engine.trade_rejected", symbol=signal.symbol, reasons=decision.reasons)
            self._log_signal_v2(signal, executed=False)
            self._record_contract_rejection(decision_contract, signal.symbol, "risk_rejected")
            if trade_lifecycle:
                trade_lifecycle.transition(TradeState.RISK_REJECTED, "; ".join(decision.reasons))
            return None

        # Risk approved — transition lifecycle
        if trade_lifecycle:
            trade_lifecycle.transition(TradeState.RISK_APPROVED, "all layers passed")

        final_qty = decision.adjusted_qty

        # Log signal
        self._log_signal_v2(signal, executed=True, intelligence_decision=intelligence_decision)

        # Execute (or dry run)
        if self.dry_run:
            logger.info("engine.dry_run", symbol=signal.symbol, side=side, qty=final_qty,
                       price=observed_price, signal_strength=signal.signal_strength)
            self._record_contract_rejection(decision_contract, signal.symbol, "dry_run")
            return {
                "symbol": signal.symbol,
                "side": side,
                "qty": final_qty,
                "price": observed_price,
                "signal_id": signal.signal_id,
                "contract_id": decision_contract.contract_id if decision_contract else None,
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
                    tr = (df['high'] - df['low']).rolling(14).mean()
                    atr = float(tr.iloc[-1]) if not tr.empty else None

            sim_result = self._simulator.simulate_execution(
                symbol=signal.symbol,
                side=side,
                qty=final_qty,
                price=observed_price,
                volume=volume,
                atr=atr,
                asset_type="crypto" if "/" in signal.symbol else "equity",
            )

            if not sim_result.get('executed', True):
                logger.warning("engine.sim_rejected", symbol=signal.symbol,
                             reason=sim_result.get('rejection_reason'))
                self._publish(OrderRejected(
                    order_id="",
                    reason=f"sim_rejected: {sim_result.get('rejection_reason', 'no_fill')}",
                    source="simulator",
                ))
                self._record_contract_rejection(decision_contract, signal.symbol, "sim_rejected")
                if trade_lifecycle:
                    trade_lifecycle.transition(TradeState.CANCELLED, "sim_rejected")
                return None

            logger.info("engine.sim_fill",
                       symbol=signal.symbol,
                       avg_price=sim_result.get('avg_price'),
                       slippage_bps=sim_result.get('slippage_bps', 0),
                       fees=sim_result.get('fees', 0))

            sim_qty = sim_result.get('total_qty', final_qty)
            if sim_qty < final_qty:
                logger.info("engine.sim_partial_fill", symbol=signal.symbol,
                           requested=final_qty, filled=sim_qty)
                final_qty = sim_qty

        # Transition lifecycle: SUBMITTED
        if trade_lifecycle:
            trade_lifecycle.transition(TradeState.SUBMITTED, "order submitted to broker")

        # Place the order — use contract SL/TP (system-determined) not strategy hints
        try:
            if final_stop_loss and final_take_profit:
                order = self.broker.bracket_order(
                    symbol=signal.symbol,
                    qty=final_qty,
                    side=side,
                    stop_loss_price=final_stop_loss,
                    take_profit_price=final_take_profit,
                )
            else:
                order = self.broker.market_order(
                    symbol=signal.symbol,
                    qty=final_qty,
                    side=side,
                )

            # Check for error response
            if not order or order.get("error"):
                error_msg = order.get("message", "Unknown order error") if order else "No response from broker"
                logger.error("engine.order_rejected_by_broker",
                            symbol=signal.symbol, error=error_msg, side=side)

                if trade_lifecycle:
                    trade_lifecycle.transition(TradeState.CANCELLED, f"broker_rejected: {error_msg}")

                self._publish(OrderRejected(
                    order_id="",
                    reason=error_msg,
                    source="broker",
                ))
                self._record_contract_rejection(decision_contract, signal.symbol, "broker_rejected")
                return None

            # Publish OrderSubmitted event
            self._publish(OrderSubmitted(
                symbol=signal.symbol,
                side=side,
                qty=final_qty,
                price=observed_price,
                order_type="bracket" if final_stop_loss else "market",
                order_id=order.get("id", ""),
                source="engine",
            ))

            # Use trade lifecycle ID as the canonical trade_id for event correlation
            trade_id = trade_lifecycle.trade_id if trade_lifecycle else order.get("id", "")
            if trade_lifecycle:
                trade_lifecycle.transition(TradeState.ACCEPTED, "broker accepted")
                trade_lifecycle.metadata['order_id'] = order.get("id", "")
                trade_lifecycle.metadata['symbol'] = signal.symbol
                trade_lifecycle.metadata['total_qty'] = final_qty
                trade_lifecycle.metadata['contract_id'] = decision_contract.contract_id if decision_contract else ""

            # Mark contract as executed in the store
            if decision_contract and self._contract_store:
                try:
                    self._contract_store.mark_executed(decision_contract.contract_id, trade_id)
                except Exception as e:
                    logger.warning("engine.contract_mark_executed_error", error=str(e))

            # Determine fill mode: immediate (simulation or no trade stream) vs deferred
            use_immediate_fill = bool(sim_result) or not self._has_trade_stream()

            fill_price = sim_result['avg_price'] if sim_result else observed_price
            fill_fees = sim_result['fees'] if sim_result else 0.0

            if use_immediate_fill:
                if trade_lifecycle:
                    trade_lifecycle.transition(TradeState.FILLED, f"filled qty={final_qty}")
                    trade_lifecycle.transition(TradeState.ACTIVE, "position active")

                self._publish(OrderFilled(
                    order_id=order.get("id", ""),
                    fill_price=fill_price,
                    fill_qty=final_qty,
                    fees=fill_fees,
                    source="engine",
                ))

                self._publish(TradeOpened(
                    trade_id=trade_id,
                    symbol=signal.symbol,
                    side=side,
                    entry_price=fill_price,
                    qty=final_qty,
                    stop_loss=final_stop_loss or 0.0,
                    take_profit=final_take_profit or 0.0,
                    contract_id=decision_contract.contract_id if decision_contract else "",
                    source="engine",
                ))

                # Store trade context for closed-loop feedback on exit
                with self._trade_context_lock:
                    self._trade_context[trade_id] = {
                        "signal_id": signal.signal_id,
                        "signal_package_id": signal_package.signal_id if signal_package else "",
                        "model_version": signal.strategy_version,
                        "strategy": signal.strategy_id,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "side": side,
                        "entry_price": fill_price,
                        "contract_id": decision_contract.contract_id if decision_contract else "",
                    }
            else:
                if trade_lifecycle:
                    trade_lifecycle.metadata['pending_fill'] = True
                    trade_lifecycle.metadata['expected_qty'] = final_qty
                logger.info("engine.pending_fill", order_id=order.get("id", ""), qty=final_qty)

            # Record trade to database (full audit trail)
            self.db.record_trade({
                "symbol": signal.symbol,
                "side": side,
                "qty": final_qty,
                "price": observed_price,
                "order_type": "bracket" if final_stop_loss else "market",
                "status": order.get("status", "submitted"),
                "order_id": order.get("id"),
                "strategy": signal.strategy_id,
                "signal_id": signal.signal_id,
                "signal_strength": signal.signal_strength,
                "stop_loss": final_stop_loss,
                "take_profit": final_take_profit,
                "sim_slippage_bps": sim_result.get("slippage_bps") if sim_result else None,
                "sim_fees": fill_fees if sim_result else None,
                "contract_id": decision_contract.contract_id if decision_contract else None,
                "contract_confidence": decision_contract.final_confidence if decision_contract else None,
                "contract_decision": decision_contract.decision.value if decision_contract else None,
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
                        "strategy_name": signal.strategy_id,
                        "model_version": signal.strategy_version,
                        "prediction": side,
                        "confidence": effective_confidence,
                        "signal_strength": signal.signal_strength,
                        "signal_id": signal.signal_id,
                        "contract_id": decision_contract.contract_id if decision_contract else "",
                        "features_snapshot": signal.indicators,
                        "market_regime": (
                            intelligence_decision.market_state.overall_regime
                            if intelligence_decision else None
                        ),
                    })
                except Exception as e:
                    logger.debug("journal.log_entry_error", error=str(e))

            logger.info("engine.order_placed",
                       symbol=signal.symbol, side=side, qty=final_qty,
                       signal_id=signal.signal_id,
                       confidence=f"{effective_confidence:.1%}")

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
            self._record_contract_rejection(decision_contract, signal.symbol, "order_exception")
            return None

    def _record_contract_rejection(
        self, contract, symbol: str, reason: str
    ) -> None:
        """Record a contract outcome for rejected/failed execution attempts."""
        if contract and self._contract_store:
            try:
                self._contract_store.record_outcome(
                    contract_id=contract.contract_id,
                    trade_id="",
                    symbol=symbol,
                    side=contract.direction,
                    entry_price=0.0,
                    exit_price=0.0,
                    pnl=0.0,
                    pnl_pct=0.0,
                    exit_reason=reason,
                )
            except Exception as e:
                logger.debug("engine.contract_rejection_record_error", error=str(e))

    def _check_exits(self, strategy: BaseStrategy, positions: list[dict], data: dict) -> list[dict]:
        """Check if any open positions should be closed based on strategy logic."""
        # Risk gate: if trading is halted, do not execute exits either
        can_trade, halt_reason = self.risk.can_trade()
        if not can_trade:
            logger.warning("engine.exits_blocked_by_risk", reason=halt_reason)
            return []

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

                        # Publish TradeClosed event with full context
                        with self._trade_context_lock:
                            ctx = self._trade_context.pop(trade_id, {})
                        self._publish(TradeClosed(
                            trade_id=trade_id,
                            symbol=symbol,
                            exit_price=current_price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=exit_signal.reason[:50] if exit_signal.reason else "strategy_exit",
                            source="engine",
                            entry_price=ctx.get("entry_price", entry_price),
                            side=ctx.get("side", pos.get("side", "")),
                            entry_time=ctx.get("entry_time", ""),
                            exit_time=datetime.now(timezone.utc).isoformat(),
                            model_version=ctx.get("model_version", ""),
                            strategy_name=ctx.get("strategy", self._current_strategy_name),
                            signal_package_id=ctx.get("signal_package_id", ""),
                            contract_id=ctx.get("contract_id", ""),
                        ))

                        # Record contract outcome for audit trail
                        contract_id = ctx.get("contract_id", "")
                        if contract_id and self._contract_store:
                            try:
                                self._contract_store.record_outcome(
                                    contract_id=contract_id,
                                    trade_id=trade_id,
                                    symbol=symbol,
                                    side=ctx.get("side", pos.get("side", "")),
                                    entry_price=ctx.get("entry_price", entry_price),
                                    exit_price=current_price,
                                    pnl=pnl,
                                    pnl_pct=pnl_pct,
                                    exit_reason=exit_signal.reason[:50] if exit_signal.reason else "strategy_exit",
                                )
                            except Exception as e:
                                logger.warning("engine.contract_outcome_error", error=str(e))

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
        """Log signal to database (legacy compat)."""
        import json
        try:
            self.db.log_signal({
                "symbol": signal.symbol,
                "strategy": self._current_strategy_name,
                "signal": signal.side.value if hasattr(signal, 'side') else "UNKNOWN",
                "confidence": signal.signal_strength if hasattr(signal, 'signal_strength') else 0.0,
                "price_at_signal": signal.features.get("observed_price", 0.0) if hasattr(signal, 'features') else 0.0,
                "indicators": json.dumps(signal.indicators if hasattr(signal, 'indicators') else {}),
                "was_executed": executed,
            })
        except Exception as e:
            logger.error("engine.signal_log_error", error=str(e))

    def _log_signal_v2(self, signal: TradeSignal, executed: bool, intelligence_decision=None):
        """Log clean TradeSignal to database with full provenance."""
        import json
        try:
            # Build extended indicators JSON (includes signal_id + intelligence)
            extended_indicators = dict(signal.indicators)
            extended_indicators["_signal_id"] = signal.signal_id
            if intelligence_decision:
                extended_indicators["_intelligence_score"] = round(intelligence_decision.final_score, 4)
                extended_indicators["_intelligence_regime"] = getattr(
                    intelligence_decision, 'market_state', None
                ) and intelligence_decision.market_state.overall_regime or ""

            self.db.log_signal({
                "symbol": signal.symbol,
                "strategy": signal.strategy_id,
                "signal": signal.side.value,
                "confidence": signal.signal_strength,
                "price_at_signal": signal.features.get("observed_price", 0.0),
                "indicators": json.dumps(extended_indicators),
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

    @staticmethod
    def _build_portfolio_context(positions: list[dict], portfolio_value: float, target_symbol: str) -> dict:
        """Build simple portfolio context used by the intelligence layer."""
        if portfolio_value <= 0:
            return {"correlation_risk": 0.0}

        active = [p for p in positions if p.get("symbol") != target_symbol]
        if not active:
            return {"correlation_risk": 0.0}

        notional_sum = sum(abs(float(p.get("market_value", 0.0))) for p in active)
        exposure = min(notional_sum / portfolio_value, 1.5)

        # Correlation proxy in absence of a full matrix: concentrated books imply
        # higher hidden co-movement risk.
        concentration = min(len(active) / 10.0, 1.0)
        correlation_risk = min((0.6 * exposure) + (0.4 * concentration), 1.0)
        return {"correlation_risk": correlation_risk}

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

        # Close all and verify — only publish events if close succeeds
        close_succeeded = False
        try:
            self.broker.close_all_positions()
            close_succeeded = True
        except Exception as e:
            logger.error("engine.emergency_close_failed", error=str(e))

        if not close_succeeded:
            # Cannot confirm closure — publish halt but NOT TradeClosed events
            self._publish(RiskHalt(
                reason="Emergency liquidation FAILED — positions may still be open",
                level="CRITICAL",
                source="engine",
            ))
            return

        for pos in positions:
            symbol = pos.get('symbol', '?')
            pnl = pos.get('unrealized_pl', 0)
            current_price = pos.get('current_price', 0)
            entry_price = pos.get('avg_entry_price', 0)
            pnl_pct = 0.0
            if entry_price and abs(entry_price) > 1e-8:
                pnl_pct = (current_price - entry_price) / entry_price * 100

            # Transition trade lifecycle and retrieve contract_id
            trade_id = pos.get('asset_id', '')
            contract_id = ""
            if self._trade_manager:
                active_trades = self._trade_manager.get_trades_by_symbol(symbol)
                for t in active_trades:
                    if not t.is_terminal:
                        t.transition(TradeState.CLOSING, "emergency_liquidation")
                        t.transition(TradeState.CLOSED, "emergency_liquidation")
                        trade_id = t.trade_id
                        contract_id = t.metadata.get("contract_id", "")
                        break

            # Fallback: check _trade_context for contract_id
            if not contract_id and trade_id and trade_id in self._trade_context:
                contract_id = self._trade_context[trade_id].get("contract_id", "")

            self._publish(TradeClosed(
                trade_id=trade_id,
                symbol=symbol,
                exit_price=current_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason="emergency_liquidation",
                contract_id=contract_id,
                source="engine",
            ))

            # Record outcome in contract store
            if contract_id and self._contract_store:
                try:
                    side = self._trade_context.get(trade_id, {}).get("side", "buy")
                    self._contract_store.record_outcome(
                        contract_id=contract_id,
                        trade_id=trade_id,
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=current_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason="emergency_liquidation",
                    )
                except Exception as e:
                    logger.debug("engine.emergency_outcome_record_error", error=str(e))

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
