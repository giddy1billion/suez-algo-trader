"""
Runtime Manager — Unified interface for all runtime capabilities.

This module ties together all hot-swap, backtest, training, and A/B testing
capabilities into a single coherent API that can be used from:
- Telegram commands
- CLI arguments
- Scheduler triggers
- Programmatic calls

Capabilities:
- switch_environment(paper/live) — hot-swap trading mode
- switch_operational_mode(research/paper/live) — three-mode operation
- run_backtest(strategies, symbols) — concurrent multi-strategy backtesting
- train_model(symbols) — end-to-end training pipeline
- swap_model(version) — transparent model hot-swap
- start_ab_test(challenger) — A/B test model versions
- get_status() — comprehensive system status
"""

import threading
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from config.settings import OperationalMode, TradingMode, settings
from src.core.environment import BrokerManager, EnvironmentManager, create_broker_for_mode
from src.core.events import OperationalModeChanged
from src.ml.model_registry import ModelRegistry
from src.ml.governance import ModelGovernance
from src.ml.predictor import ModelPredictor
from src.ml.training_pipeline import TrainingPipeline
from src.ml.ab_testing import ABTestManager, ABTestMode
from backtesting.runner import BacktestRunner
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9/]{1,10}$")
_STRATEGY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_MODEL_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_TIMEFRAME_WHITELIST = {"1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day", "1Week"}


class RuntimeManager:
    """
    Unified runtime capabilities manager.

    Provides a single entry point for all dynamic operations
    that don't require system restart.
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        event_bus=None,
        registry: Optional[ModelRegistry] = None,
        governance: Optional[ModelGovernance] = None,
        strategy_factory: Optional[Callable] = None,
        operational_mode: Optional[OperationalMode] = None,
    ):
        self._broker_manager = broker_manager
        self._event_bus = event_bus
        self._strategy_factory = strategy_factory
        self._operational_mode = operational_mode or settings.operational_mode

        # Environment switching
        self._env_manager = EnvironmentManager(
            broker_manager=broker_manager,
            broker_factory=create_broker_for_mode,
            event_bus=event_bus,
        )

        # ML components
        self._registry = registry or ModelRegistry()
        self._governance = governance or ModelGovernance()

        # Feature store for prediction reproducibility
        from src.ml.feature_store import FeatureStore
        self._feature_store = FeatureStore()

        # Closed-loop feedback components
        from src.ml.feedback_loop import ExperienceDatabase
        from src.ml.promotion_engine import ModelPromotionEngine
        from src.ml.dataset_registry import DatasetRegistry
        try:
            self._experience_db = ExperienceDatabase()
        except Exception as e:
            self._experience_db = None
            logger.warning("runtime_manager.experience_db_unavailable", error=str(e))
        self._promotion_engine = ModelPromotionEngine(
            min_evaluation_trades=30,
            min_improvement_pct=5.0,
        )
        try:
            self._dataset_registry = DatasetRegistry()
        except Exception as e:
            self._dataset_registry = None
            logger.warning("runtime_manager.dataset_registry_unavailable", error=str(e))

        self._predictor = ModelPredictor(
            registry=self._registry,
            event_bus=event_bus,
            auto_reload=True,
            feature_store=self._feature_store,
            dataset_registry=self._dataset_registry,
        )

        self._training_pipeline = TrainingPipeline(
            registry=self._registry,
            governance=self._governance,
            broker=broker_manager.broker,
            event_bus=event_bus,
            experience_db=self._experience_db,
            dataset_registry=self._dataset_registry,
        )
        self._ab_manager = ABTestManager(
            registry=self._registry,
            predictor=self._predictor,
            event_bus=event_bus,
            promotion_engine=self._promotion_engine,
        )

        # Backtesting
        self._backtest_runner = BacktestRunner(
            max_workers=4,
            event_bus=event_bus,
        )

        logger.info("runtime_manager.initialized")

    # ──────────────────────────────────────────────────────────────────────
    # Environment Switching
    # ──────────────────────────────────────────────────────────────────────

    def switch_to_paper(self, reason: str = "manual") -> dict:
        """Switch to paper trading mode without restart."""
        return self._env_manager.switch_to_paper(reason=reason)

    def switch_to_live(self, reason: str = "manual") -> dict:
        """Switch to live trading mode without restart."""
        return self._env_manager.switch_to_live(reason=reason)

    def switch_environment(self, mode: str, reason: str = "manual") -> dict:
        """Switch environment by mode string ('paper' or 'live')."""
        target = TradingMode.PAPER if mode.lower() == "paper" else TradingMode.LIVE
        return self._env_manager.switch_environment(target, reason=reason)

    @property
    def current_mode(self) -> str:
        return self._env_manager.current_mode.value

    @property
    def is_paper(self) -> bool:
        return self._env_manager.is_paper

    @property
    def is_live(self) -> bool:
        return self._env_manager.is_live

    # ──────────────────────────────────────────────────────────────────────
    # Broker Hot-Swap
    # ──────────────────────────────────────────────────────────────────────

    def swap_broker(self, new_broker, drain_positions: bool = True) -> dict:
        """Hot-swap the active broker instance."""
        return self._broker_manager.switch_broker(
            new_broker, drain_positions=drain_positions
        )

    @property
    def broker(self):
        """Get the current active broker."""
        return self._broker_manager.broker

    # ──────────────────────────────────────────────────────────────────────
    # Multi-Strategy Backtesting
    # ──────────────────────────────────────────────────────────────────────

    def run_backtest(
        self,
        strategy_names: list[str],
        symbols: Optional[list[str]] = None,
        timeframe: str = "1Hour",
        lookback: int = 200,
        blocking: bool = False,
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        Run multi-strategy backtest (concurrent, non-blocking by default).

        Args:
            strategy_names: List of strategy names to test.
            symbols: Symbols to backtest on (defaults to settings).
            timeframe: Bar timeframe.
            lookback: Number of bars.
            blocking: If True, wait for completion.
            callback: Called with results when complete (async mode).

        Returns:
            Dict with run_id (async) or full results (blocking).
        """
        symbols = symbols or settings.symbols_list
        normalized_strategies = self._validate_strategy_names(strategy_names)
        normalized_symbols = self._validate_symbols(symbols)
        normalized_timeframe = self._validate_timeframe(timeframe)
        normalized_lookback = self._validate_lookback(lookback, field="lookback", maximum=10000)

        if not self._strategy_factory:
            raise RuntimeError("No strategy_factory configured")
        logger.info(
            "runtime.backtest.started",
            strategies=normalized_strategies,
            symbols=normalized_symbols,
            timeframe=normalized_timeframe,
            lookback=normalized_lookback,
            blocking=blocking,
        )

        # Create strategy instances
        strategies = []
        for name in normalized_strategies:
            try:
                strategy = self._strategy_factory(
                    name,
                    normalized_symbols,
                    normalized_timeframe,
                    normalized_lookback,
                )
                strategies.append(strategy)
            except Exception as e:
                logger.error("runtime.backtest.strategy_creation_failed", name=name, error=str(e))

        if not strategies:
            raise RuntimeError("No valid strategies created")

        # Fetch data
        data = {}
        broker = self._broker_manager.broker
        for symbol in normalized_symbols:
            try:
                df = broker.get_bars_df(symbol, normalized_timeframe, normalized_lookback)
                if df is not None and len(df) >= 50:
                    data[symbol] = df
            except Exception as e:
                logger.warning("runtime.backtest.data_error", symbol=symbol, error=str(e))

        if not data:
            raise RuntimeError("No market data available for backtesting")

        # Run
        if blocking:
            result = self._backtest_runner.run_multiple(strategies, data)
            logger.info(
                "runtime.backtest.completed",
                run_id=result.run_id,
                status=result.status.value,
                strategies=len(normalized_strategies),
                symbols=len(data),
            )
            return {
                "run_id": result.run_id,
                "status": result.status.value,
                "summary": result.summary(),
                "best_strategy": result.get_best_strategy(),
                "comparison": result.get_comparison_df().to_dict() if not result.get_comparison_df().empty else {},
            }
        else:
            run_id = self._backtest_runner.run_async(strategies, data, callback=callback)
            logger.info(
                "runtime.backtest.launched",
                run_id=run_id,
                strategies=len(normalized_strategies),
                symbols=len(data),
            )
            return {"run_id": run_id, "status": "running", "strategies": normalized_strategies}

    def get_backtest_status(self, run_id: str) -> Optional[dict]:
        """Get status of a running backtest."""
        return self._backtest_runner.get_run_status(run_id)

    def get_backtest_result(self, run_id: str):
        """Get full results of a completed backtest."""
        return self._backtest_runner.get_result(run_id)

    def list_backtests(self, limit: int = 10) -> list[dict]:
        """List recent backtest runs."""
        return self._backtest_runner.list_runs(limit)

    # ──────────────────────────────────────────────────────────────────────
    # Model Training Pipeline
    # ──────────────────────────────────────────────────────────────────────

    def train_model(
        self,
        symbols: Optional[list[str]] = None,
        timeframe: str = "1Hour",
        lookback_bars: int = 1000,
        trigger: str = "manual",
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        Trigger end-to-end ML training pipeline (non-blocking).

        Args:
            symbols: Symbols to train on (defaults to settings).
            timeframe: Bar timeframe for training data.
            lookback_bars: Number of bars per symbol.
            trigger: What triggered training.
            callback: Called with progress when complete.

        Returns:
            Dict with pipeline_id for tracking.
        """
        symbols = symbols or settings.symbols_list
        normalized_symbols = self._validate_symbols(symbols)
        normalized_timeframe = self._validate_timeframe(timeframe)
        normalized_lookback = self._validate_lookback(
            lookback_bars,
            field="lookback_bars",
            minimum=50,
            maximum=50000,
        )
        normalized_trigger = (trigger or "manual").strip()
        if not normalized_trigger or len(normalized_trigger) > 64:
            raise ValueError("trigger must be between 1 and 64 characters")

        # Update broker reference in pipeline (may have been swapped)
        self._training_pipeline._broker = self._broker_manager.broker

        logger.info(
            "runtime.training.started",
            symbols=normalized_symbols,
            timeframe=normalized_timeframe,
            lookback_bars=normalized_lookback,
            trigger=normalized_trigger,
        )
        pipeline_id = self._training_pipeline.train(
            symbols=normalized_symbols,
            timeframe=normalized_timeframe,
            lookback_bars=normalized_lookback,
            trigger=normalized_trigger,
            callback=callback,
        )

        return {
            "pipeline_id": pipeline_id,
            "status": "running",
            "symbols": normalized_symbols,
            "trigger": normalized_trigger,
        }

    def get_training_progress(self) -> Optional[dict]:
        """Get current training pipeline progress."""
        return self._training_pipeline.get_progress()

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._training_pipeline.is_running()

    def get_training_history(self, limit: int = 10) -> list[dict]:
        """Get recent training history."""
        return self._training_pipeline.get_history(limit)

    # ──────────────────────────────────────────────────────────────────────
    # Model Hot-Swap
    # ──────────────────────────────────────────────────────────────────────

    def swap_model(self, version: str) -> dict:
        """
        Hot-swap the active ML model to a specific version.

        The model is transparently reloaded in all strategies
        that use the ModelPredictor.
        """
        normalized_version = self._validate_model_version(version)
        logger.info("runtime.model_swap.started", version=normalized_version)
        # Update registry active version
        self._registry.rollback(normalized_version)
        # Force predictor reload
        result = self._predictor.swap_model(normalized_version)
        logger.info("runtime.model_swap.completed", version=normalized_version)
        return result

    def get_model_status(self) -> dict:
        """Get current model predictor status."""
        return self._predictor.get_metrics()

    def list_model_versions(self) -> list[dict]:
        """List all available model versions."""
        return self._registry.list_versions()

    @property
    def predictor(self) -> ModelPredictor:
        """Access the centralized model predictor."""
        return self._predictor

    # ──────────────────────────────────────────────────────────────────────
    # A/B Testing
    # ──────────────────────────────────────────────────────────────────────

    def start_ab_test(
        self,
        challenger_version: str,
        mode: str = "shadow",
        allocation_pct: float = 0.2,
        min_trades: int = 30,
        auto_promote: bool = True,
    ) -> dict:
        """
        Start an A/B test comparing current model vs a challenger.

        Args:
            challenger_version: Version to test.
            mode: "shadow", "split", or "interleaved".
            allocation_pct: Fraction of decisions for challenger.
            min_trades: Min trades before conclusion.
            auto_promote: Auto-deploy winner.

        Returns:
            Dict with test_id.
        """
        normalized_version = self._validate_model_version(challenger_version)
        if not (0.0 < allocation_pct <= 1.0):
            raise ValueError("allocation_pct must be in (0, 1]")
        if not (1 <= min_trades <= 100000):
            raise ValueError("min_trades must be between 1 and 100000")
        mode_name = mode.strip().lower()
        try:
            mode_enum = ABTestMode(mode_name)
        except ValueError:
            valid = ", ".join(m.value for m in ABTestMode)
            raise ValueError(f"Invalid A/B test mode: {mode}. Must be one of: {valid}") from None

        logger.info(
            "runtime.ab_test.started",
            challenger_version=normalized_version,
            mode=mode_enum.value,
            allocation_pct=allocation_pct,
            min_trades=min_trades,
            auto_promote=auto_promote,
        )
        test_id = self._ab_manager.start_test(
            challenger_version=normalized_version,
            mode=mode_enum,
            allocation_pct=allocation_pct,
            min_trades=min_trades,
            auto_promote=auto_promote,
        )
        return {"test_id": test_id, "status": "active", "mode": mode_enum.value}

    @staticmethod
    def _validate_strategy_names(strategy_names: list[str]) -> list[str]:
        if not strategy_names:
            raise ValueError("At least one strategy name is required")
        if len(strategy_names) > 20:
            raise ValueError("No more than 20 strategy names are allowed")
        normalized = []
        for name in strategy_names:
            strategy_name = (name or "").strip().lower()
            if not _STRATEGY_PATTERN.match(strategy_name):
                raise ValueError(
                    f"Invalid strategy name '{name}'. Use lowercase letters, digits, underscores (max 32 chars)."
                )
            normalized.append(strategy_name)
        return normalized

    @staticmethod
    def _validate_symbols(symbols: list[str]) -> list[str]:
        if not symbols:
            raise ValueError("At least one symbol is required")
        if len(symbols) > 100:
            raise ValueError("No more than 100 symbols are allowed")
        normalized = []
        for raw_symbol in symbols:
            symbol = (raw_symbol or "").strip().upper()
            if not _SYMBOL_PATTERN.match(symbol):
                raise ValueError(
                    f"Invalid symbol '{raw_symbol}'. Use uppercase letters, digits, and / (max 10 chars)."
                )
            normalized.append(symbol)
        return normalized

    @staticmethod
    def _validate_timeframe(timeframe: str) -> str:
        normalized = (timeframe or "").strip()
        if normalized not in _TIMEFRAME_WHITELIST:
            allowed = ", ".join(sorted(_TIMEFRAME_WHITELIST))
            raise ValueError(f"Invalid timeframe '{timeframe}'. Allowed: {allowed}")
        return normalized

    @staticmethod
    def _validate_lookback(
        value: int,
        field: str = "lookback",
        minimum: int = 1,
        maximum: int = 10000,
    ) -> int:
        if not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
        if value < minimum or value > maximum:
            raise ValueError(f"{field} must be between {minimum} and {maximum}")
        return value

    @staticmethod
    def _validate_model_version(version: str) -> str:
        normalized = (version or "").strip()
        if not _MODEL_VERSION_PATTERN.match(normalized):
            raise ValueError(
                "Invalid model version. Use letters, digits, dot, underscore, or hyphen (max 64 chars)."
            )
        return normalized

    def record_ab_trade(self, model_version: str, trade_result: dict) -> None:
        """Record a trade for A/B test evaluation."""
        self._ab_manager.record_trade(model_version, trade_result)

    def get_ab_test_status(self) -> Optional[dict]:
        """Get active A/B test status."""
        return self._ab_manager.get_test_status()

    def cancel_ab_test(self, reason: str = "manual") -> Optional[dict]:
        """Cancel the active A/B test."""
        return self._ab_manager.cancel_test(reason)

    def list_ab_tests(self, limit: int = 10) -> list[dict]:
        """List recent A/B tests."""
        return self._ab_manager.list_tests(limit)

    # ──────────────────────────────────────────────────────────────────────
    # Operational Modes (Research / Paper / Live)
    # ──────────────────────────────────────────────────────────────────────

    def switch_operational_mode(
        self, mode: str, reason: str = "manual"
    ) -> dict:
        """
        Switch operational mode (research/paper/live).

        - Research: data ingestion, backtests, training, no execution
        - Paper: full pipeline with simulated execution
        - Live: real orders with all safeguards

        Args:
            mode: "research", "paper", or "live"
            reason: Why the mode is being changed

        Returns:
            Dict with old/new mode and status
        """
        try:
            new_mode = OperationalMode(mode.lower())
        except ValueError:
            raise ValueError(f"Invalid operational mode: {mode}. Must be research/paper/live")

        old_mode = self._operational_mode

        if new_mode == old_mode:
            return {"status": "unchanged", "mode": old_mode.value}

        # Transition validation
        if new_mode == OperationalMode.LIVE and old_mode == OperationalMode.RESEARCH:
            raise RuntimeError(
                "Cannot transition directly from research to live. "
                "Must go through paper mode first."
            )

        # If transitioning to paper/live, sync the broker environment
        if new_mode in (OperationalMode.PAPER, OperationalMode.LIVE):
            target_trading = (
                TradingMode.LIVE if new_mode == OperationalMode.LIVE
                else TradingMode.PAPER
            )
            try:
                self._env_manager.switch_environment(target_trading, reason=reason)
            except Exception as e:
                logger.error("runtime.mode_switch.env_failed", error=str(e))
                raise

        self._operational_mode = new_mode

        # Publish event
        if self._event_bus:
            self._event_bus.publish(OperationalModeChanged(
                old_mode=old_mode.value,
                new_mode=new_mode.value,
                reason=reason,
                source="runtime_manager",
            ))

        logger.info(
            "runtime.operational_mode_changed",
            old=old_mode.value,
            new=new_mode.value,
            reason=reason,
        )

        return {
            "status": "switched",
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
            "reason": reason,
        }

    @property
    def operational_mode(self) -> OperationalMode:
        """Current operational mode."""
        return self._operational_mode

    @property
    def is_research_mode(self) -> bool:
        return self._operational_mode == OperationalMode.RESEARCH

    @property
    def can_execute_trades(self) -> bool:
        """Whether the current mode allows trade execution."""
        return self._operational_mode != OperationalMode.RESEARCH

    # ──────────────────────────────────────────────────────────────────────
    # System Status
    # ──────────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Get comprehensive system runtime status.

        Returns single-pane view of all runtime capabilities.
        """
        return {
            "environment": {
                "mode": self._env_manager.current_mode.value,
                "operational_mode": self._operational_mode.value,
                "state": self._env_manager.state.value,
                "broker": self._broker_manager.get_status(),
            },
            "model": self._predictor.get_metrics(),
            "training": self._training_pipeline.get_progress(),
            "ab_test": self._ab_manager.get_test_status(),
            "backtests": self._backtest_runner.list_runs(5),
            "capabilities": {
                "env_switching": True,
                "broker_hot_swap": True,
                "multi_strategy_backtest": True,
                "concurrent_backtest": True,
                "model_hot_swap": True,
                "training_pipeline": True,
                "ab_testing": True,
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def shutdown(self):
        """Clean shutdown of all runtime components."""
        logger.info("runtime_manager.shutting_down")
        try:
            self._training_pipeline.stop()
        except Exception as e:
            logger.warning("runtime_manager.training_stop_failed", error=str(e))
        try:
            self._ab_manager.cancel_test(reason="runtime_shutdown")
        except Exception as e:
            logger.warning("runtime_manager.ab_cancel_failed", error=str(e))
        try:
            self._backtest_runner.stop()
        except Exception as e:
            logger.warning("runtime_manager.backtest_stop_failed", error=str(e))
        self._predictor.stop()
        logger.info("runtime_manager.shutdown_complete")
