from unittest.mock import MagicMock, patch

import pytest

from src.core.events import EventBus, BrokerSwitched
from config.settings import TradingMode, settings
from src.core.environment import BrokerManager, EnvironmentManager


class _DummyBroker:
    def __init__(self, name: str, paper: bool, account_ok: bool = True):
        self._name = name
        self.paper = paper
        self._account_ok = account_ok

    @property
    def name(self) -> str:
        return self._name

    def get_account(self):
        if not self._account_ok:
            raise RuntimeError("account unavailable")
        return {"equity": 100000.0}

    def get_positions(self):
        return []

    def close_position(self, symbol: str):
        return None


def test_environment_switch_rolls_back_on_verify_failure():
    old_settings_mode = settings.trading_mode
    old_live_key = settings.alpaca_live_api_key
    old_live_secret = settings.alpaca_live_secret_key
    old_broker = _DummyBroker("paper", paper=True)
    manager = BrokerManager(old_broker)
    settings.alpaca_live_api_key = "LIVEKEY123"
    settings.alpaca_live_secret_key = "LIVESECRET123"

    def _factory(_mode):
        # Returning paper=True for LIVE target forces verify failure post-swap.
        return _DummyBroker("bad-live", paper=True)

    env_mgr = EnvironmentManager(
        broker_manager=manager,
        broker_factory=_factory,
    )

    try:
        with pytest.raises(RuntimeError, match="Environment switch failed"):
            env_mgr.switch_environment(TradingMode.LIVE, reason="test-verify-failure")

        assert env_mgr.current_mode == TradingMode.PAPER
        assert settings.trading_mode == old_settings_mode
        assert manager.broker is old_broker
        assert env_mgr.state.value == "idle"
        last = env_mgr.get_switch_history(limit=1)[0]
        assert last["rollback_performed"] is True
        assert last["rollback_success"] is True
    finally:
        settings.trading_mode = old_settings_mode
        settings.alpaca_live_api_key = old_live_key
        settings.alpaca_live_secret_key = old_live_secret


def test_environment_rollback_is_idempotent_and_non_recursive():
    old_settings_mode = settings.trading_mode
    old_live_key = settings.alpaca_live_api_key
    old_live_secret = settings.alpaca_live_secret_key
    settings.alpaca_live_api_key = "LIVEKEY123"
    settings.alpaca_live_secret_key = "LIVESECRET123"

    old_broker = _DummyBroker("paper", paper=True)
    bus = EventBus()
    broker_events = []
    bus.subscribe(BrokerSwitched, lambda e: broker_events.append(e))
    manager = BrokerManager(old_broker, event_bus=bus)

    def _factory(_mode):
        # Invalid live broker (paper=True) forces verify failure after swap.
        return _DummyBroker("bad-live", paper=True)

    env_mgr = EnvironmentManager(
        broker_manager=manager,
        broker_factory=_factory,
        event_bus=bus,
    )

    try:
        with pytest.raises(RuntimeError, match="Environment switch failed"):
            env_mgr.switch_environment(TradingMode.LIVE, reason="rollback-idempotence")

        assert manager.broker is old_broker
        assert env_mgr.current_mode == TradingMode.PAPER
        assert settings.trading_mode == old_settings_mode
        # Exactly one broker switch event (forward attempt), no recursive restore event.
        assert len(broker_events) == 1

        last = env_mgr.get_switch_history(limit=1)[0]
        assert last["rollback_performed"] is True
        assert last["rollback_success"] is True
    finally:
        settings.trading_mode = old_settings_mode
        settings.alpaca_live_api_key = old_live_key
        settings.alpaca_live_secret_key = old_live_secret


def test_runtime_shutdown_stops_all_components():
    with patch("src.core.runtime.ModelRegistry"), \
         patch("src.core.runtime.ModelGovernance"), \
         patch("src.core.runtime.ModelPredictor"), \
         patch("src.core.runtime.TrainingPipeline"), \
         patch("src.core.runtime.ABTestManager"), \
         patch("src.core.runtime.BacktestRunner"), \
         patch("src.core.runtime.EnvironmentManager"):
        from src.core.runtime import RuntimeManager

        broker_manager = MagicMock()
        broker_manager.broker = MagicMock()
        broker_manager.get_status.return_value = {"broker": "paper"}

        rm = RuntimeManager(broker_manager=broker_manager, event_bus=MagicMock())
        rm.shutdown()

        rm._training_pipeline.stop.assert_called_once()
        rm._ab_manager.cancel_test.assert_called_once_with(reason="runtime_shutdown")
        rm._backtest_runner.stop.assert_called_once()
        rm._predictor.stop.assert_called_once()


def test_alpaca_stop_paths_join_threads():
    from src.broker.alpaca_client import AlpacaBroker

    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._shutdown_flag = False
    broker._trade_stream = MagicMock()
    broker._trade_stream_thread = MagicMock()
    broker._trade_stream_thread.is_alive.return_value = True
    broker._stock_stream = MagicMock()
    broker._crypto_stream = MagicMock()

    trade_thread = broker._trade_stream_thread
    broker.stop_trade_stream()

    assert broker._shutdown_flag is True
    trade_thread.join.assert_called_once()
    broker._stock_stream = MagicMock()
    broker._crypto_stream = MagicMock()
    stock_stream = broker._stock_stream
    crypto_stream = broker._crypto_stream
    broker.stop_market_data_streams()

    stock_stream.stop.assert_called_once()
    crypto_stream.stop.assert_called_once()
    assert broker._trade_stream is None
    assert broker._stock_stream is None
    assert broker._crypto_stream is None


def test_backtest_runner_stop_joins_async_threads():
    from backtesting.runner import BacktestRunner

    runner = BacktestRunner()
    thread = MagicMock()
    thread.is_alive.return_value = True
    runner._async_threads = {"run123": thread}

    runner.stop(timeout=2.0)

    thread.join.assert_called_once_with(timeout=2.0)


def test_training_pipeline_stop_logs_timeout():
    from src.ml.training_pipeline import TrainingPipeline

    pipeline = TrainingPipeline(registry=MagicMock(), governance=MagicMock())
    pipeline._thread = MagicMock()
    pipeline._thread.is_alive.return_value = True

    with patch("src.ml.training_pipeline.logger.warning") as warn:
        pipeline.stop(timeout=1.5)

    pipeline._thread.join.assert_called_once_with(timeout=1.5)
    warn.assert_called_once()
