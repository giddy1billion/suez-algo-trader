"""
End-to-end tests for /status and /modelinfo Telegram bot commands.

Verifies:
1. /status always returns a valid account summary in paper mode (regression).
2. After training a model, /modelinfo can find and load it (end-to-end).
3. Model persistence path consistency between training and loading.
"""

import os
import tempfile
import threading
from datetime import datetime
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.broker.paper import PaperBroker


class _PicklableModel:
    """Simple picklable stand-in for a trained model."""
    n_estimators = 100
    max_depth = 6


# ──────────────────────────────────────────────────────────────────────────────
# Regression: /status returns valid account summary in paper mode
# ──────────────────────────────────────────────────────────────────────────────


class TestStatusPaperMode:
    """Regression tests ensuring /status works correctly with PaperBroker."""

    def test_paper_broker_get_account_has_all_required_fields(self):
        """PaperBroker.get_account() must return all fields used by /status handler."""
        broker = PaperBroker(starting_equity=100_000.0)
        account = broker.get_account()

        # Fields used by the /status Telegram command handler
        required_fields = [
            "equity",
            "cash",
            "buying_power",
            "last_equity",
            "day_trade_count",
        ]
        for field in required_fields:
            assert field in account, f"Missing required field: {field}"

    def test_paper_broker_account_equity_calculation(self):
        """Equity should equal cash + position market value."""
        broker = PaperBroker(starting_equity=50_000.0)
        account = broker.get_account()
        assert account["equity"] == 50_000.0
        assert account["cash"] == 50_000.0
        assert account["last_equity"] == 50_000.0
        assert account["day_trade_count"] == 0

    def test_paper_broker_account_after_position(self):
        """Account fields remain valid after opening a position."""
        broker = PaperBroker(starting_equity=100_000.0)
        broker.set_price("AAPL", 150.0)
        broker.market_order("AAPL", 10, "buy")

        account = broker.get_account()
        assert account["equity"] > 0
        assert account["cash"] >= 0
        assert account["last_equity"] == 100_000.0
        assert account["buying_power"] >= 0
        assert account["day_trade_count"] == 0
        assert account["pattern_day_trader"] is False
        assert "portfolio_value" in account

    def test_paper_broker_positions_have_unrealized_plpc(self):
        """get_positions() must include unrealized_plpc for /positions handler."""
        broker = PaperBroker(starting_equity=100_000.0)
        broker.set_price("AAPL", 100.0)
        broker.market_order("AAPL", 10, "buy")
        broker.set_price("AAPL", 110.0)

        positions = broker.get_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert "unrealized_plpc" in pos
        assert pos["unrealized_plpc"] == pytest.approx(0.1, rel=1e-6)
        assert pos["unrealized_pl"] == pytest.approx(100.0, rel=1e-6)

    def test_status_command_does_not_raise_with_paper_broker(self):
        """Simulate the /status handler logic to verify no KeyError."""
        broker = PaperBroker(starting_equity=100_000.0)
        account = broker.get_account()
        positions = broker.get_positions()

        # Reproduce the exact calculation from cmd_status in telegram_bot.py
        pnl = account['equity'] - account['last_equity']
        pnl_pct = (pnl / account['last_equity'] * 100) if account['last_equity'] > 0 else 0

        text = (
            f"Equity: ${account['equity']:>12,.2f}\n"
            f"Cash: ${account['cash']:>12,.2f}\n"
            f"Buying Power: ${account['buying_power']:>12,.2f}\n"
            f"Day P&L: ${pnl:>+12,.2f} ({pnl_pct:+.2f}%)\n"
            f"Positions: {len(positions):>12d}\n"
            f"Day Trades: {account['day_trade_count']:>12d}\n"
        )
        assert "Equity" in text
        assert "Day Trades" in text


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end: Train model → /modelinfo finds it after restart
# ──────────────────────────────────────────────────────────────────────────────


def _make_training_data(symbols: list[str], n_bars: int = 500) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data suitable for MLStrategy.train().

    Uses alternating trends to ensure all 3 target classes (up/down/flat)
    are represented in the training data.
    """
    data = {}
    for i, symbol in enumerate(symbols):
        np.random.seed(42 + i)
        # Create data with clear trends to produce all 3 label classes
        noise = np.random.randn(n_bars) * 0.3
        trend = np.sin(np.linspace(0, 8 * np.pi, n_bars)) * 5  # oscillating
        close = 100 + np.cumsum(noise) + trend
        close = np.maximum(close, 10.0)  # keep positive
        df = pd.DataFrame({
            "open": close + np.random.randn(n_bars) * 0.2,
            "high": close + abs(np.random.randn(n_bars) * 1.0),
            "low": close - abs(np.random.randn(n_bars) * 1.0),
            "close": close,
            "volume": np.random.randint(10000, 1000000, n_bars).astype(float),
        })
        df.index = pd.date_range("2023-01-01", periods=n_bars, freq="h")
        data[symbol] = df
    return data


class TestModelInfoEndToEnd:
    """End-to-end test: train → persist → restart → /modelinfo loads model."""

    def test_train_and_modelinfo_finds_model(self, tmp_path):
        """After training, the model file should be loadable by /modelinfo logic."""
        import joblib
        from src.strategy.ml_strategy import MLStrategy

        model_path = str(tmp_path / "models" / "latest_model.joblib")
        symbols = ["AAPL", "MSFT"]

        # 1. Train and save model
        strategy = MLStrategy(
            symbols=symbols,
            timeframe="1Hour",
            lookback=500,
            model_path=model_path,
            min_confidence=0.65,
        )
        training_data = _make_training_data(symbols)
        strategy.train(training_data)

        # Verify model file was created
        assert os.path.exists(model_path), f"Model not saved at {model_path}"

        # 2. Simulate restart: create a fresh strategy instance (like bot restart)
        new_strategy = MLStrategy(
            symbols=symbols,
            timeframe="1Hour",
            lookback=500,
            model_path=model_path,
            min_confidence=0.65,
        )
        assert new_strategy.model is not None, "Model not loaded after restart"

        # 3. Simulate /modelinfo handler logic
        data = joblib.load(model_path)
        assert "model" in data
        assert "features" in data
        assert "trained_at" in data
        assert data["model"] is not None
        assert len(data["features"]) > 0
        assert isinstance(data["trained_at"], datetime)

    def test_modelinfo_path_matches_settings(self, tmp_path):
        """Verify that settings.ml_model_path is used consistently."""
        import joblib
        from src.strategy.ml_strategy import MLStrategy

        model_path = str(tmp_path / "latest_model.joblib")

        # Patch settings to use our temp path
        with patch("config.settings.settings.ml_model_path", model_path):
            from config.settings import settings
            assert settings.ml_model_path == model_path

            # Train using settings path
            strategy = MLStrategy(
                symbols=["AAPL"],
                timeframe="1Hour",
                lookback=500,
                model_path=settings.ml_model_path,
                min_confidence=0.65,
            )
            training_data = _make_training_data(["AAPL"])
            strategy.train(training_data)

            # Verify /modelinfo can find it at the same path
            assert os.path.exists(settings.ml_model_path)
            data = joblib.load(settings.ml_model_path)
            assert data["model"] is not None

    def test_model_registry_creates_latest_symlink(self, tmp_path):
        """ModelRegistry.save_version(activate=True) updates latest_model.joblib."""
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))

        version = registry.save_version(
            model=_PicklableModel(),
            features=["rsi_14", "ema_12", "volume_ratio"],
            metrics={"cv_accuracy": 0.62, "sharpe": 1.2},
            symbols=["AAPL"],
            activate=True,
        )

        # latest_model.joblib should exist
        latest = os.path.join(str(tmp_path / "models"), "latest_model.joblib")
        assert os.path.exists(latest), "latest_model.joblib not created on activate"

        # Should be loadable with expected structure
        import joblib
        data = joblib.load(latest)
        assert "model" in data
        assert "features" in data
        assert "trained_at" in data

    def test_model_not_activated_without_governance_approval(self, tmp_path):
        """ModelRegistry.save_version(activate=False) should NOT update latest."""
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(models_dir=str(tmp_path / "models"))

        registry.save_version(
            model=_PicklableModel(),
            features=["f1"],
            metrics={},
            symbols=["AAPL"],
            activate=False,
        )

        latest = os.path.join(str(tmp_path / "models"), "latest_model.joblib")
        assert not os.path.exists(latest), (
            "latest_model.joblib should NOT be updated when activate=False"
        )
