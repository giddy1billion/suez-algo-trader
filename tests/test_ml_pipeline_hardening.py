import numpy as np
import pandas as pd

from backtesting import vbt_adapter
from src.ml.features import engineer_features
from src.ml.governance import ModelGovernance
from src.ml.model_registry import ModelRegistry
from src.ml.training_pipeline import TrainingPipeline


def _make_ohlcv(n: int = 240) -> pd.DataFrame:
    close = 100 + np.linspace(0, 12, n) + np.sin(np.linspace(0, 18, n)) * 0.5
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 25_000.0),
        }
    )


class _AlwaysUpModel:
    def fit(self, X, y, sample_weight=None, verbose=False):
        return self

    def predict(self, X):
        # DirectionEncoder class 2 == "up"
        return np.full(len(X), 2, dtype=int)


def test_walk_forward_validation_uses_prepare_output_without_unpack_errors(tmp_path, monkeypatch):
    featured = engineer_features(_make_ohlcv(), include_target=False)
    featured = featured.dropna(axis=1, how="all").ffill().bfill()
    feature_data = {"AAPL": featured}

    pipeline = TrainingPipeline(
        registry=ModelRegistry(models_dir=str(tmp_path / "models")),
        governance=ModelGovernance(governance_dir=str(tmp_path / "governance")),
        min_training_samples=100,
    )

    X, y, feature_cols, _, _, _ = pipeline._prepare_training_data(feature_data)
    assert len(X) == len(y)
    assert len(feature_cols) > 0

    monkeypatch.setattr("xgboost.XGBClassifier", lambda **kwargs: _AlwaysUpModel())
    wf = pipeline._walk_forward_validation(feature_data, feature_cols, n_splits=2)

    assert "sharpe" in wf
    assert "total_return" in wf
    assert "n_trades" in wf
    assert wf["n_trades"] >= 0


def test_parameter_sweep_falls_back_when_vectorbt_alignment_breaks(monkeypatch):
    sentinel = pd.DataFrame(
        [{"fast_window": 12, "slow_window": 26, "total_return": 0.01, "sharpe_ratio": 1.0, "max_drawdown": 0.02, "win_rate": 0.6, "total_trades": 10}]
    )

    monkeypatch.setattr(vbt_adapter, "_VBT_AVAILABLE", True)
    monkeypatch.setattr(
        vbt_adapter,
        "vectorbt_momentum_backtest",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("Index at position 0 could not be aligned")),
    )
    monkeypatch.setattr(vbt_adapter, "_numpy_parameter_sweep", lambda *args, **kwargs: sentinel)

    df = _make_ohlcv(120)
    out = vbt_adapter.vectorbt_parameter_sweep(df, fast_range=range(12, 14), slow_range=range(26, 28))
    assert out.equals(sentinel)
