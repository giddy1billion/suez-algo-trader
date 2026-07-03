"""
ML Feature Engineering — Data Leakage Regression Tests.

FEATURE GENERATION TIMELINE PROOF:
===================================
All features in src/ml/features.py are computed using ONLY backward-looking operations:
  1. rolling(window, min_periods=N) — uses only past N bars
  2. ewm(span=N, adjust=False) — exponential decay of past values only
  3. shift(+N) or shift(1) — references PREVIOUS bars (positive shift = look back)
  4. pct_change(N) — difference from N bars AGO
  5. diff() — difference from previous bar
  6. cumsum() on past deltas

The ONLY forward-looking operation is TARGET generation (line ~577):
  close.shift(-forward_bars) — explicitly gated behind `if include_target:`

This means:
  - Features at bar T depend ONLY on bars [0..T]
  - Appending new bars T+1..T+K CANNOT change features at bars [0..T]
  - Target columns are NEVER present unless explicitly requested

These tests enforce these invariants as a regression gate.
"""

import re
import inspect
import numpy as np
import pandas as pd
import pytest

from src.ml.features import engineer_features, MINIMUM_BARS_REQUIRED


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate n bars of synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n) * 0.3)
    low = close - np.abs(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    volume = rng.integers(100_000, 5_000_000, size=n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestNoFutureLeakage:
    """Verify the feature pipeline never looks into the future."""

    def test_features_contain_no_future_shift(self):
        """Parse features.py source: no .shift(-N) before the TARGET section."""
        import src.ml.features as features_module

        source = inspect.getsource(features_module)

        # Find the TARGET section marker
        target_marker = "TARGET (uses look-ahead"
        target_idx = source.find(target_marker)
        assert target_idx > 0, "Could not find TARGET section marker in source"

        # Only inspect code BEFORE the target section
        feature_section = source[:target_idx]

        # Find all shift() calls with negative arguments
        # Pattern: .shift(-<number>) — any negative shift is future leakage
        negative_shifts = re.findall(r'\.shift\(\s*-\s*\d+', feature_section)

        assert negative_shifts == [], (
            f"Found future-looking shift() in feature section: {negative_shifts}"
        )

    def test_feature_values_unchanged_by_future_data(self):
        """Features for first 200 bars must be identical whether or not future bars exist."""
        # Generate 250 bars, then slice to get matching first-200
        df_250 = _make_ohlcv(250, seed=123)
        df_200 = df_250.iloc[:200].copy()

        features_200 = engineer_features(df_200, include_target=False)
        features_250 = engineer_features(df_250, include_target=False)

        # The first 200 rows of features must match exactly
        cols_200 = features_200.columns.tolist()
        cols_250 = features_250.columns.tolist()

        # Same feature set
        assert set(cols_200) == set(cols_250), (
            f"Feature columns differ: extra={set(cols_250) - set(cols_200)}, "
            f"missing={set(cols_200) - set(cols_250)}"
        )

        # Compare values for first 200 rows (use shared columns)
        shared_cols = sorted(set(cols_200) & set(cols_250))
        slice_200 = features_200[shared_cols].iloc[:200].reset_index(drop=True)
        slice_250 = features_250[shared_cols].iloc[:200].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            slice_200,
            slice_250,
            check_exact=False,
            atol=1e-12,
            obj="Features must not change when future data is appended",
        )

    def test_target_column_not_in_feature_set(self):
        """'future_return' and 'target' must NOT be present when include_target=False."""
        df = _make_ohlcv(200, seed=7)
        features = engineer_features(df, include_target=False)

        assert "future_return" not in features.columns, (
            "future_return leaked into feature set with include_target=False"
        )
        assert "target" not in features.columns, (
            "target leaked into feature set with include_target=False"
        )

    def test_target_present_when_requested(self):
        """Sanity check: target columns ARE present when include_target=True."""
        df = _make_ohlcv(200, seed=7)
        features = engineer_features(df, include_target=True)

        assert "future_return" in features.columns
        assert "target" in features.columns
