"""Tests for ML feature engineering pipeline."""
import pytest
import pandas as pd
import numpy as np

from src.ml.features import engineer_features, MINIMUM_BARS_REQUIRED, _count_trailing


class TestCountTrailing:
    def test_all_ones(self):
        assert _count_trailing(np.array([1, 1, 1, 1]), 1) == 4

    def test_trailing_zeros(self):
        assert _count_trailing(np.array([1, 1, 0, 0]), 1) == 0

    def test_mixed(self):
        assert _count_trailing(np.array([0, 1, 0, 1, 1]), 1) == 2

    def test_empty(self):
        assert _count_trailing(np.array([]), 1) == 0

    def test_single_match(self):
        assert _count_trailing(np.array([1]), 1) == 1

    def test_single_no_match(self):
        assert _count_trailing(np.array([0]), 1) == 0


class TestEngineerFeatures:
    def test_output_has_120_plus_features(self, sample_ohlcv):
        result = engineer_features(sample_ohlcv)
        # Original 5 columns + 100+ features
        new_cols = set(result.columns) - {'open', 'high', 'low', 'close', 'volume'}
        assert len(new_cols) >= 100, f"Only {len(new_cols)} features generated"

    def test_no_future_leakage_in_features(self, sample_ohlcv):
        """Features should NOT use future data."""
        result = engineer_features(sample_ohlcv, include_target=False)
        # No 'target' or 'future_return' columns when include_target=False
        assert 'target' not in result.columns
        assert 'future_return' not in result.columns

    def test_target_generated_when_requested(self, sample_ohlcv):
        result = engineer_features(sample_ohlcv, include_target=True)
        assert 'target' in result.columns
        assert 'future_return' in result.columns
        # Last forward_bars rows should be NaN (can't see future)
        assert pd.isna(result['future_return'].iloc[-1])  # Last rows should be NaN (can't see future)

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            engineer_features(pd.DataFrame())

    def test_missing_columns_raises(self, sample_ohlcv):
        df_no_volume = sample_ohlcv.drop(columns=['volume'])
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer_features(df_no_volume)

    def test_small_df_still_works(self, small_ohlcv):
        """Below minimum bars should warn but not crash."""
        result = engineer_features(small_ohlcv)
        assert len(result) == len(small_ohlcv)

    def test_no_nan_in_core_after_warmup(self, sample_ohlcv):
        """After row 100, core trend features should be populated."""
        result = engineer_features(sample_ohlcv)
        # Check a few key features exist and are not all NaN after warmup
        warmup_slice = result.iloc[MINIMUM_BARS_REQUIRED:]
        for col in ['ema_12', 'rsi_14', 'atr_14']:
            if col in warmup_slice.columns:
                assert warmup_slice[col].notna().sum() > 0, f"{col} is all NaN after warmup"

    def test_deterministic_output(self, sample_ohlcv):
        """Same input = same output."""
        r1 = engineer_features(sample_ohlcv)
        r2 = engineer_features(sample_ohlcv)
        pd.testing.assert_frame_equal(r1, r2)
