"""
Mean Reversion Strategy — Trades toward the mean when price deviates.
Uses Bollinger Bands, Z-score, and RSI for entry/exit signals.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.strategy.base import BaseStrategy, TradeSignal, Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy:
    - Bollinger Band squeeze and expansion for entries
    - Z-score for deviation measurement
    - RSI divergence confirmation
    - Reversion toward the SMA as the thesis
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback: int = 200,
        bb_period: int = 20,
        bb_std: float = 2.0,
        zscore_threshold: float = 2.0,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_sl_multiplier: float = 1.5,
    ):
        super().__init__(name="mean_reversion", symbols=symbols, timeframe=timeframe, lookback=lookback)

        self.bb_period = bb_period
        self.bb_std = bb_std
        self.zscore_threshold = zscore_threshold
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands, Z-score, RSI, ATR."""
        df = df.copy()

        # Simple Moving Average
        df['sma'] = df['close'].rolling(window=self.bb_period).mean()
        df['std'] = df['close'].rolling(window=self.bb_period).std()

        # Bollinger Bands
        df['bb_upper'] = df['sma'] + (df['std'] * self.bb_std)
        df['bb_lower'] = df['sma'] - (df['std'] * self.bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

        # Z-Score
        df['zscore'] = (df['close'] - df['sma']) / df['std'].replace(0, np.nan)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()

        return df

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list[TradeSignal]:
        """Generate mean reversion signals."""
        signals = []

        for symbol, df in data.items():
            if len(df) < self.bb_period + 10:
                continue

            df = self.calculate_indicators(df)
            signal = self._evaluate_symbol(symbol, df)
            if signal and signal.is_actionable:
                signals.append(signal)

        return signals

    def _evaluate_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Evaluate mean reversion opportunity for one symbol."""
        latest = df.iloc[-1]
        price = latest['close']

        if pd.isna(latest.get('zscore')) or pd.isna(latest.get('rsi')):
            return None

        zscore = latest['zscore']
        rsi = latest['rsi']
        bb_pct = latest['bb_pct']
        atr = latest['atr'] if not pd.isna(latest['atr']) else price * 0.02

        reasons = []
        confidence = 0.0

        # BUY signal: price below lower band + oversold RSI
        if zscore <= -self.zscore_threshold and rsi < 35:
            signal = Signal.STRONG_BUY if zscore <= -(self.zscore_threshold * 1.5) else Signal.BUY
            confidence = min(0.9, 0.5 + abs(zscore) * 0.15)
            reasons.append(f"Z-score={zscore:.2f} (below -{self.zscore_threshold})")
            reasons.append(f"RSI={rsi:.0f} (oversold)")
            stop_loss = price - (atr * self.atr_sl_multiplier)
            take_profit = latest['sma']  # Target the mean

        # SELL signal: price above upper band + overbought RSI
        elif zscore >= self.zscore_threshold and rsi > 65:
            signal = Signal.STRONG_SELL if zscore >= self.zscore_threshold * 1.5 else Signal.SELL
            confidence = min(0.9, 0.5 + abs(zscore) * 0.15)
            reasons.append(f"Z-score={zscore:.2f} (above +{self.zscore_threshold})")
            reasons.append(f"RSI={rsi:.0f} (overbought)")
            stop_loss = price + (atr * self.atr_sl_multiplier)
            take_profit = latest['sma']  # Target the mean

        else:
            return TradeSignal(
                symbol=symbol, signal=Signal.HOLD, confidence=0.0,
                price=price, reason="No mean reversion setup"
            )

        return TradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=" | ".join(reasons),
            indicators={
                "zscore": round(zscore, 3),
                "rsi": round(rsi, 2),
                "bb_pct": round(bb_pct, 3) if not pd.isna(bb_pct) else None,
                "sma": round(latest['sma'], 4),
                "atr": round(atr, 4),
            },
        )
