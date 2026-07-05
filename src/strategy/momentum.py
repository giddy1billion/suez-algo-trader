"""
Momentum Strategy — Trend-following based on multiple technical indicators.
Combines RSI, MACD, EMA crossovers, and volume confirmation.
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.strategy.base import BaseStrategy, LegacyTradeSignal, Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Multi-indicator momentum strategy:
    - EMA crossover (fast/slow) for trend direction
    - RSI for overbought/oversold confirmation
    - MACD for momentum strength
    - Volume spike for confirmation
    - ATR for dynamic stop-loss placement
    """

    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1Hour",
        lookback: int = 200,
        fast_ema: int = 12,
        slow_ema: int = 26,
        signal_ema: int = 9,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        atr_period: int = 14,
        atr_sl_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
        volume_ma_period: int = 20,
        volume_spike_threshold: float = 1.5,
        min_confirming_indicators: int = 2,
    ):
        super().__init__(name="momentum", symbols=symbols, timeframe=timeframe, lookback=lookback)

        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_ema = signal_ema
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.volume_ma_period = volume_ma_period
        self.volume_spike_threshold = volume_spike_threshold
        self.min_confirming_indicators = min_confirming_indicators

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the momentum strategy."""
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()

        # MACD
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['macd_signal'] = df['macd'].ewm(span=self.signal_ema, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.atr_period).mean()

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)

        # EMA crossover signals
        df['ema_cross'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        df['ema_cross_prev'] = df['ema_cross'].shift(1)

        return df

    def generate_signals(self, data: dict[str, pd.DataFrame]) -> list:
        """Generate momentum-based trade signals for each symbol."""
        signals = []

        for symbol, df in data.items():
            required = {'close', 'high', 'low', 'volume'}
            if not required.issubset(df.columns):
                logger.warning("momentum.missing_columns", symbol=symbol, missing=list(required - set(df.columns)))
                continue

            if len(df) < self.slow_ema + 10:
                continue

            df = self.calculate_indicators(df)
            signal = self._evaluate_symbol(symbol, df)
            if signal and signal.is_actionable:
                signals.append(signal)

        return signals

    def _evaluate_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[LegacyTradeSignal]:
        """Evaluate a single symbol and return a signal."""
        if len(df) < 2:
            return None
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price = latest['close']

        # Skip if indicators aren't ready
        if pd.isna(latest.get('rsi')) or pd.isna(latest.get('atr')):
            return None

        score = 0
        reasons = []
        confidence_factors = []

        # 1. EMA Crossover (strong signal)
        if latest['ema_cross'] == 1 and latest['ema_cross_prev'] == -1:
            score += 2
            reasons.append("EMA bullish crossover")
            confidence_factors.append(0.8)
        elif latest['ema_cross'] == -1 and latest['ema_cross_prev'] == 1:
            score -= 2
            reasons.append("EMA bearish crossover")
            confidence_factors.append(0.8)
        elif latest['ema_fast'] > latest['ema_slow']:
            score += 1
            reasons.append("Above slow EMA")
            confidence_factors.append(0.5)
        else:
            score -= 1
            reasons.append("Below slow EMA")
            confidence_factors.append(0.5)

        # 2. RSI
        rsi = latest['rsi']
        if rsi < self.rsi_oversold:
            score += 1
            reasons.append(f"RSI oversold ({rsi:.0f})")
            confidence_factors.append(0.7)
        elif rsi > self.rsi_overbought:
            score -= 1
            reasons.append(f"RSI overbought ({rsi:.0f})")
            confidence_factors.append(0.7)

        # 3. MACD
        if latest['macd_hist'] > 0 and prev['macd_hist'] <= 0:
            score += 1
            reasons.append("MACD histogram turned positive")
            confidence_factors.append(0.6)
        elif latest['macd_hist'] < 0 and prev['macd_hist'] >= 0:
            score -= 1
            reasons.append("MACD histogram turned negative")
            confidence_factors.append(0.6)

        # 4. Volume confirmation
        vol_ratio = latest.get('volume_ratio', 1.0)
        if not pd.isna(vol_ratio) and vol_ratio >= self.volume_spike_threshold:
            # Volume confirms the direction
            if score > 0:
                confidence_factors.append(0.8)
                reasons.append(f"Volume spike ({vol_ratio:.1f}x)")
            elif score < 0:
                confidence_factors.append(0.8)
                reasons.append(f"Volume spike confirms sell ({vol_ratio:.1f}x)")

        # Calculate confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.0

        # Require minimum confirming indicators for a BUY/SELL signal.
        # A single weak indicator (e.g., just "below slow EMA") should not
        # generate actionable signals — it produces score=±1 at confidence=0.5
        # which fires every cycle for every symbol, creating signal spam.
        # For crypto (min_confirming_indicators=1), single strong signals are allowed.
        confirming_count = len(confidence_factors)
        if confirming_count < self.min_confirming_indicators and abs(score) <= 1:
            signal = Signal.HOLD
        elif score >= 3:
            signal = Signal.STRONG_BUY
        elif score >= 1:
            signal = Signal.BUY
        elif score <= -3:
            signal = Signal.STRONG_SELL
        elif score <= -1:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        # Calculate stop-loss and take-profit using ATR
        atr = latest['atr'] if not pd.isna(latest['atr']) else price * 0.02
        if signal in (Signal.BUY, Signal.STRONG_BUY):
            stop_loss = price - (atr * self.atr_sl_multiplier)
            take_profit = price + (atr * self.atr_tp_multiplier)
        elif signal in (Signal.SELL, Signal.STRONG_SELL):
            stop_loss = price + (atr * self.atr_sl_multiplier)
            take_profit = price - (atr * self.atr_tp_multiplier)
        else:
            stop_loss = None
            take_profit = None

        return LegacyTradeSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=" | ".join(reasons),
            indicators={
                "rsi": round(rsi, 2),
                "macd_hist": round(latest['macd_hist'], 4),
                "ema_fast": round(latest['ema_fast'], 4),
                "ema_slow": round(latest['ema_slow'], 4),
                "atr": round(atr, 4),
                "volume_ratio": round(vol_ratio, 2) if not pd.isna(vol_ratio) else None,
                "score": score,
            },
        )
