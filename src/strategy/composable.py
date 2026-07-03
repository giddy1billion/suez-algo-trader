"""
Composable Strategy Framework — Mix-and-match modular components for building strategies.

Components:
- EntryModel: Generates entry signals
- ExitModel: Determines exit conditions
- FilterModel: Gates signals through market filters
- SizingModel: Position sizing logic
- StopModel: Stop-loss and take-profit calculation

Usage:
    from src.strategy.composable import ComposableStrategy, momentum_preset

    strategy = momentum_preset(symbols=["EURUSD", "GBPUSD"])
    signals = strategy.generate_signals(data)
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy, Signal, TradeSignal
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Component Interfaces
# =============================================================================


class EntryModel(ABC):
    """Generates entry signals from OHLCV data."""

    @abstractmethod
    def should_enter(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Evaluate whether to enter a position.

        Args:
            df: OHLCV DataFrame with at least columns: open, high, low, close, volume
            symbol: The trading symbol

        Returns:
            Signal (BUY/SELL/STRONG_BUY/STRONG_SELL) or None/HOLD if no entry.
        """
        pass


class ExitModel(ABC):
    """Determines when to exit an existing position."""

    @abstractmethod
    def should_exit(self, df: pd.DataFrame, symbol: str, position: dict) -> Optional[Signal]:
        """
        Evaluate whether to exit a position.

        Args:
            df: OHLCV DataFrame
            symbol: The trading symbol
            position: Dict with keys like 'side', 'entry_price', 'bars_held', 'entry_time'

        Returns:
            Signal indicating exit direction, or None to hold.
        """
        pass


class FilterModel(ABC):
    """Filters out bad signals. Returns True to allow, False to block."""

    @abstractmethod
    def allow(self, df: pd.DataFrame, signal: Signal) -> bool:
        """
        Determine if market conditions allow trading.

        Args:
            df: OHLCV DataFrame
            signal: The proposed entry signal

        Returns:
            True to allow the signal through, False to block it.
        """
        pass


class SizingModel(ABC):
    """Determines position size."""

    @abstractmethod
    def size(self, signal: Signal, price: float, portfolio_value: float, atr: float) -> float:
        """
        Calculate position size.

        Args:
            signal: The trade signal
            price: Current price
            portfolio_value: Total portfolio value
            atr: Current ATR value

        Returns:
            Position size as a fraction of portfolio (0.0 to 1.0).
        """
        pass


class StopModel(ABC):
    """Calculates stop loss and take profit levels."""

    @abstractmethod
    def levels(self, df: pd.DataFrame, side: str, entry_price: float) -> tuple:
        """
        Calculate stop-loss and take-profit.

        Args:
            df: OHLCV DataFrame
            side: 'buy' or 'sell'
            entry_price: The entry price

        Returns:
            Tuple of (stop_loss, take_profit) — either may be None.
        """
        pass


# =============================================================================
# Helper: Indicator Computations (shared across components)
# =============================================================================


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = _atr(df, period)
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_safe)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_safe)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(window=period).mean()
    return adx


# =============================================================================
# Entry Models
# =============================================================================


class EMAEntryModel(EntryModel):
    """EMA crossover entry — buy on fast crossing above slow, sell on cross below."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def should_enter(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(df) < self.slow + 2:
            return None

        ema_fast = _ema(df['close'], self.fast)
        ema_slow = _ema(df['close'], self.slow)

        curr_fast, curr_slow = ema_fast.iloc[-1], ema_slow.iloc[-1]
        prev_fast, prev_slow = ema_fast.iloc[-2], ema_slow.iloc[-2]

        # Bullish crossover
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            return Signal.BUY
        # Bearish crossover
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            return Signal.SELL

        return Signal.HOLD


class RSIEntryModel(EntryModel):
    """RSI reversal entry — buy when RSI exits oversold, sell when exits overbought."""

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def should_enter(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(df) < self.period + 2:
            return None

        rsi = _rsi(df['close'], self.period)
        curr_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            return None

        # Exiting oversold
        if prev_rsi < self.oversold and curr_rsi >= self.oversold:
            return Signal.BUY
        # Exiting overbought
        if prev_rsi > self.overbought and curr_rsi <= self.overbought:
            return Signal.SELL

        return Signal.HOLD


class MACDEntryModel(EntryModel):
    """MACD histogram direction change entry."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def should_enter(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(df) < self.slow + self.signal_period + 2:
            return None

        ema_fast = _ema(df['close'], self.fast)
        ema_slow = _ema(df['close'], self.slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, self.signal_period)
        histogram = macd_line - signal_line

        curr_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        if pd.isna(curr_hist) or pd.isna(prev_hist):
            return None

        # Histogram crosses above zero
        if curr_hist > 0 and prev_hist <= 0:
            return Signal.BUY
        # Histogram crosses below zero
        if curr_hist < 0 and prev_hist >= 0:
            return Signal.SELL

        return Signal.HOLD


class MLEntryModel(EntryModel):
    """Delegates entry decision to a pre-trained ML model (e.g., XGBoost)."""

    def __init__(self, model_path: str, threshold: float = 0.6):
        self.model_path = model_path
        self.threshold = threshold
        self._model = None

    def _load_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            import pickle
            with open(self.model_path, 'rb') as f:
                self._model = pickle.load(f)
            logger.info("ml_model_loaded", path=self.model_path)

    def should_enter(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        if len(df) < 50:
            return None

        self._load_model()

        # Build feature vector from latest bar
        features = self._extract_features(df)
        if features is None:
            return None

        prediction = self._model.predict_proba(features.reshape(1, -1))[0]

        # Assume classes: [sell, hold, buy] or binary [down, up]
        if len(prediction) == 3:
            if prediction[2] > self.threshold:
                return Signal.BUY
            elif prediction[0] > self.threshold:
                return Signal.SELL
        elif len(prediction) == 2:
            if prediction[1] > self.threshold:
                return Signal.BUY
            elif prediction[0] > self.threshold:
                return Signal.SELL

        return Signal.HOLD

    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract standard features for ML prediction."""
        try:
            close = df['close']
            features = []
            # Returns
            for period in [1, 5, 10, 20]:
                features.append(close.pct_change(period).iloc[-1])
            # RSI
            features.append(_rsi(close, 14).iloc[-1])
            # Volatility
            features.append(close.pct_change().rolling(20).std().iloc[-1])
            # Volume ratio
            if 'volume' in df.columns:
                vol_ma = df['volume'].rolling(20).mean()
                features.append((df['volume'].iloc[-1] / vol_ma.iloc[-1]) if vol_ma.iloc[-1] > 0 else 1.0)
            else:
                features.append(1.0)

            arr = np.array(features, dtype=np.float64)
            if np.any(np.isnan(arr)):
                return None
            return arr
        except Exception:
            return None


# =============================================================================
# Exit Models
# =============================================================================


class EMAExitModel(ExitModel):
    """Exit when EMA crosses against the position direction."""

    def __init__(self, fast: int = 12, slow: int = 26):
        self.fast = fast
        self.slow = slow

    def should_exit(self, df: pd.DataFrame, symbol: str, position: dict) -> Optional[Signal]:
        if len(df) < self.slow + 2:
            return None

        ema_fast = _ema(df['close'], self.fast)
        ema_slow = _ema(df['close'], self.slow)

        side = position.get('side', 'buy')

        # Exit long if fast crosses below slow
        if side == 'buy' and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return Signal.SELL
        # Exit short if fast crosses above slow
        if side == 'sell' and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return Signal.BUY

        return None


class RSIExitModel(ExitModel):
    """Exit when RSI reverts to a neutral level."""

    def __init__(self, period: int = 14, exit_level: float = 50):
        self.period = period
        self.exit_level = exit_level

    def should_exit(self, df: pd.DataFrame, symbol: str, position: dict) -> Optional[Signal]:
        if len(df) < self.period + 2:
            return None

        rsi = _rsi(df['close'], self.period)
        curr_rsi = rsi.iloc[-1]

        if pd.isna(curr_rsi):
            return None

        side = position.get('side', 'buy')

        # Exit long when RSI crosses above exit level (mean reverted)
        if side == 'buy' and curr_rsi >= self.exit_level:
            return Signal.SELL
        # Exit short when RSI drops below exit level
        if side == 'sell' and curr_rsi <= self.exit_level:
            return Signal.BUY

        return None


class TimeExitModel(ExitModel):
    """Exit after holding for N bars."""

    def __init__(self, max_bars: int = 20):
        self.max_bars = max_bars

    def should_exit(self, df: pd.DataFrame, symbol: str, position: dict) -> Optional[Signal]:
        bars_held = position.get('bars_held', 0)
        if bars_held >= self.max_bars:
            side = position.get('side', 'buy')
            return Signal.SELL if side == 'buy' else Signal.BUY
        return None


class TrailingStopExitModel(ExitModel):
    """ATR-based trailing stop exit."""

    def __init__(self, atr_period: int = 14, atr_mult: float = 2.0):
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    def should_exit(self, df: pd.DataFrame, symbol: str, position: dict) -> Optional[Signal]:
        if len(df) < self.atr_period + 2:
            return None

        atr = _atr(df, self.atr_period)
        curr_atr = atr.iloc[-1]
        if pd.isna(curr_atr):
            return None

        side = position.get('side', 'buy')
        entry_price = position.get('entry_price', df['close'].iloc[-1])
        current_price = df['close'].iloc[-1]

        if side == 'buy':
            # Track highest price since entry
            highest = df['high'].iloc[-self.atr_period:].max()
            trailing_stop = highest - (curr_atr * self.atr_mult)
            if current_price <= trailing_stop:
                return Signal.SELL
        else:
            lowest = df['low'].iloc[-self.atr_period:].min()
            trailing_stop = lowest + (curr_atr * self.atr_mult)
            if current_price >= trailing_stop:
                return Signal.BUY

        return None


# =============================================================================
# Filter Models
# =============================================================================


class ADXFilter(FilterModel):
    """Only allow trades when ADX indicates a trending market."""

    def __init__(self, period: int = 14, min_adx: float = 20):
        self.period = period
        self.min_adx = min_adx

    def allow(self, df: pd.DataFrame, signal: Signal) -> bool:
        if len(df) < self.period * 3:
            return False

        adx = _adx(df, self.period)
        curr_adx = adx.iloc[-1]

        if pd.isna(curr_adx):
            return False

        return curr_adx >= self.min_adx


class VolumeFilter(FilterModel):
    """Only allow trades with volume confirmation (above MA)."""

    def __init__(self, ma_period: int = 20, min_spike: float = 1.5):
        self.ma_period = ma_period
        self.min_spike = min_spike

    def allow(self, df: pd.DataFrame, signal: Signal) -> bool:
        if 'volume' not in df.columns or len(df) < self.ma_period + 1:
            return True  # Pass through if no volume data

        vol_ma = df['volume'].rolling(window=self.ma_period).mean().iloc[-1]
        if pd.isna(vol_ma) or vol_ma == 0:
            return True

        ratio = df['volume'].iloc[-1] / vol_ma
        return ratio >= self.min_spike


class RegimeFilter(FilterModel):
    """
    Only allow trades in favorable market regimes.
    Trending regime for momentum signals, mean-reverting for RSI signals.
    Uses ADX + Hurst exponent approximation.
    """

    def __init__(self, adx_period: int = 14, trending_threshold: float = 25):
        self.adx_period = adx_period
        self.trending_threshold = trending_threshold

    def allow(self, df: pd.DataFrame, signal: Signal) -> bool:
        if len(df) < self.adx_period * 3:
            return False

        adx = _adx(df, self.adx_period)
        curr_adx = adx.iloc[-1]

        if pd.isna(curr_adx):
            return False

        is_trending = curr_adx >= self.trending_threshold

        # Momentum signals need trending markets
        if signal in (Signal.BUY, Signal.STRONG_BUY, Signal.SELL, Signal.STRONG_SELL):
            return is_trending

        return True


class TimeFilter(FilterModel):
    """Only allow trades during specified hours (UTC)."""

    def __init__(self, allowed_hours: tuple = (9, 16)):
        self.start_hour = allowed_hours[0]
        self.end_hour = allowed_hours[1]

    def allow(self, df: pd.DataFrame, signal: Signal) -> bool:
        if df.index.dtype == 'datetime64[ns]' or hasattr(df.index, 'hour'):
            try:
                last_hour = df.index[-1].hour
                return self.start_hour <= last_hour < self.end_hour
            except (AttributeError, TypeError):
                pass
        # If index isn't datetime, pass through
        return True


# =============================================================================
# Sizing Models
# =============================================================================


class FixedRiskSizing(SizingModel):
    """Fixed percentage risk per trade — size based on distance to stop."""

    def __init__(self, risk_pct: float = 0.02):
        self.risk_pct = risk_pct

    def size(self, signal: Signal, price: float, portfolio_value: float, atr: float) -> float:
        if price <= 0 or atr <= 0:
            return 0.0

        risk_amount = portfolio_value * self.risk_pct
        # Position size = risk / (ATR * 2) as fraction of portfolio
        stop_distance = atr * 2.0
        position_value = risk_amount / (stop_distance / price)
        return min(position_value / portfolio_value, 0.25)  # Cap at 25%


class KellySizing(SizingModel):
    """Half-Kelly criterion sizing."""

    def __init__(self, win_rate: float = 0.55, avg_win: float = 0.03,
                 avg_loss: float = 0.02, fraction: float = 0.5):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.fraction = fraction

    def size(self, signal: Signal, price: float, portfolio_value: float, atr: float) -> float:
        if self.avg_loss <= 0:
            return 0.0

        # Kelly formula: f* = (p*b - q) / b
        # where p=win_rate, q=1-p, b=avg_win/avg_loss
        b = self.avg_win / self.avg_loss
        q = 1 - self.win_rate
        kelly = (self.win_rate * b - q) / b

        # Apply fraction (half-kelly by default)
        sized = kelly * self.fraction

        # Clamp to [0, 0.25]
        return max(0.0, min(sized, 0.25))


class VolatilitySizing(SizingModel):
    """Volatility-targeted position sizing."""

    def __init__(self, target_vol: float = 0.15):
        self.target_vol = target_vol

    def size(self, signal: Signal, price: float, portfolio_value: float, atr: float) -> float:
        if price <= 0 or atr <= 0:
            return 0.0

        # Annualized vol approximation from ATR
        # ATR is roughly daily range; annualize with sqrt(252)
        daily_vol = atr / price
        annualized_vol = daily_vol * np.sqrt(252)

        if annualized_vol <= 0:
            return 0.0

        # Target position = target_vol / asset_vol
        target_weight = self.target_vol / annualized_vol
        return max(0.0, min(target_weight, 0.25))


# =============================================================================
# Stop Models
# =============================================================================


class ATRStop(StopModel):
    """ATR-based stop loss and take profit."""

    def __init__(self, period: int = 14, sl_mult: float = 2.0, tp_mult: float = 3.0):
        self.period = period
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult

    def levels(self, df: pd.DataFrame, side: str, entry_price: float) -> tuple:
        if len(df) < self.period + 1:
            return (None, None)

        atr = _atr(df, self.period)
        curr_atr = atr.iloc[-1]

        if pd.isna(curr_atr) or curr_atr <= 0:
            return (None, None)

        if side == 'buy':
            sl = entry_price - (curr_atr * self.sl_mult)
            tp = entry_price + (curr_atr * self.tp_mult)
        else:
            sl = entry_price + (curr_atr * self.sl_mult)
            tp = entry_price - (curr_atr * self.tp_mult)

        return (sl, tp)


class PercentStop(StopModel):
    """Fixed percentage stop loss and take profit."""

    def __init__(self, sl_pct: float = 0.02, tp_pct: float = 0.04):
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct

    def levels(self, df: pd.DataFrame, side: str, entry_price: float) -> tuple:
        if side == 'buy':
            sl = entry_price * (1 - self.sl_pct)
            tp = entry_price * (1 + self.tp_pct)
        else:
            sl = entry_price * (1 + self.sl_pct)
            tp = entry_price * (1 - self.tp_pct)

        return (sl, tp)


class SwingStop(StopModel):
    """Stop loss at previous swing high/low."""

    def __init__(self, lookback: int = 20, tp_ratio: float = 2.0):
        self.lookback = lookback
        self.tp_ratio = tp_ratio

    def levels(self, df: pd.DataFrame, side: str, entry_price: float) -> tuple:
        if len(df) < self.lookback + 1:
            return (None, None)

        window = df.iloc[-(self.lookback + 1):-1]  # Exclude current bar

        if side == 'buy':
            sl = window['low'].min()
            risk = entry_price - sl
            tp = entry_price + (risk * self.tp_ratio) if risk > 0 else None
        else:
            sl = window['high'].max()
            risk = sl - entry_price
            tp = entry_price - (risk * self.tp_ratio) if risk > 0 else None

        return (sl, tp)


# =============================================================================
# ComposableStrategy
# =============================================================================


class ComposableStrategy(BaseStrategy):
    """
    Strategy built from composable components.
    Compatible with the existing BaseStrategy interface and ExecutionEngine.
    """

    def __init__(
        self,
        name: str,
        symbols: list,
        entry: EntryModel,
        exit_model: Optional[ExitModel] = None,
        filters: Optional[list] = None,
        sizing: Optional[SizingModel] = None,
        stop: Optional[StopModel] = None,
        timeframe: str = "1Hour",
        lookback: int = 200,
    ):
        super().__init__(name=name, symbols=symbols, timeframe=timeframe, lookback=lookback)
        self.entry = entry
        self.exit_model = exit_model
        self.filters = filters or []
        self.sizing = sizing
        self.stop = stop

        logger.info(
            "composable_strategy_created",
            name=name,
            entry=entry.__class__.__name__,
            exit_model=exit_model.__class__.__name__ if exit_model else None,
            filters=[f.__class__.__name__ for f in self.filters],
            sizing=sizing.__class__.__name__ if sizing else None,
            stop=stop.__class__.__name__ if stop else None,
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        No-op for composable strategies — each component computes its own indicators.
        Returns the DataFrame unchanged.
        """
        return df

    def generate_signals(self, data: dict) -> list:
        """
        Compose components into trade signals.

        Pipeline: entry → filters → stops → signal assembly.
        """
        signals = []

        for symbol, df in data.items():
            if len(df) < 50:
                continue

            try:
                signal = self._process_symbol(symbol, df)
                if signal and signal.is_actionable:
                    signals.append(signal)
            except Exception as e:
                logger.error("composable_signal_error", symbol=symbol, error=str(e))

        return signals

    def _process_symbol(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Process a single symbol through the component pipeline."""
        # 1. Generate entry signal
        entry_signal = self.entry.should_enter(df, symbol)
        if entry_signal is None or entry_signal == Signal.HOLD:
            return None

        # 2. Apply all filters
        for f in self.filters:
            if not f.allow(df, entry_signal):
                logger.debug(
                    "signal_filtered",
                    symbol=symbol,
                    filter=f.__class__.__name__,
                    signal=entry_signal.name,
                )
                return None

        # 3. Determine side and price
        side = "buy" if entry_signal.value > 0 else "sell"
        price = df['close'].iloc[-1]

        # 4. Calculate stops
        sl, tp = (None, None)
        if self.stop:
            sl, tp = self.stop.levels(df, side, price)

        # 5. Calculate confidence from signal strength and sizing
        confidence = self._compute_confidence(entry_signal, df)

        return TradeSignal(
            symbol=symbol,
            signal=entry_signal,
            confidence=confidence,
            price=price,
            stop_loss=sl,
            take_profit=tp,
            reason=f"{self.name}:{self.entry.__class__.__name__}",
            indicators=self._gather_indicators(df),
        )

    def should_exit(self, symbol: str, position: dict, current_price: float) -> Optional[TradeSignal]:
        """
        Check exit conditions using the exit model.
        Compatible with BaseStrategy.should_exit interface.
        """
        if self.exit_model is None:
            return None

        # We need a DataFrame for the exit model — caller should provide via position
        df = position.get('_dataframe')
        if df is None:
            return None

        exit_signal = self.exit_model.should_exit(df, symbol, position)
        if exit_signal is None:
            return None

        return TradeSignal(
            symbol=symbol,
            signal=exit_signal,
            confidence=0.8,
            price=current_price,
            reason=f"{self.name}:exit:{self.exit_model.__class__.__name__}",
        )

    def _compute_confidence(self, signal: Signal, df: pd.DataFrame) -> float:
        """Compute confidence score based on signal strength."""
        base = 0.5
        if signal in (Signal.STRONG_BUY, Signal.STRONG_SELL):
            base = 0.8
        elif signal in (Signal.BUY, Signal.SELL):
            base = 0.65

        # Boost confidence if all filters passed (they already did at this point)
        filter_bonus = min(len(self.filters) * 0.05, 0.15)
        return min(base + filter_bonus, 1.0)

    def _gather_indicators(self, df: pd.DataFrame) -> dict:
        """Gather basic indicator values for the signal metadata."""
        indicators = {}
        close = df['close']

        try:
            rsi = _rsi(close, 14)
            indicators['rsi'] = round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else None
        except Exception:
            pass

        try:
            atr = _atr(df, 14)
            indicators['atr'] = round(float(atr.iloc[-1]), 4) if not pd.isna(atr.iloc[-1]) else None
        except Exception:
            pass

        return indicators


# =============================================================================
# Factory Presets
# =============================================================================


def momentum_preset(symbols: list, **kwargs) -> ComposableStrategy:
    """Pre-configured momentum strategy using composable modules."""
    return ComposableStrategy(
        name="momentum_v2",
        symbols=symbols,
        entry=EMAEntryModel(fast=12, slow=26),
        exit_model=EMAExitModel(fast=12, slow=26),
        filters=[ADXFilter(min_adx=20), VolumeFilter(min_spike=1.5)],
        sizing=FixedRiskSizing(risk_pct=0.02),
        stop=ATRStop(sl_mult=2.0, tp_mult=3.0),
        **kwargs,
    )


def mean_reversion_preset(symbols: list, **kwargs) -> ComposableStrategy:
    """Pre-configured mean reversion strategy using composable modules."""
    return ComposableStrategy(
        name="mean_reversion_v2",
        symbols=symbols,
        entry=RSIEntryModel(period=14, oversold=30, overbought=70),
        exit_model=RSIExitModel(period=14, exit_level=50),
        filters=[VolumeFilter(min_spike=1.2)],
        sizing=VolatilitySizing(target_vol=0.10),
        stop=PercentStop(sl_pct=0.02, tp_pct=0.04),
        **kwargs,
    )


def ml_momentum_preset(symbols: list, model_path: str, **kwargs) -> ComposableStrategy:
    """ML entry with momentum-style filters and stops."""
    return ComposableStrategy(
        name="ml_momentum",
        symbols=symbols,
        entry=MLEntryModel(model_path=model_path, threshold=0.6),
        exit_model=TrailingStopExitModel(atr_mult=2.5),
        filters=[ADXFilter(min_adx=20), VolumeFilter(min_spike=1.3)],
        sizing=KellySizing(win_rate=0.55, avg_win=0.03, avg_loss=0.02, fraction=0.5),
        stop=ATRStop(sl_mult=2.0, tp_mult=4.0),
        **kwargs,
    )
