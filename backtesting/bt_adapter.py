"""
Backtrader Integration -- Adapts our strategies to run on Backtrader's engine.
Provides cerebro setup, data feeds from Alpaca, and performance analyzers.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import backtrader as bt
    _BT_AVAILABLE = True
except ImportError:
    _BT_AVAILABLE = False
    bt = None


def _require_backtrader():
    if not _BT_AVAILABLE:
        raise ImportError(
            "backtrader is not installed. Install with: pip install backtrader"
        )


# ──────────────────────────────────────────────────────────────────────────
# Classes are only defined when backtrader is installed
# ──────────────────────────────────────────────────────────────────────────

if _BT_AVAILABLE:

    class AlpacaPandasData(bt.feeds.PandasData):
        """
        Backtrader data feed from a Pandas DataFrame (as returned by our broker).
        Expects columns: open, high, low, close, volume with a DatetimeIndex.
        """
        params = (
            ('datetime', None),
            ('open', 'open'),
            ('high', 'high'),
            ('low', 'low'),
            ('close', 'close'),
            ('volume', 'volume'),
            ('openinterest', -1),
        )

    class BTMomentumStrategy(bt.Strategy):
        """
        Backtrader-compatible momentum strategy.
        Mirrors our custom MomentumStrategy logic but in Backtrader's framework.
        """
        params = (
            ('fast_ema', 12),
            ('slow_ema', 26),
            ('rsi_period', 14),
            ('rsi_oversold', 30),
            ('rsi_overbought', 70),
            ('atr_period', 14),
            ('atr_sl_mult', 2.0),
            ('atr_tp_mult', 3.0),
            ('risk_pct', 0.02),
        )

        def __init__(self):
            self.ema_fast = bt.indicators.EMA(period=self.p.fast_ema)
            self.ema_slow = bt.indicators.EMA(period=self.p.slow_ema)
            self.ema_cross = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
            self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
            self.macd = bt.indicators.MACD()
            self.atr = bt.indicators.ATR(period=self.p.atr_period)

            self.order = None
            self.entry_price = None

        def notify_order(self, order):
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.entry_price = order.executed.price
                    logger.info("bt.buy_executed", price=order.executed.price, size=order.executed.size)
                elif order.issell():
                    logger.info("bt.sell_executed", price=order.executed.price, size=order.executed.size)
            self.order = None

        def next(self):
            if self.order:
                return

            if not self.position:
                score = 0
                if self.ema_cross[0] > 0:
                    score += 2
                elif self.ema_fast[0] > self.ema_slow[0]:
                    score += 1

                if self.rsi[0] < self.p.rsi_oversold:
                    score += 1
                elif self.rsi[0] > self.p.rsi_overbought:
                    score -= 1

                if self.macd.macd[0] > self.macd.signal[0]:
                    score += 1

                if score >= 2:
                    risk_amount = self.broker.getvalue() * self.p.risk_pct
                    risk_per_share = self.atr[0] * self.p.atr_sl_mult
                    if risk_per_share > 0:
                        size = int(risk_amount / risk_per_share)
                        if size > 0:
                            self.order = self.buy(size=size)

                elif score <= -2:
                    risk_amount = self.broker.getvalue() * self.p.risk_pct
                    risk_per_share = self.atr[0] * self.p.atr_sl_mult
                    if risk_per_share > 0:
                        size = int(risk_amount / risk_per_share)
                        if size > 0:
                            self.order = self.sell(size=size)

            else:
                # Exit logic: trailing stop via ATR
                if self.position.size > 0:
                    stop = self.data.close[0] - (self.atr[0] * self.p.atr_sl_mult)
                    if self.entry_price and self.data.close[0] < stop:
                        self.order = self.close()
                elif self.position.size < 0:
                    stop = self.data.close[0] + (self.atr[0] * self.p.atr_sl_mult)
                    if self.entry_price and self.data.close[0] > stop:
                        self.order = self.close()

    class BTMeanReversionStrategy(bt.Strategy):
        """Backtrader mean reversion using Bollinger Bands."""
        params = (
            ('bb_period', 20),
            ('bb_dev', 2.0),
            ('rsi_period', 14),
            ('risk_pct', 0.02),
        )

        def __init__(self):
            self.bband = bt.indicators.BollingerBands(period=self.p.bb_period, devfactor=self.p.bb_dev)
            self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
            self.order = None

        def next(self):
            if self.order:
                return

            if not self.position:
                if self.data.close[0] < self.bband.lines.bot[0] and self.rsi[0] < 35:
                    size = int((self.broker.getvalue() * self.p.risk_pct) / self.data.close[0])
                    if size > 0:
                        self.order = self.buy(size=size)
                elif self.data.close[0] > self.bband.lines.top[0] and self.rsi[0] > 65:
                    size = int((self.broker.getvalue() * self.p.risk_pct) / self.data.close[0])
                    if size > 0:
                        self.order = self.sell(size=size)
            else:
                if self.position.size > 0 and self.data.close[0] >= self.bband.lines.mid[0]:
                    self.order = self.close()
                elif self.position.size < 0 and self.data.close[0] <= self.bband.lines.mid[0]:
                    self.order = self.close()


# ──────────────────────────────────────────────────────────────────────────
# Cerebro Runner
# ──────────────────────────────────────────────────────────────────────────

def run_backtrader_backtest(
    df: pd.DataFrame,
    strategy_class=None,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    strategy_params: dict = None,
    plot: bool = False,
) -> dict:
    """
    Run a full Backtrader backtest with analyzers.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        strategy_class: Backtrader Strategy class (defaults to BTMomentumStrategy)
        initial_cash: Starting capital
        commission: Commission per trade (0.001 = 0.1%)
        strategy_params: Override strategy parameters
        plot: Whether to plot results

    Returns:
        dict with performance metrics
    """
    _require_backtrader()

    if strategy_class is None:
        strategy_class = BTMomentumStrategy

    cerebro = bt.Cerebro()

    # Add data feed
    data = AlpacaPandasData(dataname=df)
    cerebro.adddata(data)

    # Add strategy
    params = strategy_params or {}
    cerebro.addstrategy(strategy_class, **params)

    # Broker settings
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.04)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    # Run
    results = cerebro.run()
    strat = results[0]

    # Extract metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_cash) - 1

    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    won = trade_analysis.get('won', {}).get('total', 0)
    lost = trade_analysis.get('lost', {}).get('total', 0)

    sharpe_data = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_data.get('sharperatio', 0) or 0

    dd_data = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_data.get('max', {}).get('drawdown', 0) / 100

    sqn_data = strat.analyzers.sqn.get_analysis()
    sqn = sqn_data.get('sqn', 0) or 0

    metrics = {
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'total_trades': total_trades,
        'won': won,
        'lost': lost,
        'win_rate': won / total_trades if total_trades > 0 else 0,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'sqn': sqn,
        'strategy': strategy_class.__name__,
    }

    if plot:
        cerebro.plot(style='candlestick')

    return metrics


def compare_strategies(
    df: pd.DataFrame,
    strategies: list = None,
    initial_cash: float = 10000.0,
) -> pd.DataFrame:
    """
    Compare multiple strategies on the same data.
    Returns a DataFrame with metrics for each strategy.
    """
    _require_backtrader()

    if strategies is None:
        strategies = [
            (BTMomentumStrategy, {}),
            (BTMeanReversionStrategy, {}),
        ]

    results = []
    for strat_class, params in strategies:
        metrics = run_backtrader_backtest(
            df, strategy_class=strat_class,
            initial_cash=initial_cash,
            strategy_params=params,
        )
        results.append(metrics)

    return pd.DataFrame(results)
