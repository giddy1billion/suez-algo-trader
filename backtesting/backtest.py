"""
Backtesting Engine â€” Run strategies against historical data to evaluate performance.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy, Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """A single trade in a backtest."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_bars: int = 0

    def __post_init__(self):
        if self.side == "buy":
            self.pnl = (self.exit_price - self.entry_price) * self.qty
            self.pnl_pct = (self.exit_price / self.entry_price) - 1
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.qty
            self.pnl_pct = (self.entry_price / self.exit_price) - 1


@dataclass
class BacktestResult:
    """Full backtest results and metrics."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        return (self.final_capital / self.initial_capital) - 1

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"BACKTEST RESULTS: {self.strategy_name}\n"
            f"{'='*60}\n"
            f"Symbol:        {self.symbol} ({self.timeframe})\n"
            f"Period:        {self.start_date} -> {self.end_date}\n"
            f"{'â”€'*60}\n"
            f"Initial:       ${self.initial_capital:,.2f}\n"
            f"Final:         ${self.final_capital:,.2f}\n"
            f"Total Return:  {self.total_return_pct:.2%}\n"
            f"{'â”€'*60}\n"
            f"Total Trades:  {self.total_trades}\n"
            f"Win Rate:      {self.win_rate:.1%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Avg Trade:     ${self.avg_trade_pnl:.2f}\n"
            f"Avg Win:       ${self.avg_win:.2f}\n"
            f"Avg Loss:      ${self.avg_loss:.2f}\n"
            f"{'â”€'*60}\n"
            f"Max Drawdown:  {self.max_drawdown:.2%}\n"
            f"Sharpe Ratio:  {self.sharpe_ratio:.3f}\n"
            f"Max Consec. L: {self.max_consecutive_losses}\n"
            f"{'='*60}\n"
        )


class Backtester:
    """
    Event-driven backtesting engine.
    Iterates through historical bars and simulates strategy execution.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,  # 0.1% per trade (Alpaca is commission-free for stocks)
        slippage_pct: float = 0.0005,   # 0.05% slippage estimate
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self, data: pd.DataFrame, symbol: str = "TEST") -> BacktestResult:
        """
        Run backtest on a single symbol's OHLCV data.

        Args:
            data: DataFrame with [open, high, low, close, volume] columns
            symbol: Symbol identifier for results

        Returns:
            BacktestResult with full metrics
        """
        capital = self.initial_capital
        position: Optional[dict] = None
        trades: list[BacktestTrade] = []
        equity_curve = [capital]

        # Need enough bars for strategy lookback
        min_bars = self.strategy.lookback
        if len(data) <= min_bars:
            logger.warning("backtest.insufficient_data", bars=len(data), needed=min_bars)
            return BacktestResult(
                strategy_name=self.strategy.name,
                symbol=symbol,
                timeframe=self.strategy.timeframe,
                start_date=str(data.index[0]) if len(data) > 0 else "",
                end_date=str(data.index[-1]) if len(data) > 0 else "",
                initial_capital=self.initial_capital,
                final_capital=capital,
            )

        # Iterate through bars
        for i in range(min_bars, len(data)):
            window = data.iloc[:i+1]
            current_bar = data.iloc[i]
            current_price = current_bar['close']
            current_time = data.index[i]

            # Generate signal on this bar
            signals = self.strategy.generate_signals({symbol: window})
            signal = signals[0] if signals else None

            # Check exit conditions for open position
            if position:
                should_exit = False
                exit_price = current_price

                # Stop loss
                if position['stop_loss']:
                    if position['side'] == 'buy' and current_bar['low'] <= position['stop_loss']:
                        should_exit = True
                        exit_price = position['stop_loss']
                    elif position['side'] == 'sell' and current_bar['high'] >= position['stop_loss']:
                        should_exit = True
                        exit_price = position['stop_loss']

                # Take profit
                if not should_exit and position['take_profit']:
                    if position['side'] == 'buy' and current_bar['high'] >= position['take_profit']:
                        should_exit = True
                        exit_price = position['take_profit']
                    elif position['side'] == 'sell' and current_bar['low'] <= position['take_profit']:
                        should_exit = True
                        exit_price = position['take_profit']

                # Strategy exit signal
                if not should_exit and signal:
                    if position['side'] == 'buy' and signal.signal in (Signal.SELL, Signal.STRONG_SELL):
                        should_exit = True
                    elif position['side'] == 'sell' and signal.signal in (Signal.BUY, Signal.STRONG_BUY):
                        should_exit = True

                if should_exit:
                    # Apply slippage
                    if position['side'] == 'buy':
                        exit_price *= (1 - self.slippage_pct)
                    else:
                        exit_price *= (1 + self.slippage_pct)

                    trade = BacktestTrade(
                        symbol=symbol,
                        side=position['side'],
                        entry_price=position['entry_price'],
                        exit_price=exit_price,
                        qty=position['qty'],
                        entry_time=position['entry_time'],
                        exit_time=current_time,
                        hold_bars=i - position['entry_bar'],
                    )
                    trades.append(trade)
                    capital += trade.pnl - (trade.pnl * self.commission_pct)
                    position = None

            # Enter new position (only if no existing position)
            elif signal and signal.is_actionable:
                if signal.signal in (Signal.BUY, Signal.STRONG_BUY):
                    side = 'buy'
                elif signal.signal in (Signal.SELL, Signal.STRONG_SELL):
                    side = 'sell'
                else:
                    equity_curve.append(capital)
                    continue

                # Position sizing: risk 2% of capital
                risk_pct = 0.02
                if signal.stop_loss:
                    risk_per_share = abs(current_price - signal.stop_loss)
                    if risk_per_share > 0:
                        qty = (capital * risk_pct) / risk_per_share
                    else:
                        qty = (capital * 0.1) / current_price
                else:
                    qty = (capital * 0.1) / current_price

                # Apply slippage to entry
                entry_price = current_price * (1 + self.slippage_pct) if side == 'buy' else current_price * (1 - self.slippage_pct)

                position = {
                    'side': side,
                    'entry_price': entry_price,
                    'qty': qty,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'entry_time': current_time,
                    'entry_bar': i,
                }

            # Track equity
            if position:
                if position['side'] == 'buy':
                    unrealized = (current_price - position['entry_price']) * position['qty']
                else:
                    unrealized = (position['entry_price'] - current_price) * position['qty']
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)

        # Close any remaining position at last price
        if position:
            last_price = data.iloc[-1]['close']
            trade = BacktestTrade(
                symbol=symbol,
                side=position['side'],
                entry_price=position['entry_price'],
                exit_price=last_price,
                qty=position['qty'],
                entry_time=position['entry_time'],
                exit_time=data.index[-1],
                hold_bars=len(data) - position['entry_bar'],
            )
            trades.append(trade)
            capital += trade.pnl

        # Calculate metrics
        return self._calculate_metrics(symbol, data, capital, trades, equity_curve)

    def _calculate_metrics(
        self, symbol: str, data: pd.DataFrame,
        final_capital: float, trades: list[BacktestTrade], equity_curve: list[float]
    ) -> BacktestResult:
        """Calculate all performance metrics from trades."""

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio (annualized, calendar-aware)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                # Use calendar-aware annualization factor
                from src.market_calendar import classify_symbol, get_annualization_factor
                instrument = classify_symbol(symbol)
                ann_factor = get_annualization_factor(instrument, self.strategy.timeframe)
                sharpe = (returns.mean() / returns.std()) * ann_factor
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for p in pnls:
            if p < 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            timeframe=self.strategy.timeframe,
            start_date=str(data.index[0]),
            end_date=str(data.index[-1]),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=sum(pnls),
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=len(wins) / len(trades) if trades else 0,
            profit_factor=abs(sum(wins) / sum(losses)) if losses else float('inf'),
            avg_trade_pnl=np.mean(pnls) if pnls else 0,
            avg_win=np.mean(wins) if wins else 0,
            avg_loss=np.mean(losses) if losses else 0,
            max_consecutive_losses=max_consec,
            trades=trades,
            equity_curve=equity_curve,
        )
