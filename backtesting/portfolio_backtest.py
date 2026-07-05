"""
Portfolio-Level Backtesting — Multi-symbol simultaneous backtest with capital allocation.
Tests strategy performance across correlated assets with realistic constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PortfolioTrade:
    """Record of a single portfolio trade."""
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
    fees_paid: float = 0.0

    def __post_init__(self):
        if self.side == "buy":
            self.pnl = (self.exit_price - self.entry_price) * self.qty - self.fees_paid
            self.pnl_pct = (self.exit_price / self.entry_price) - 1
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.qty - self.fees_paid
            self.pnl_pct = (self.entry_price / self.exit_price) - 1


@dataclass
class _OpenPosition:
    """Internal tracker for an open position."""
    symbol: str
    side: str
    entry_price: float
    qty: float
    entry_time: datetime
    entry_bar: int
    invested_value: float


def _compute_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average using numpy for performance."""
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
    return ema


def _align_data(data: dict[str, pd.DataFrame]) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """
    Align all symbol DataFrames to a common DatetimeIndex via union + forward-fill.

    Calendar-aware: symbols are only forward-filled within their valid trading
    sessions. Crypto bars during equity "closed" hours are preserved without
    forcing equity NaN fills at those timestamps.

    Returns:
        common_index: The unified DatetimeIndex covering all dates.
        aligned: Dict of symbol -> reindexed DataFrame with forward-filled prices.
    """
    from src.market_calendar import classify_symbol

    all_indices = [df.index for df in data.values()]
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)
    common_index = common_index.sort_values()

    aligned = {}
    for symbol, df in data.items():
        instrument = classify_symbol(symbol)
        reindexed = df.reindex(common_index)

        if instrument.trades_24_7:
            # Crypto: forward-fill all gaps (should be continuous)
            reindexed = reindexed.ffill()
        else:
            # Equities: only forward-fill within trading sessions.
            # Do NOT fill weekend/overnight gaps with stale prices for more than
            # a reasonable lookback. Use ffill with a limit to avoid propagating
            # Friday's close across an entire weekend of crypto bars.
            # Limit = max intraday bars (e.g., 7 hourly bars per NYSE session)
            reindexed = reindexed.ffill(limit=7)

        aligned[symbol] = reindexed

    return common_index, aligned


def _compute_correlation_impact(equity_curve: np.ndarray, per_symbol_equity: dict[str, np.ndarray]) -> float:
    """
    Compute correlation-adjusted diversification ratio.

    Measures how much portfolio diversification reduced volatility compared
    to the weighted average of individual symbol volatilities.
    Returns a value between 0 and 1 where lower means more diversification benefit.
    """
    if len(per_symbol_equity) < 2:
        return 1.0

    # Compute returns for each symbol that has enough data
    symbol_returns = []
    for sym, eq in per_symbol_equity.items():
        if len(eq) > 1 and np.any(eq[:-1] != 0):
            rets = np.diff(eq) / np.where(eq[:-1] == 0, 1, eq[:-1])
            symbol_returns.append(rets)

    if len(symbol_returns) < 2:
        return 1.0

    # Pad to same length
    min_len = min(len(r) for r in symbol_returns)
    symbol_returns = [r[:min_len] for r in symbol_returns]
    returns_matrix = np.column_stack(symbol_returns)

    # Average pairwise correlation
    corr_matrix = np.corrcoef(returns_matrix, rowvar=False)
    n = corr_matrix.shape[0]
    if n < 2:
        return 1.0

    upper_tri = corr_matrix[np.triu_indices(n, k=1)]
    avg_corr = np.nanmean(upper_tri) if len(upper_tri) > 0 else 1.0

    return float(np.clip(avg_corr, -1.0, 1.0))


class PortfolioBacktester:
    """
    Portfolio-level backtester that runs an EMA crossover strategy across
    multiple symbols simultaneously with proper capital allocation and risk constraints.

    Parameters:
        initial_cash: Starting capital.
        max_positions: Maximum number of concurrent open positions.
        risk_per_trade: Fraction of equity risked per trade for position sizing.
        fees: Trading fee as a fraction of trade value (e.g., 0.001 = 0.1%).
        max_exposure: Maximum fraction of equity that can be invested at any time.
        max_single_pct: Maximum fraction of equity allocated to a single position.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        max_positions: int = 10,
        risk_per_trade: float = 0.02,
        fees: float = 0.001,
        max_exposure: float = 0.8,
        max_single_pct: float = 0.15,
    ):
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.fees = fees
        self.max_exposure = max_exposure
        self.max_single_pct = max_single_pct

    def run(self, data: dict[str, pd.DataFrame], fast_ema: int = 12, slow_ema: int = 26) -> dict:
        """
        Run portfolio backtest across all symbols simultaneously.

        Args:
            data: Dict of symbol -> OHLCV DataFrame. Each DataFrame must have columns
                  ['open', 'high', 'low', 'close', 'volume'] with a DatetimeIndex.
            fast_ema: Fast EMA period for crossover signal.
            slow_ema: Slow EMA period for crossover signal.

        Returns:
            Dict with comprehensive backtest results including equity curve,
            per-symbol stats, trade list, and risk metrics.
        """
        if not data:
            raise ValueError("data dict must contain at least one symbol")

        logger.info(
            f"Starting portfolio backtest: {len(data)} symbols, "
            f"fast_ema={fast_ema}, slow_ema={slow_ema}, "
            f"initial_cash={self.initial_cash}"
        )

        # Align all symbols to common dates
        common_index, aligned = _align_data(data)
        n_bars = len(common_index)
        symbols = list(aligned.keys())

        if n_bars < slow_ema + 1:
            raise ValueError(
                f"Not enough bars ({n_bars}) for slow EMA period ({slow_ema}). "
                f"Need at least {slow_ema + 1} bars."
            )

        # Pre-compute EMA signals for all symbols
        ema_fast = {}
        ema_slow = {}
        close_prices = {}

        for symbol in symbols:
            closes = aligned[symbol]["close"].values.astype(float)
            close_prices[symbol] = closes
            # Only compute EMAs where we have valid data (non-NaN)
            valid_mask = ~np.isnan(closes)
            if valid_mask.sum() >= slow_ema:
                # Fill NaN with forward fill for EMA computation
                filled = pd.Series(closes).ffill().bfill().values
                ema_fast[symbol] = _compute_ema(filled, fast_ema)
                ema_slow[symbol] = _compute_ema(filled, slow_ema)
            else:
                ema_fast[symbol] = np.full(n_bars, np.nan)
                ema_slow[symbol] = np.full(n_bars, np.nan)

        # State tracking
        cash = self.initial_cash
        positions: dict[str, _OpenPosition] = {}  # symbol -> position
        all_trades: list[PortfolioTrade] = []
        equity_curve = np.zeros(n_bars)
        exposure_history = np.zeros(n_bars)
        max_concurrent = 0
        per_symbol_equity: dict[str, np.ndarray] = {s: np.zeros(n_bars) for s in symbols}

        # Day-by-day iteration
        for bar_idx in range(n_bars):
            # Skip warmup period
            if bar_idx < slow_ema:
                equity_curve[bar_idx] = cash
                continue

            # Calculate current portfolio value
            position_value = 0.0
            for sym, pos in positions.items():
                current_price = close_prices[sym][bar_idx]
                if not np.isnan(current_price):
                    position_value += current_price * pos.qty

            equity = cash + position_value
            exposure = position_value / equity if equity > 0 else 0.0

            # Process signals for each symbol
            exit_symbols = []
            entry_candidates = []

            for symbol in symbols:
                price = close_prices[symbol][bar_idx]
                if np.isnan(price) or np.isnan(ema_fast[symbol][bar_idx]):
                    continue

                fast_val = ema_fast[symbol][bar_idx]
                slow_val = ema_slow[symbol][bar_idx]
                prev_fast = ema_fast[symbol][bar_idx - 1]
                prev_slow = ema_slow[symbol][bar_idx - 1]

                if np.isnan(prev_fast) or np.isnan(prev_slow):
                    continue

                # Bullish crossover: fast crosses above slow
                is_entry = prev_fast <= prev_slow and fast_val > slow_val
                # Bearish crossover: fast crosses below slow
                is_exit = prev_fast >= prev_slow and fast_val < slow_val

                if symbol in positions and is_exit:
                    exit_symbols.append(symbol)
                elif symbol not in positions and is_entry:
                    entry_candidates.append((symbol, price))

            # Process exits first (frees capital)
            for symbol in exit_symbols:
                pos = positions[symbol]
                exit_price = close_prices[symbol][bar_idx]
                exit_fee = exit_price * pos.qty * self.fees
                proceeds = exit_price * pos.qty - exit_fee

                trade = PortfolioTrade(
                    symbol=symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    qty=pos.qty,
                    entry_time=common_index[pos.entry_bar],
                    exit_time=common_index[bar_idx],
                    hold_bars=bar_idx - pos.entry_bar,
                    fees_paid=(pos.entry_price * pos.qty * self.fees) + exit_fee,
                )
                all_trades.append(trade)
                cash += proceeds
                del positions[symbol]
                logger.debug(
                    f"EXIT {symbol} @ {exit_price:.4f}, PnL: {trade.pnl:.2f} "
                    f"({trade.pnl_pct:.2%})"
                )

            # Process entries (respect constraints)
            for symbol, price in entry_candidates:
                # Check position limit
                if len(positions) >= self.max_positions:
                    break

                # Recalculate equity after exits
                pos_val = sum(
                    close_prices[s][bar_idx] * p.qty
                    for s, p in positions.items()
                    if not np.isnan(close_prices[s][bar_idx])
                )
                current_equity = cash + pos_val
                current_exposure = pos_val / current_equity if current_equity > 0 else 0.0

                # Check exposure limit
                if current_exposure >= self.max_exposure:
                    break

                # Position sizing: risk_per_trade * equity, capped at max_single_pct
                alloc = min(
                    self.risk_per_trade * current_equity / 0.02,  # normalize to ~1x risk
                    self.max_single_pct * current_equity,
                )
                # Also cap by remaining exposure budget
                remaining_budget = (self.max_exposure - current_exposure) * current_equity
                alloc = min(alloc, remaining_budget, cash)

                if alloc <= 0 or price <= 0:
                    continue

                # Calculate quantity and apply entry fee
                entry_fee = alloc * self.fees
                invest_amount = alloc - entry_fee
                qty = invest_amount / price

                if qty <= 0:
                    continue

                positions[symbol] = _OpenPosition(
                    symbol=symbol,
                    side="buy",
                    entry_price=price,
                    qty=qty,
                    entry_time=common_index[bar_idx],
                    entry_bar=bar_idx,
                    invested_value=invest_amount,
                )
                cash -= alloc
                logger.debug(
                    f"ENTRY {symbol} @ {price:.4f}, qty={qty:.4f}, "
                    f"alloc={alloc:.2f}"
                )

            # Update tracking
            max_concurrent = max(max_concurrent, len(positions))

            # Final equity for this bar
            pos_val = sum(
                close_prices[s][bar_idx] * p.qty
                for s, p in positions.items()
                if not np.isnan(close_prices[s][bar_idx])
            )
            equity_curve[bar_idx] = cash + pos_val
            exposure_history[bar_idx] = pos_val / equity_curve[bar_idx] if equity_curve[bar_idx] > 0 else 0.0

            # Track per-symbol equity contribution
            for sym in symbols:
                if sym in positions:
                    p = positions[sym]
                    sym_price = close_prices[sym][bar_idx]
                    if not np.isnan(sym_price):
                        per_symbol_equity[sym][bar_idx] = sym_price * p.qty

        # Close any remaining positions at final bar
        final_bar = n_bars - 1
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            exit_price = close_prices[symbol][final_bar]
            if np.isnan(exit_price):
                # Use last valid price
                valid = ~np.isnan(close_prices[symbol])
                if valid.any():
                    exit_price = close_prices[symbol][valid][-1]
                else:
                    exit_price = pos.entry_price

            exit_fee = exit_price * pos.qty * self.fees
            trade = PortfolioTrade(
                symbol=symbol,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                qty=pos.qty,
                entry_time=common_index[pos.entry_bar],
                exit_time=common_index[final_bar],
                hold_bars=final_bar - pos.entry_bar,
                fees_paid=(pos.entry_price * pos.qty * self.fees) + exit_fee,
            )
            all_trades.append(trade)

        # Compute results
        final_value = equity_curve[final_bar] if equity_curve[final_bar] > 0 else equity_curve[equity_curve > 0][-1]
        total_return = (final_value / self.initial_cash) - 1

        # Sharpe ratio (annualized)
        valid_equity = equity_curve[slow_ema:]
        daily_returns = np.diff(valid_equity) / np.where(valid_equity[:-1] == 0, 1, valid_equity[:-1])
        sharpe_ratio = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

        # Max drawdown
        running_max = np.maximum.accumulate(valid_equity)
        drawdowns = (valid_equity - running_max) / np.where(running_max == 0, 1, running_max)
        max_drawdown = abs(float(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0

        # Trade statistics
        total_trades = len(all_trades)
        winning_trades = [t for t in all_trades if t.pnl > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_trade_return = (
            float(np.mean([t.pnl_pct for t in all_trades])) if total_trades > 0 else 0.0
        )
        avg_holding_period = (
            float(np.mean([t.hold_bars for t in all_trades])) if total_trades > 0 else 0.0
        )

        # Per-symbol results
        per_symbol_results = {}
        for symbol in symbols:
            sym_trades = [t for t in all_trades if t.symbol == symbol]
            sym_wins = [t for t in sym_trades if t.pnl > 0]
            sym_return = sum(t.pnl for t in sym_trades)
            per_symbol_results[symbol] = {
                "return": sym_return,
                "trades": len(sym_trades),
                "win_rate": len(sym_wins) / len(sym_trades) if sym_trades else 0.0,
            }

        # Correlation impact
        correlation_impact = _compute_correlation_impact(equity_curve, per_symbol_equity)

        # Format trades list
        trades_list = [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "hold_bars": t.hold_bars,
                "fees_paid": t.fees_paid,
            }
            for t in all_trades
        ]

        logger.info(
            f"Backtest complete: {total_trades} trades, "
            f"return={total_return:.2%}, sharpe={sharpe_ratio:.2f}, "
            f"max_dd={max_drawdown:.2%}"
        )

        return {
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_trade_return": avg_trade_return,
            "avg_holding_period": avg_holding_period,
            "max_concurrent_positions": max_concurrent,
            "per_symbol_results": per_symbol_results,
            "equity_curve": equity_curve,
            "trades": trades_list,
            "correlation_impact": correlation_impact,
            "exposure_history": exposure_history,
        }


def portfolio_backtest(
    data: dict[str, pd.DataFrame],
    fast_ema: int = 12,
    slow_ema: int = 26,
    initial_cash: float = 10000.0,
    max_positions: int = 10,
    fees: float = 0.001,
) -> dict:
    """
    Convenience function to run a portfolio-level backtest with default risk parameters.

    Args:
        data: Dict of symbol -> OHLCV DataFrame with DatetimeIndex.
        fast_ema: Fast EMA period.
        slow_ema: Slow EMA period.
        initial_cash: Starting capital.
        max_positions: Max concurrent positions.
        fees: Trading fee fraction.

    Returns:
        Backtest results dict (see PortfolioBacktester.run for full schema).
    """
    bt = PortfolioBacktester(
        initial_cash=initial_cash,
        max_positions=max_positions,
        fees=fees,
    )
    return bt.run(data, fast_ema=fast_ema, slow_ema=slow_ema)


def portfolio_comparison(
    data: dict[str, pd.DataFrame],
    param_sets: list[dict],
) -> pd.DataFrame:
    """
    Run portfolio backtest with multiple parameter sets and return a comparison DataFrame.

    Each param_set dict can contain any kwargs accepted by PortfolioBacktester.__init__
    and PortfolioBacktester.run (fast_ema, slow_ema).

    Args:
        data: Dict of symbol -> OHLCV DataFrame.
        param_sets: List of dicts, each containing backtest parameters to compare.

    Returns:
        DataFrame with one row per param set and columns for key metrics.

    Example:
        >>> params = [
        ...     {"fast_ema": 8, "slow_ema": 21, "max_positions": 5},
        ...     {"fast_ema": 12, "slow_ema": 26, "max_positions": 10},
        ...     {"fast_ema": 20, "slow_ema": 50, "fees": 0.002},
        ... ]
        >>> comparison = portfolio_comparison(data, params)
    """
    init_keys = {"initial_cash", "max_positions", "risk_per_trade", "fees", "max_exposure", "max_single_pct"}
    run_keys = {"fast_ema", "slow_ema"}

    results = []
    for i, params in enumerate(param_sets):
        init_kwargs = {k: v for k, v in params.items() if k in init_keys}
        run_kwargs = {k: v for k, v in params.items() if k in run_keys}

        logger.info(f"Running comparison set {i + 1}/{len(param_sets)}: {params}")

        bt = PortfolioBacktester(**init_kwargs)
        result = bt.run(data, **run_kwargs)

        row = {
            "params": str(params),
            "initial_cash": result["initial_cash"],
            "final_value": result["final_value"],
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "calmar_ratio": result["calmar_ratio"],
            "total_trades": result["total_trades"],
            "win_rate": result["win_rate"],
            "avg_trade_return": result["avg_trade_return"],
            "avg_holding_period": result["avg_holding_period"],
            "max_concurrent_positions": result["max_concurrent_positions"],
            "correlation_impact": result["correlation_impact"],
        }
        # Include individual params for easy filtering
        for k, v in params.items():
            row[f"param_{k}"] = v

        results.append(row)

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    logger.info(
        f"Comparison complete: {len(param_sets)} sets, "
        f"best sharpe={comparison_df['sharpe_ratio'].iloc[0]:.2f}"
    )

    return comparison_df
