"""
Backtesting package — strategy evaluation against historical data.

Modules:
    backtest        — Event-driven backtesting engine (per-bar simulation)
    runner          — Multi-strategy concurrent backtest runner
    vbt_adapter     — Vectorized EMA crossover backtests (numpy/vectorbt)
    walk_forward    — Walk-forward optimization (rolling OOS validation)
    param_validator — Parameter validation via walk-forward before promotion
    monte_carlo     — Monte Carlo simulation for confidence intervals
    portfolio_backtest — Multi-asset portfolio backtesting
    bt_adapter      — Backtrader framework integration
"""
