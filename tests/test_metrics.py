"""Tests for live trading metrics."""
import pytest
from datetime import datetime, timezone, timedelta
from src.monitoring.metrics import LiveMetrics


class TestLiveMetrics:
    def test_empty_metrics(self):
        m = LiveMetrics()
        result = m.get_metrics()
        assert result['total_trades'] == 0
        assert result['sharpe_ratio'] == 0.0
        assert result['win_rate'] == 0.0

    def test_win_rate_calculation(self):
        m = LiveMetrics()
        now = datetime.now(timezone.utc)
        # 3 wins, 2 losses = 60% win rate
        for i, pnl in enumerate([100, -50, 200, -30, 150]):
            m.record_trade({
                'pnl': pnl,
                'entry_time': now - timedelta(hours=10-i),
                'exit_time': now - timedelta(hours=9-i),
                'symbol': 'TEST', 'side': 'long',
            })
        metrics = m.get_metrics(period_days=1)
        assert abs(metrics['win_rate'] - 0.6) < 0.01

    def test_profit_factor(self):
        m = LiveMetrics()
        now = datetime.now(timezone.utc)
        # wins = 100+200=300, losses = 50+30=80, PF = 3.75
        for i, pnl in enumerate([100, -50, 200, -30]):
            m.record_trade({
                'pnl': pnl,
                'entry_time': now - timedelta(hours=8-i),
                'exit_time': now - timedelta(hours=7-i),
                'symbol': 'TEST', 'side': 'long',
            })
        metrics = m.get_metrics(period_days=1)
        assert abs(metrics['profit_factor'] - 3.75) < 0.1

    def test_daily_pnl_bounded(self):
        m = LiveMetrics(max_daily_records=5)
        now = datetime.now(timezone.utc)
        # Add 10 days of trades
        for i in range(10):
            m.record_trade({
                'pnl': 100,
                'exit_time': now - timedelta(days=i),
                'entry_time': now - timedelta(days=i, hours=1),
                'symbol': 'X', 'side': 'long',
            })
        assert len(m._daily_pnl) <= 5

    def test_var_zero_volatility(self):
        m = LiveMetrics()
        # All identical returns
        result = m._calc_var([0.01, 0.01, 0.01, 0.01, 0.01])
        assert result == (0.0, 0.0)

    def test_equity_recording(self):
        m = LiveMetrics()
        m.record_equity(10000)
        m.record_equity(10500)
        m.record_equity(10200)
        assert len(m._equity_curve) == 3


class TestMaxDrawdown:
    def test_known_drawdown(self):
        m = LiveMetrics()
        # Equity: 100 -> 110 -> 90 -> 95
        # Peak: 110, Trough: 90, DD = (110-90)/110 = 18.18%
        m.record_equity(100)
        m.record_equity(110)
        m.record_equity(90)
        m.record_equity(95)
        max_dd, current_dd = m._calc_drawdowns()
        assert abs(max_dd - 18.18) < 0.5  # ~18% DD
