"""
Multi-Strategy Backtest Runner — Run backtests concurrently during live trading.

Provides:
- Run multiple strategies simultaneously on the same or different data
- Non-blocking execution (runs in background threads)
- Comparative results with automatic ranking
- Event-driven notifications on completion
- Configurable capital allocation per strategy
- Engine abstraction (native, VectorBT, Backtrader)
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backtesting.backtest import Backtester, BacktestResult
from src.strategy.base import BaseStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestStatus(str, Enum):
    """Status of a backtest run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestJob:
    """A single backtest job within a multi-strategy run."""
    job_id: str
    strategy_name: str
    symbol: str
    status: BacktestStatus = BacktestStatus.PENDING
    result: Optional[BacktestResult] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0


@dataclass
class MultiBacktestResult:
    """Aggregated results from a multi-strategy backtest run."""
    run_id: str
    status: BacktestStatus
    jobs: list[BacktestJob] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    @property
    def completed_jobs(self) -> list[BacktestJob]:
        return [j for j in self.jobs if j.status == BacktestStatus.COMPLETED]

    @property
    def failed_jobs(self) -> list[BacktestJob]:
        return [j for j in self.jobs if j.status == BacktestStatus.FAILED]

    def get_comparison_df(self) -> pd.DataFrame:
        """Get a comparison DataFrame of all completed strategy results."""
        rows = []
        for job in self.completed_jobs:
            if job.result:
                rows.append({
                    "strategy": job.strategy_name,
                    "symbol": job.symbol,
                    "total_return_pct": job.result.total_return_pct,
                    "sharpe_ratio": job.result.sharpe_ratio,
                    "max_drawdown": job.result.max_drawdown,
                    "win_rate": job.result.win_rate,
                    "total_trades": job.result.total_trades,
                    "profit_factor": job.result.profit_factor,
                    "avg_trade_pnl": job.result.avg_trade_pnl,
                })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)

    def get_best_strategy(self) -> Optional[str]:
        """Return the strategy name with the highest Sharpe ratio."""
        df = self.get_comparison_df()
        if df.empty:
            return None
        return df.iloc[0]["strategy"]

    def summary(self) -> str:
        """Human-readable summary of all results."""
        lines = [
            f"\n{'='*70}",
            f"MULTI-STRATEGY BACKTEST RESULTS (run_id={self.run_id[:8]})",
            f"{'='*70}",
            f"Status: {self.status.value} | Duration: {self.duration_seconds:.1f}s",
            f"Jobs: {len(self.completed_jobs)} completed, {len(self.failed_jobs)} failed",
            f"{'─'*70}",
        ]

        df = self.get_comparison_df()
        if not df.empty:
            lines.append(f"{'Strategy':<20} {'Symbol':<8} {'Return':<10} {'Sharpe':<8} "
                        f"{'MaxDD':<8} {'WinRate':<8} {'Trades':<7}")
            lines.append(f"{'─'*70}")
            for _, row in df.iterrows():
                lines.append(
                    f"{row['strategy']:<20} {row['symbol']:<8} "
                    f"{row['total_return_pct']:>8.2%} {row['sharpe_ratio']:>7.3f} "
                    f"{row['max_drawdown']:>7.2%} {row['win_rate']:>7.1%} "
                    f"{row['total_trades']:>6}"
                )
            lines.append(f"{'─'*70}")
            best = self.get_best_strategy()
            lines.append(f"🏆 Best Strategy: {best}")

        for job in self.failed_jobs:
            lines.append(f"❌ FAILED: {job.strategy_name}/{job.symbol}: {job.error}")

        lines.append(f"{'='*70}")
        return "\n".join(lines)


class BacktestRunner:
    """
    Multi-strategy backtest runner with concurrent execution.

    Supports:
    - Running multiple strategies on same data (comparison mode)
    - Running same strategy on multiple symbols (portfolio mode)
    - Background execution that doesn't block live trading
    - Event notifications on completion
    """

    def __init__(
        self,
        max_workers: int = 4,
        event_bus=None,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        self._max_workers = max_workers
        self._event_bus = event_bus
        self._initial_capital = initial_capital
        self._commission_pct = commission_pct
        self._slippage_pct = slippage_pct
        self._active_runs: dict[str, MultiBacktestResult] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._async_threads: dict[str, threading.Thread] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Synchronous API (blocking)
    # ──────────────────────────────────────────────────────────────────────

    def run_multiple(
        self,
        strategies: list[BaseStrategy],
        data: dict[str, pd.DataFrame],
        capital_allocation: Optional[dict[str, float]] = None,
    ) -> MultiBacktestResult:
        """
        Run multiple strategies on provided data (blocking).

        Args:
            strategies: List of strategy instances to test.
            data: Dict of symbol -> OHLCV DataFrame.
            capital_allocation: Optional dict of strategy_name -> allocation fraction.
                               Defaults to equal allocation.

        Returns:
            MultiBacktestResult with all strategy results.
        """
        run_id = uuid.uuid4().hex[:12]
        result = MultiBacktestResult(
            run_id=run_id,
            status=BacktestStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # Build job list
        jobs = []
        for strategy in strategies:
            allocation = 1.0
            if capital_allocation:
                allocation = capital_allocation.get(strategy.name, 1.0 / len(strategies))
            capital = self._initial_capital * allocation

            for symbol, df in data.items():
                job = BacktestJob(
                    job_id=uuid.uuid4().hex[:8],
                    strategy_name=strategy.name,
                    symbol=symbol,
                )
                jobs.append((job, strategy, df, symbol, capital))

        result.jobs = [j[0] for j in jobs]

        # Publish start event
        self._publish_start_event(run_id, strategies, list(data.keys()))

        # Execute concurrently
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures: dict[Future, BacktestJob] = {}
            for job, strategy, df, symbol, capital in jobs:
                future = executor.submit(
                    self._run_single_backtest, job, strategy, df, symbol, capital
                )
                futures[future] = job

            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()  # Raises if the task raised
                except Exception as e:
                    job.status = BacktestStatus.FAILED
                    job.error = str(e)
                    logger.error(
                        "backtest_runner.job_failed",
                        job_id=job.job_id,
                        strategy=job.strategy_name,
                        symbol=job.symbol,
                        error=str(e),
                    )

        # Finalize
        result.completed_at = datetime.now(timezone.utc)
        result.duration_seconds = (result.completed_at - result.started_at).total_seconds()
        result.status = BacktestStatus.COMPLETED

        # Publish completion events
        self._publish_completion_events(result)

        with self._lock:
            self._active_runs[run_id] = result

        logger.info(
            "backtest_runner.completed",
            run_id=run_id,
            strategies=len(strategies),
            symbols=len(data),
            jobs_completed=len(result.completed_jobs),
            jobs_failed=len(result.failed_jobs),
            duration_s=result.duration_seconds,
        )

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Asynchronous API (non-blocking, for use during live trading)
    # ──────────────────────────────────────────────────────────────────────

    def run_async(
        self,
        strategies: list[BaseStrategy],
        data: dict[str, pd.DataFrame],
        capital_allocation: Optional[dict[str, float]] = None,
        callback: Optional[Callable[[MultiBacktestResult], None]] = None,
    ) -> str:
        """
        Run multiple strategies in the background (non-blocking).

        Args:
            strategies: List of strategy instances to test.
            data: Dict of symbol -> OHLCV DataFrame.
            capital_allocation: Optional capital allocation per strategy.
            callback: Optional function called with results when complete.

        Returns:
            run_id string for tracking progress.
        """
        run_id = uuid.uuid4().hex[:12]

        def _background():
            try:
                result = self.run_multiple(strategies, data, capital_allocation)
                if callback:
                    callback(result)
            except Exception as e:
                logger.error("backtest_runner.async_failed", run_id=run_id, error=str(e))
            finally:
                with self._lock:
                    self._async_threads.pop(run_id, None)

        thread = threading.Thread(
            target=_background,
            name=f"backtest-{run_id}",
            daemon=True,
        )
        with self._lock:
            self._async_threads[run_id] = thread
        thread.start()

        logger.info(
            "backtest_runner.async_started",
            run_id=run_id,
            strategies=[s.name for s in strategies],
            symbols=list(data.keys()),
        )

        return run_id

    def stop(self, timeout: float = 10.0) -> None:
        """Join active async backtest threads for deterministic shutdown."""
        with self._lock:
            threads = list(self._async_threads.items())
        for run_id, thread in threads:
            if thread.is_alive():
                thread.join(timeout=timeout)
                if thread.is_alive():
                    logger.warning("backtest_runner.stop_timeout", run_id=run_id, timeout=timeout)

    # ──────────────────────────────────────────────────────────────────────
    # Status & Management
    # ──────────────────────────────────────────────────────────────────────

    def get_run_status(self, run_id: str) -> Optional[dict]:
        """Get status of a specific backtest run."""
        with self._lock:
            result = self._active_runs.get(run_id)
            if not result:
                return None
            return {
                "run_id": run_id,
                "status": result.status.value,
                "total_jobs": len(result.jobs),
                "completed": len(result.completed_jobs),
                "failed": len(result.failed_jobs),
                "duration_seconds": result.duration_seconds,
                "best_strategy": result.get_best_strategy(),
            }

    def get_result(self, run_id: str) -> Optional[MultiBacktestResult]:
        """Get full result object for a completed run."""
        with self._lock:
            return self._active_runs.get(run_id)

    def list_runs(self, limit: int = 20) -> list[dict]:
        """List recent backtest runs."""
        with self._lock:
            runs = sorted(
                self._active_runs.values(),
                key=lambda r: r.started_at or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )[:limit]
            return [
                {
                    "run_id": r.run_id,
                    "status": r.status.value,
                    "jobs": len(r.jobs),
                    "completed": len(r.completed_jobs),
                    "duration_seconds": r.duration_seconds,
                    "best_strategy": r.get_best_strategy(),
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                }
                for r in runs
            ]

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _run_single_backtest(
        self,
        job: BacktestJob,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
        capital: float,
    ) -> None:
        """Execute a single backtest job. Modifies job in-place."""
        job.status = BacktestStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)

        try:
            # Use asset-class-aware costs when runner uses default commission/slippage
            if self._commission_pct == 0.001 and self._slippage_pct == 0.0005:
                bt = Backtester.for_symbol(
                    strategy=strategy,
                    symbol=symbol,
                    initial_capital=capital,
                )
            else:
                bt = Backtester(
                    strategy=strategy,
                    initial_capital=capital,
                    commission_pct=self._commission_pct,
                    slippage_pct=self._slippage_pct,
                )
            result = bt.run(data, symbol=symbol)
            job.result = result
            job.status = BacktestStatus.COMPLETED
        except Exception as e:
            job.status = BacktestStatus.FAILED
            job.error = str(e)
            raise
        finally:
            job.completed_at = datetime.now(timezone.utc)
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

    def _publish_start_event(self, run_id: str, strategies: list, symbols: list):
        """Publish BacktestStarted event."""
        if self._event_bus:
            from src.core.events import BacktestStarted
            self._event_bus.publish(BacktestStarted(
                run_id=run_id,
                strategies=[s.name for s in strategies],
                symbols=symbols,
                engine="native",
                source="backtest_runner",
            ))

    def _publish_completion_events(self, result: MultiBacktestResult):
        """Publish BacktestCompleted events for each finished job."""
        if not self._event_bus:
            return
        from src.core.events import BacktestCompleted
        for job in result.completed_jobs:
            if job.result:
                self._event_bus.publish(BacktestCompleted(
                    run_id=result.run_id,
                    strategy=job.strategy_name,
                    total_return_pct=job.result.total_return_pct,
                    sharpe_ratio=job.result.sharpe_ratio,
                    total_trades=job.result.total_trades,
                    duration_seconds=job.duration_seconds,
                    source="backtest_runner",
                ))
