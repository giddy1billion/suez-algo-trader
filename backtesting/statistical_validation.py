"""
Advanced Statistical Validation — Prevents overfitting through rigorous hypothesis testing.

Implements:
1. Deflated Sharpe Ratio (DSR) — adjusts Sharpe for skewness, kurtosis, and multiple testing
2. Probability of Backtest Overfitting (PBO) — combinatorial symmetric cross-validation
3. White's Reality Check — bootstrap stepwise test for multiple strategy comparison
4. Purged Time-Series Cross-Validation — gap-aware splitting for autocorrelated data

References:
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
- Bailey et al. (2017): "Probability of Backtest Overfitting"
- White (2000): "A Reality Check for Data Snooping"
- de Prado (2018): "Advances in Financial Machine Learning" Ch. 7, 12
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio test."""

    observed_sharpe: float
    deflated_sharpe: float
    p_value: float
    is_significant: bool
    sharpe_threshold: float
    n_trials: int
    skewness: float
    kurtosis: float
    var_sharpe: float


@dataclass
class PBOResult:
    """Probability of Backtest Overfitting result."""

    pbo: float  # probability of overfitting (0-1)
    performance_degradation: float  # avg OOS rank loss
    logit_values: List[float]
    n_combinations: int
    is_overfit: bool  # PBO > 0.5 threshold
    stochastic_dominance: float  # 2nd-order SD ratio


@dataclass
class RealityCheckResult:
    """White's Reality Check / SPA test result."""

    best_strategy_return: float
    p_value: float
    is_significant: bool
    bootstrap_distribution: np.ndarray
    n_strategies: int
    consistent_p_value: float  # Hansen's SPA


@dataclass
class PurgedCVResult:
    """Purged time-series cross-validation result."""

    splits: List[Dict[str, Any]]
    mean_oos_return: float
    std_oos_return: float
    mean_oos_sharpe: float
    all_oos_returns: List[float]
    all_oos_sharpes: List[float]
    n_splits: int


@dataclass
class StatisticalReport:
    """Comprehensive statistical validation report."""

    symbol: str
    strategy_name: str
    dsr: Optional[DeflatedSharpeResult] = None
    pbo: Optional[PBOResult] = None
    reality_check: Optional[RealityCheckResult] = None
    purged_cv: Optional[PurgedCVResult] = None
    overall_pass: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*65}",
            f"STATISTICAL VALIDATION REPORT: {self.symbol} ({self.strategy_name})",
            f"{'='*65}",
            f"Overall: {'✅ PASS' if self.overall_pass else '❌ FAIL'}",
            f"{'─'*65}",
        ]
        if self.dsr:
            lines.append(f"Deflated Sharpe:  {self.dsr.deflated_sharpe:.3f} (observed: {self.dsr.observed_sharpe:.3f}, p={self.dsr.p_value:.4f})")
        if self.pbo:
            lines.append(f"PBO:              {self.pbo.pbo:.1%} ({'OVERFIT' if self.pbo.is_overfit else 'OK'})")
        if self.reality_check:
            lines.append(f"Reality Check:    p={self.reality_check.p_value:.4f} ({'significant' if self.reality_check.is_significant else 'not significant'})")
        if self.purged_cv:
            lines.append(f"Purged CV:        mean_return={self.purged_cv.mean_oos_return:.4f} ± {self.purged_cv.std_oos_return:.4f}")
        if self.rejection_reasons:
            lines.append(f"{'─'*65}")
            lines.append("Rejection Reasons:")
            for r in self.rejection_reasons:
                lines.append(f"  • {r}")
        lines.append(f"{'='*65}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# 1. Deflated Sharpe Ratio
# ──────────────────────────────────────────────────────────────────────────


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int = 1,
    annualization_factor: float = np.sqrt(252),
    significance_level: float = 0.05,
) -> DeflatedSharpeResult:
    """
    Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts the observed Sharpe ratio for:
    - Non-normality (skewness and kurtosis)
    - Multiple testing (number of trials/strategies tested)

    Args:
        returns: Array of period returns.
        n_trials: Number of strategies/parameter sets tested (for multiple testing).
        annualization_factor: Factor to annualize Sharpe (sqrt(252) for daily).
        significance_level: p-value threshold for significance.

    Returns:
        DeflatedSharpeResult with adjusted Sharpe and significance test.
    """
    T = len(returns)
    if T < 10:
        return DeflatedSharpeResult(
            observed_sharpe=0.0, deflated_sharpe=0.0, p_value=1.0,
            is_significant=False, sharpe_threshold=0.0, n_trials=n_trials,
            skewness=0.0, kurtosis=0.0, var_sharpe=0.0,
        )

    # Observed Sharpe
    sr = returns.mean() / returns.std() if returns.std() > 1e-10 else 0.0
    sr_annualized = sr * annualization_factor

    # Higher moments
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # excess kurtosis

    # Variance of Sharpe estimator (Lo, 2002 corrected for non-normality)
    var_sr = (1 - skew * sr + ((kurt - 1) / 4) * sr ** 2) / T

    # Expected maximum Sharpe under null (multiple testing adjustment)
    # E[max(SR)] ≈ sqrt(2 * log(N)) * (1 - gamma / (2 * log(N))) + gamma / sqrt(2 * log(N))
    # Simplified: Euler-Mascheroni approximation
    if n_trials > 1:
        euler_mascheroni = 0.5772156649
        z = np.sqrt(2 * np.log(n_trials))
        expected_max_sr = z - (np.log(np.pi) + euler_mascheroni) / (2 * z)
        sr_threshold = expected_max_sr / annualization_factor
    else:
        sr_threshold = 0.0
        expected_max_sr = 0.0

    # Deflated Sharpe = (SR - E[max(SR)]) / sqrt(Var[SR])
    if var_sr > 0:
        dsr_stat = (sr - sr_threshold) / np.sqrt(var_sr)
        p_value = 1 - stats.norm.cdf(dsr_stat)
    else:
        dsr_stat = 0.0
        p_value = 1.0

    deflated_sr = sr_annualized - expected_max_sr

    return DeflatedSharpeResult(
        observed_sharpe=sr_annualized,
        deflated_sharpe=deflated_sr,
        p_value=p_value,
        is_significant=p_value < significance_level,
        sharpe_threshold=expected_max_sr,
        n_trials=n_trials,
        skewness=skew,
        kurtosis=kurt,
        var_sharpe=var_sr,
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. Probability of Backtest Overfitting (PBO)
# ──────────────────────────────────────────────────────────────────────────


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_partitions: int = 16,
) -> PBOResult:
    """
    Estimate Probability of Backtest Overfitting (Bailey et al., 2017).

    Uses Combinatorial Symmetric Cross-Validation (CSCV):
    1. Partition time series into S sub-samples
    2. For each combination of S/2 training partitions:
       a. Select best strategy in-sample (IS)
       b. Measure its out-of-sample (OOS) rank
    3. PBO = fraction of combinations where IS-best underperforms OOS median

    Args:
        returns_matrix: Shape (T, N) where T=periods, N=strategies/trials.
            Each column is the return series of one strategy configuration.
        n_partitions: Number of sub-samples (must be even, typically 8-16).

    Returns:
        PBOResult with overfitting probability.
    """
    T, N = returns_matrix.shape

    if N < 2:
        return PBOResult(pbo=0.0, performance_degradation=0.0, logit_values=[],
                         n_combinations=0, is_overfit=False, stochastic_dominance=0.0)

    # Ensure even partitions
    n_partitions = min(n_partitions, T // 10)  # Need at least 10 obs per partition
    if n_partitions < 4:
        n_partitions = 4
    if n_partitions % 2 != 0:
        n_partitions -= 1

    partition_size = T // n_partitions
    if partition_size < 5:
        return PBOResult(pbo=0.0, performance_degradation=0.0, logit_values=[],
                         n_combinations=0, is_overfit=False, stochastic_dominance=0.0)

    # Partition returns into sub-samples
    partitioned = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size
        partitioned.append(returns_matrix[start:end])

    # CSCV: enumerate combinations of S/2 partitions for training
    from itertools import combinations
    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))

    # Limit combinations for computational tractability
    max_combos = 100
    if len(all_combos) > max_combos:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_combos), max_combos, replace=False)
        all_combos = [all_combos[i] for i in indices]

    logit_values = []
    rank_losses = []

    for train_indices in all_combos:
        test_indices = tuple(i for i in range(n_partitions) if i not in train_indices)

        # Aggregate IS and OOS returns per strategy
        is_returns = np.vstack([partitioned[i] for i in train_indices])
        oos_returns = np.vstack([partitioned[i] for i in test_indices])

        # IS performance: cumulative return per strategy
        is_perf = is_returns.sum(axis=0)  # shape (N,)
        oos_perf = oos_returns.sum(axis=0)  # shape (N,)

        # IS-best strategy
        best_is_idx = np.argmax(is_perf)

        # OOS rank of IS-best (0 = best, N-1 = worst)
        oos_ranks = np.argsort(np.argsort(-oos_perf))  # rank descending
        oos_rank_of_best = oos_ranks[best_is_idx]

        # Relative rank (normalized to [0, 1])
        relative_rank = oos_rank_of_best / (N - 1) if N > 1 else 0.0
        rank_losses.append(relative_rank)

        # Logit: log(rank / (1 - rank)) — measure of IS-best degradation OOS
        # Clip to avoid log(0) or log(inf)
        clipped_rank = np.clip(relative_rank, 0.01, 0.99)
        logit = np.log(clipped_rank / (1 - clipped_rank))
        logit_values.append(logit)

    # PBO = fraction of combinations where IS-best is below median OOS
    pbo = np.mean(np.array(rank_losses) > 0.5)

    # Performance degradation: average rank loss
    perf_degradation = np.mean(rank_losses)

    # Stochastic dominance: fraction where logit > 0 (overfit dominates)
    sd_ratio = np.mean(np.array(logit_values) > 0)

    return PBOResult(
        pbo=float(pbo),
        performance_degradation=float(perf_degradation),
        logit_values=[float(l) for l in logit_values],
        n_combinations=len(all_combos),
        is_overfit=pbo > 0.5,
        stochastic_dominance=float(sd_ratio),
    )


# ──────────────────────────────────────────────────────────────────────────
# 3. White's Reality Check / Hansen's SPA
# ──────────────────────────────────────────────────────────────────────────


def whites_reality_check(
    returns_matrix: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    block_size: int = 10,
    significance_level: float = 0.05,
    seed: int = 42,
) -> RealityCheckResult:
    """
    White's Reality Check with Hansen's Stepwise SPA enhancement.

    Tests H0: No strategy beats the benchmark after accounting for data snooping.

    Uses stationary bootstrap (Politis & Romano, 1994) to preserve
    autocorrelation structure in the block-resampled returns.

    Args:
        returns_matrix: Shape (T, N) — each column is a strategy's excess returns.
        benchmark_returns: Benchmark return series (default: zero/buy-and-hold).
        n_bootstrap: Number of bootstrap replications.
        block_size: Expected block length for stationary bootstrap.
        significance_level: p-value threshold.
        seed: Random seed for reproducibility.

    Returns:
        RealityCheckResult with bootstrap p-value.
    """
    T, N = returns_matrix.shape
    rng = np.random.default_rng(seed)

    # Excess returns over benchmark
    if benchmark_returns is not None:
        excess = returns_matrix - benchmark_returns.reshape(-1, 1)
    else:
        excess = returns_matrix

    # Observed test statistic: max average excess return
    avg_excess = excess.mean(axis=0)
    observed_max = avg_excess.max()

    # Stationary bootstrap
    bootstrap_maxes = np.zeros(n_bootstrap)
    prob_continue = 1 - 1 / block_size

    for b in range(n_bootstrap):
        # Generate bootstrap indices using geometric block sampling
        indices = np.zeros(T, dtype=int)
        indices[0] = rng.integers(0, T)
        for t in range(1, T):
            if rng.random() < prob_continue:
                indices[t] = (indices[t - 1] + 1) % T
            else:
                indices[t] = rng.integers(0, T)

        # Bootstrap sample
        boot_excess = excess[indices]
        boot_avg = boot_excess.mean(axis=0)

        # Center to enforce null hypothesis (no outperformance)
        bootstrap_maxes[b] = (boot_avg - avg_excess).max()

    # p-value: fraction of bootstrap maxes >= observed
    p_value = float(np.mean(bootstrap_maxes >= observed_max))

    # Hansen's SPA: consistent p-value (accounts for poor strategies)
    # Only count strategies with positive mean (stepwise adjustment)
    positive_mask = avg_excess > 0
    if positive_mask.any():
        spa_observed = avg_excess[positive_mask].max()
        spa_boot_maxes = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            indices = np.zeros(T, dtype=int)
            indices[0] = rng.integers(0, T)
            for t in range(1, T):
                if rng.random() < prob_continue:
                    indices[t] = (indices[t - 1] + 1) % T
                else:
                    indices[t] = rng.integers(0, T)
            boot_excess = excess[indices][:, positive_mask]
            boot_avg = boot_excess.mean(axis=0)
            spa_boot_maxes[b] = (boot_avg - avg_excess[positive_mask]).max()
        consistent_p = float(np.mean(spa_boot_maxes >= spa_observed))
    else:
        consistent_p = 1.0

    return RealityCheckResult(
        best_strategy_return=float(observed_max),
        p_value=p_value,
        is_significant=p_value < significance_level,
        bootstrap_distribution=bootstrap_maxes,
        n_strategies=N,
        consistent_p_value=consistent_p,
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. Purged Time-Series Cross-Validation
# ──────────────────────────────────────────────────────────────────────────


def purged_time_series_cv(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, Dict[str, Any]], List[Dict]],
    params: Dict[str, Any],
    n_splits: int = 5,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
) -> PurgedCVResult:
    """
    Purged k-fold time-series cross-validation (de Prado, 2018).

    Unlike standard k-fold, this:
    1. Respects temporal ordering (no future data in training)
    2. Purges observations near train/test boundary to avoid leakage
    3. Adds embargo period after test set to prevent information bleed

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        strategy_fn: Function(df, params) → list[trade_dicts] (same interface as WF).
        params: Strategy parameters to validate.
        n_splits: Number of CV folds.
        purge_pct: Fraction of data to purge at train/test boundary.
        embargo_pct: Fraction of data to embargo after test set.

    Returns:
        PurgedCVResult with per-fold OOS metrics.
    """
    T = len(df)
    purge_size = max(1, int(T * purge_pct))
    embargo_size = max(1, int(T * embargo_pct))
    fold_size = T // n_splits

    splits = []
    oos_returns = []
    oos_sharpes = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min(test_start + fold_size, T)

        # Training: everything except test + purge + embargo
        train_mask = np.ones(T, dtype=bool)
        # Remove test period
        train_mask[test_start:test_end] = False
        # Purge before test
        purge_start = max(0, test_start - purge_size)
        train_mask[purge_start:test_start] = False
        # Embargo after test
        embargo_end = min(T, test_end + embargo_size)
        train_mask[test_end:embargo_end] = False

        train_indices = np.where(train_mask)[0]
        if len(train_indices) < 50:
            continue

        # Test data
        test_df = df.iloc[test_start:test_end].copy()
        if len(test_df) < 20:
            continue

        # Run strategy on test fold
        trades = strategy_fn(test_df, params)

        if not trades:
            oos_returns.append(0.0)
            oos_sharpes.append(0.0)
            splits.append({
                "fold": i, "test_start": test_start, "test_end": test_end,
                "n_trades": 0, "oos_return": 0.0, "oos_sharpe": 0.0,
            })
            continue

        # Compute OOS metrics
        trade_returns = np.array([t["return"] for t in trades])
        fold_return = float(np.sum(trade_returns))
        fold_sharpe = (float(np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252))
                       if np.std(trade_returns) > 1e-10 else 0.0)

        oos_returns.append(fold_return)
        oos_sharpes.append(fold_sharpe)
        splits.append({
            "fold": i, "test_start": test_start, "test_end": test_end,
            "n_trades": len(trades), "oos_return": fold_return, "oos_sharpe": fold_sharpe,
        })

    mean_ret = float(np.mean(oos_returns)) if oos_returns else 0.0
    std_ret = float(np.std(oos_returns)) if oos_returns else 0.0
    mean_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0

    return PurgedCVResult(
        splits=splits,
        mean_oos_return=mean_ret,
        std_oos_return=std_ret,
        mean_oos_sharpe=mean_sharpe,
        all_oos_returns=oos_returns,
        all_oos_sharpes=oos_sharpes,
        n_splits=len(splits),
    )


# ──────────────────────────────────────────────────────────────────────────
# 5. Comprehensive Validation Pipeline
# ──────────────────────────────────────────────────────────────────────────


def full_statistical_validation(
    df: pd.DataFrame,
    symbol: str,
    strategy_fn: Callable[[pd.DataFrame, Dict[str, Any]], List[Dict]],
    params: Dict[str, Any],
    n_trials: int = 1,
    returns_matrix: Optional[np.ndarray] = None,
    significance_level: float = 0.05,
    pbo_partitions: int = 10,
) -> StatisticalReport:
    """
    Run comprehensive statistical validation on a strategy configuration.

    Combines DSR, PBO, White's Reality Check, and Purged CV into a single
    validation pipeline with pass/fail decision.

    Args:
        df: OHLCV DataFrame.
        symbol: Symbol identifier.
        strategy_fn: Strategy function compatible with WalkForward interface.
        params: Parameters to validate.
        n_trials: Number of parameter sets tested (for DSR multiple-testing correction).
        returns_matrix: (T, N) matrix of all strategy variants' returns (for PBO/WRC).
            If None, only DSR and Purged CV are computed.
        significance_level: p-value threshold for all tests.
        pbo_partitions: Number of partitions for PBO CSCV.

    Returns:
        StatisticalReport with all test results and overall pass/fail.
    """
    from src.config.backtest_params import get_backtest_config
    config = get_backtest_config(symbol)

    report = StatisticalReport(
        symbol=symbol,
        strategy_name=f"EMA({params.get('fast_ema', '?')}/{params.get('slow_ema', '?')})",
    )
    rejection_reasons = []

    # Run strategy to get trade returns
    trades = strategy_fn(df, params)
    if not trades or len(trades) < 5:
        report.rejection_reasons = ["Insufficient trades for statistical validation"]
        return report

    trade_returns = np.array([t["return"] for t in trades])

    # 1. Deflated Sharpe Ratio
    ann_factor = np.sqrt(config.get("annualization_periods", 252))
    dsr = deflated_sharpe_ratio(
        trade_returns,
        n_trials=max(n_trials, 1),
        annualization_factor=ann_factor,
        significance_level=significance_level,
    )
    report.dsr = dsr
    if not dsr.is_significant:
        rejection_reasons.append(
            f"DSR not significant: p={dsr.p_value:.4f} > {significance_level}"
        )

    # 2. PBO (only if returns_matrix provided)
    if returns_matrix is not None and returns_matrix.shape[1] >= 2:
        pbo = probability_of_backtest_overfitting(
            returns_matrix, n_partitions=pbo_partitions,
        )
        report.pbo = pbo
        if pbo.is_overfit:
            rejection_reasons.append(f"PBO indicates overfitting: {pbo.pbo:.1%} > 50%")

    # 3. White's Reality Check (only if returns_matrix provided)
    if returns_matrix is not None and returns_matrix.shape[1] >= 2:
        wrc = whites_reality_check(
            returns_matrix,
            n_bootstrap=500,
            significance_level=significance_level,
        )
        report.reality_check = wrc
        if not wrc.is_significant:
            rejection_reasons.append(
                f"White's RC: no strategy significantly beats benchmark (p={wrc.p_value:.4f})"
            )

    # 4. Purged Time-Series CV
    if len(df) >= 200:
        pcv = purged_time_series_cv(
            df, strategy_fn, params,
            n_splits=5, purge_pct=0.01, embargo_pct=0.01,
        )
        report.purged_cv = pcv
        if pcv.mean_oos_return < -0.05:
            rejection_reasons.append(
                f"Purged CV negative: mean OOS return = {pcv.mean_oos_return:.4f}"
            )

    # Overall decision
    report.rejection_reasons = rejection_reasons
    report.overall_pass = len(rejection_reasons) == 0

    logger.info(
        "statistical_validation.complete",
        symbol=symbol,
        passed=report.overall_pass,
        dsr_p=round(dsr.p_value, 4),
        n_reasons=len(rejection_reasons),
    )

    return report
