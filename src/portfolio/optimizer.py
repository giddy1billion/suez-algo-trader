"""
Portfolio Optimizer — Capital allocation and portfolio construction.

Elevates the system from individual trade selection to portfolio-level
capital allocation using modern portfolio theory approaches.

Strategies:
- Equal Weight
- Risk Parity (inverse volatility weighting)
- Minimum Variance (minimize portfolio variance)
- Kelly Criterion (growth-optimal allocation)
- Hierarchical Risk Parity (HRP via clustering)
- Volatility Targeting (scale to target vol)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    KELLY = "kelly"
    HRP = "hrp"
    VOL_TARGET = "vol_target"


@dataclass
class PortfolioAllocation:
    """Result of portfolio optimization."""
    weights: Dict[str, float]  # symbol → weight (0.0 to 1.0)
    method: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def total_weight(self) -> float:
        return sum(self.weights.values())

    def get_position_sizes(self, capital: float) -> Dict[str, float]:
        """Convert weights to dollar amounts."""
        return {sym: w * capital for sym, w in self.weights.items()}


class PortfolioOptimizer:
    """
    Multi-strategy portfolio optimizer.

    Usage:
        optimizer = PortfolioOptimizer()
        returns = pd.DataFrame(...)  # columns = symbols, rows = daily returns
        allocation = optimizer.optimize(returns, method=OptimizationMethod.RISK_PARITY)
        sizes = allocation.get_position_sizes(capital=100000)
    """

    def __init__(self, risk_free_rate: float = 0.05, max_weight: float = 0.25,
                 min_weight: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight  # Maximum allocation per asset
        self.min_weight = min_weight

    def optimize(self, returns: pd.DataFrame,
                 method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
                 target_vol: float = 0.15,
                 **kwargs) -> PortfolioAllocation:
        """
        Optimize portfolio weights.

        Args:
            returns: DataFrame of asset returns (columns = symbols).
            method: Optimization strategy.
            target_vol: Target annualized volatility (for VOL_TARGET method).

        Returns:
            PortfolioAllocation with optimized weights.
        """
        if returns.empty or returns.shape[1] == 0:
            return PortfolioAllocation(weights={}, method=method.value)

        symbols = returns.columns.tolist()

        if method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(symbols)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(returns)
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance(returns)
        elif method == OptimizationMethod.KELLY:
            weights = self._kelly(returns)
        elif method == OptimizationMethod.HRP:
            weights = self._hrp(returns)
        elif method == OptimizationMethod.VOL_TARGET:
            weights = self._vol_target(returns, target_vol)
        else:
            weights = self._equal_weight(symbols)

        # Apply constraints
        weights = self._apply_constraints(weights)

        # Compute portfolio metrics
        w = np.array([weights[s] for s in symbols])
        mu = returns.mean().values * 252  # annualized
        cov = returns.cov().values * 252

        port_return = float(w @ mu)
        port_vol = float(np.sqrt(w @ cov @ w))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        return PortfolioAllocation(
            weights=weights,
            method=method.value,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            metadata={"n_assets": len(symbols), "target_vol": target_vol},
        )

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """1/N allocation."""
        n = len(symbols)
        return {s: 1.0 / n for s in symbols}

    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Inverse-volatility weighting (approximate risk parity)."""
        vols = returns.std() * np.sqrt(252)
        inv_vols = 1.0 / vols.replace(0, np.inf)
        weights_raw = inv_vols / inv_vols.sum()
        return dict(zip(returns.columns, weights_raw.values))

    def _min_variance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Minimum variance portfolio (analytical solution for long-only).
        Uses the inverse covariance method.
        """
        cov = returns.cov().values * 252
        n = cov.shape[0]

        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Singular matrix — fall back to equal weight
            return self._equal_weight(returns.columns.tolist())

        ones = np.ones(n)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)

        # Clip negative weights (long-only constraint)
        w = np.maximum(w, 0.0)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(n) / n

        return dict(zip(returns.columns, w))

    def _kelly(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Kelly Criterion (half-Kelly for safety).
        Kelly fraction: f* = μ / σ² for each asset independently,
        then normalized. Uses half-Kelly for conservatism.
        """
        mu = returns.mean() * 252
        var = returns.var() * 252

        # Kelly fraction per asset (half-Kelly)
        kelly_fracs = (mu / var.replace(0, np.inf)) * 0.5

        # Only invest in positive expectancy assets
        kelly_fracs = kelly_fracs.clip(lower=0.0)

        total = kelly_fracs.sum()
        if total > 0:
            weights = kelly_fracs / total
        else:
            weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

        return dict(zip(returns.columns, weights.values))

    def _hrp(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (simplified implementation).

        Steps:
        1. Compute correlation distance matrix
        2. Hierarchical clustering (single linkage)
        3. Quasi-diagonalize
        4. Recursive bisection for weights
        """
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        corr = returns.corr()
        n = len(corr)

        if n <= 1:
            return self._equal_weight(returns.columns.tolist())

        # Distance matrix from correlation
        dist = np.sqrt(0.5 * (1 - corr.values))
        np.fill_diagonal(dist, 0.0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method='single')
        sort_idx = leaves_list(link)

        # Recursive bisection
        sorted_symbols = [returns.columns[i] for i in sort_idx]
        sorted_returns = returns[sorted_symbols]

        weights = self._hrp_recursive_bisection(sorted_returns)
        return weights

    def _hrp_recursive_bisection(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Recursive bisection for HRP weights."""
        symbols = returns.columns.tolist()
        n = len(symbols)

        if n == 1:
            return {symbols[0]: 1.0}

        # Split in half
        mid = n // 2
        left_syms = symbols[:mid]
        right_syms = symbols[mid:]

        # Cluster variance
        left_var = self._cluster_variance(returns[left_syms])
        right_var = self._cluster_variance(returns[right_syms])

        # Allocate inversely proportional to variance
        total_inv_var = (1.0 / left_var + 1.0 / right_var) if (left_var > 0 and right_var > 0) else 1.0
        alpha = (1.0 / left_var) / total_inv_var if left_var > 0 else 0.5

        # Recurse
        left_weights = self._hrp_recursive_bisection(returns[left_syms])
        right_weights = self._hrp_recursive_bisection(returns[right_syms])

        # Scale
        result = {}
        for s, w in left_weights.items():
            result[s] = w * alpha
        for s, w in right_weights.items():
            result[s] = w * (1.0 - alpha)

        return result

    def _cluster_variance(self, returns: pd.DataFrame) -> float:
        """Compute variance of an equal-weighted cluster."""
        if returns.shape[1] == 0:
            return 1.0
        w = np.ones(returns.shape[1]) / returns.shape[1]
        cov = returns.cov().values * 252
        return float(w @ cov @ w)

    def _vol_target(self, returns: pd.DataFrame, target_vol: float) -> Dict[str, float]:
        """
        Volatility targeting: scale equal-weight portfolio to target vol.
        """
        n = len(returns.columns)
        w_eq = np.ones(n) / n
        cov = returns.cov().values * 252
        port_vol = np.sqrt(w_eq @ cov @ w_eq)

        if port_vol > 0:
            scale = target_vol / port_vol
        else:
            scale = 1.0

        # Cap scale at 2x (don't lever more than 2x)
        scale = min(scale, 2.0)

        weights_scaled = w_eq * scale
        # Normalize to sum to 1 (or less if de-levered)
        total = weights_scaled.sum()
        if total > 1.0:
            weights_scaled = weights_scaled / total

        return dict(zip(returns.columns, weights_scaled))

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply max/min weight constraints with iterative clipping."""
        constrained = dict(weights)
        n_assets = len(constrained)
        
        if n_assets == 0:
            return constrained
        
        # If max_weight is too restrictive to allow sum=1, relax it
        effective_max = max(self.max_weight, 1.0 / n_assets)

        # Iteratively clip and redistribute
        for _ in range(50):
            violated = False
            capped_symbols = []
            free_symbols = []
            
            for s, w in constrained.items():
                if w > effective_max + 1e-12:
                    violated = True
                    capped_symbols.append(s)
                else:
                    free_symbols.append(s)
            
            if not violated:
                break
            
            # Cap violated assets and redistribute excess
            capped_total = effective_max * len(capped_symbols)
            remaining = 1.0 - capped_total
            free_total = sum(constrained[s] for s in free_symbols)
            
            for s in capped_symbols:
                constrained[s] = effective_max
            
            if free_symbols and free_total > 0:
                for s in free_symbols:
                    constrained[s] = constrained[s] / free_total * remaining
            elif free_symbols:
                for s in free_symbols:
                    constrained[s] = remaining / len(free_symbols)
        
        # Apply min_weight
        for s in constrained:
            constrained[s] = max(self.min_weight, constrained[s])
        
        # Final normalization
        total = sum(constrained.values())
        if total > 0 and abs(total - 1.0) > 1e-10:
            constrained = {s: w / total for s, w in constrained.items()}
        
        return constrained
