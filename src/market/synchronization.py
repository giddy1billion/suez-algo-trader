"""
Cross-market synchronization policies.

Defines explicit behavior for how mixed-asset portfolios handle
periods when some markets are open and others are closed.

No hidden assumptions — every cross-calendar interaction is governed
by a configurable policy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from src.market.instruments import AssetClass, Instrument
from src.market.registry import classify_symbol


class SyncMode(str, Enum):
    """
    Synchronization modes for cross-calendar portfolio management.

    STRICT: Only process signals when ALL constituent markets are open.
        Safest for research. May miss opportunities.

    RELAXED: Process signals for any market that is currently open.
        Most responsive. May introduce asymmetric exposure.

    CARRY_FORWARD: Forward-fill closed-market positions at last known price.
        Common in backtesting. Can introduce stale-price artifacts.

    MARKET_LOCAL: Each asset follows its own calendar independently.
        Most realistic for execution. Portfolio metrics only computed
        when all markets have fresh data.
    """
    STRICT = "strict"
    RELAXED = "relaxed"
    CARRY_FORWARD = "carry_forward"
    MARKET_LOCAL = "market_local"


@dataclass
class SynchronizationPolicy:
    """
    Configuration for cross-market synchronization behavior.

    Attributes:
        mode: The synchronization mode to use.
        max_forward_fill: Maximum number of bars to forward-fill for
            closed markets (only used in CARRY_FORWARD mode).
        rebalance_requires_all_open: Whether portfolio rebalancing
            requires all constituent markets to be open.
        hedge_during_partial_close: Whether to hedge exposure when
            some markets are closed.
    """
    mode: SyncMode = SyncMode.MARKET_LOCAL
    max_forward_fill: int = 7
    rebalance_requires_all_open: bool = True
    hedge_during_partial_close: bool = False

    def should_forward_fill(self, instrument: Instrument) -> bool:
        """Determine if forward-fill is appropriate for this instrument."""
        if self.mode == SyncMode.STRICT:
            return False
        if self.mode == SyncMode.CARRY_FORWARD:
            return True
        if self.mode == SyncMode.MARKET_LOCAL:
            return False
        # RELAXED: forward-fill equities during off-hours only
        return instrument.is_equity

    def get_fill_limit(self, instrument: Instrument) -> Optional[int]:
        """
        Get the forward-fill limit for an instrument.

        Returns:
            Number of bars to forward-fill, or None for unlimited (crypto).
        """
        if instrument.trades_24_7:
            return None  # Crypto: fill all (should be continuous)
        return self.max_forward_fill


def align_multi_asset_data(
    data: dict[str, pd.DataFrame],
    policy: Optional[SynchronizationPolicy] = None,
) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    """
    Align multi-asset DataFrames according to a synchronization policy.

    Handles the fundamental challenge of mixing 24/7 crypto with
    session-based equity data.

    Args:
        data: Dict of symbol -> DataFrame with DatetimeIndex.
        policy: Synchronization policy. Defaults to MARKET_LOCAL.

    Returns:
        Tuple of (common_index, aligned_data_dict).
    """
    if policy is None:
        policy = SynchronizationPolicy()

    if not data:
        return pd.DatetimeIndex([]), {}

    # Build union index
    all_indices = [df.index for df in data.values()]
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)
    common_index = common_index.sort_values()

    aligned = {}
    for symbol, df in data.items():
        instrument = classify_symbol(symbol)
        reindexed = df.reindex(common_index)

        if policy.should_forward_fill(instrument):
            fill_limit = policy.get_fill_limit(instrument)
            if instrument.trades_24_7:
                # Crypto: forward-fill all gaps (should be continuous)
                reindexed = reindexed.ffill()
            else:
                # Equities: limited forward-fill within sessions
                reindexed = reindexed.ffill(limit=fill_limit)
        elif policy.mode == SyncMode.MARKET_LOCAL:
            # Market-local: fill crypto (should be continuous), limit equities
            if instrument.trades_24_7:
                reindexed = reindexed.ffill()
            else:
                reindexed = reindexed.ffill(limit=policy.max_forward_fill)

        aligned[symbol] = reindexed

    return common_index, aligned


def group_by_calendar(symbols: list[str]) -> dict[str, list[str]]:
    """
    Group symbols by their calendar.

    Backward-compatible utility.

    Returns:
        Dict mapping calendar name -> list of symbols.
    """
    groups: dict[str, list[str]] = {}
    for symbol in symbols:
        instrument = classify_symbol(symbol)
        cal = instrument.calendar
        if cal not in groups:
            groups[cal] = []
        groups[cal].append(symbol)
    return groups
