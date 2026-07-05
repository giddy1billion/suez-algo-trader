"""
Corporate Actions — data model for stock splits, dividends, and symbol changes.

Corporate actions affect historical price data and must be handled correctly
to ensure research integrity. This module provides the data model; integration
with the data loader is handled separately.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class CorporateActionType(str, Enum):
    """Types of corporate actions."""
    STOCK_SPLIT = "stock_split"
    REVERSE_SPLIT = "reverse_split"
    CASH_DIVIDEND = "cash_dividend"
    STOCK_DIVIDEND = "stock_dividend"
    SYMBOL_CHANGE = "symbol_change"
    DELISTING = "delisting"
    MERGER = "merger"
    SPINOFF = "spinoff"


@dataclass(frozen=True)
class CorporateAction:
    """
    A corporate action event.

    Attributes:
        symbol: The affected symbol.
        action_type: Type of corporate action.
        effective_date: Date the action takes effect.
        ratio: Split ratio (e.g., 4.0 for a 4:1 split, 0.1 for 1:10 reverse).
        amount: Dividend amount per share (for dividend actions).
        new_symbol: New symbol (for symbol changes/mergers).
        description: Human-readable description.
    """
    symbol: str
    action_type: CorporateActionType
    effective_date: date
    ratio: Optional[float] = None
    amount: Optional[float] = None
    new_symbol: Optional[str] = None
    description: str = ""

    @property
    def is_split(self) -> bool:
        """Whether this is a forward or reverse split."""
        return self.action_type in (
            CorporateActionType.STOCK_SPLIT,
            CorporateActionType.REVERSE_SPLIT,
        )

    @property
    def is_dividend(self) -> bool:
        """Whether this is a dividend action."""
        return self.action_type in (
            CorporateActionType.CASH_DIVIDEND,
            CorporateActionType.STOCK_DIVIDEND,
        )

    @property
    def price_adjustment_factor(self) -> float:
        """
        Factor to multiply historical prices by to adjust for this action.

        For splits: 1/ratio (prices go down, shares go up).
        For reverse splits: 1/ratio (prices go up, shares go down).
        For dividends: (price - amount) / price (approximation).
        For other actions: 1.0 (no adjustment).
        """
        if self.is_split and self.ratio:
            return 1.0 / self.ratio
        return 1.0


class CorporateActionRegistry:
    """
    Registry of corporate actions for backtesting and data adjustment.

    Usage:
        registry = CorporateActionRegistry()
        registry.add(CorporateAction(...))
        actions = registry.get_actions("AAPL", start, end)
    """

    def __init__(self):
        self._actions: list[CorporateAction] = []

    def add(self, action: CorporateAction) -> None:
        """Add a corporate action to the registry."""
        self._actions.append(action)

    def get_actions(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> list[CorporateAction]:
        """
        Get corporate actions for a symbol in a date range.

        Args:
            symbol: Symbol to query.
            start: Start date (inclusive). None means no lower bound.
            end: End date (inclusive). None means no upper bound.

        Returns:
            List of corporate actions, sorted by effective_date.
        """
        actions = [a for a in self._actions if a.symbol == symbol]
        if start:
            actions = [a for a in actions if a.effective_date >= start]
        if end:
            actions = [a for a in actions if a.effective_date <= end]
        return sorted(actions, key=lambda a: a.effective_date)

    def has_actions(self, symbol: str) -> bool:
        """Check if any corporate actions exist for a symbol."""
        return any(a.symbol == symbol for a in self._actions)
