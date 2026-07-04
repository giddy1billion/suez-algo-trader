"""
Market Constraints — tradable instrument rules.

Moves market-specific rules (shortable, marginable, fractional, etc.)
out of execution logic and into the instrument/market definition layer.

The execution engine should ask instrument.constraints rather than hardcoding.
"""

from dataclasses import dataclass
from typing import Optional

from src.market.instruments import Instrument


@dataclass(frozen=True)
class MarketConstraints:
    """
    Trading constraints for an instrument.

    These are the rules and limits that the execution engine must respect.
    They are derived from the instrument's exchange and broker capabilities.

    Attributes:
        shortable: Whether short selling is allowed.
        marginable: Whether margin trading is allowed.
        fractional: Whether fractional quantities are supported.
        min_tick: Minimum price increment.
        lot_size: Minimum order quantity increment.
        min_order_value: Minimum order value in settlement currency.
        max_order_value: Maximum single order value.
        settlement_days: T+N settlement period.
        borrow_required: Whether a borrow locate is needed for shorts.
        max_leverage: Maximum allowed leverage ratio.
        day_trade_restricted: Whether pattern day trader rules apply.
    """
    shortable: bool = True
    marginable: bool = True
    fractional: bool = False
    min_tick: float = 0.01
    lot_size: float = 1.0
    min_order_value: float = 1.0
    max_order_value: Optional[float] = None
    settlement_days: int = 2
    borrow_required: bool = False
    max_leverage: float = 2.0
    day_trade_restricted: bool = False

    def validate_quantity(self, quantity: float) -> bool:
        """Check if a quantity respects lot size constraints."""
        if self.lot_size <= 0:
            return True
        if self.fractional:
            return quantity > 0
        # Check if quantity is a multiple of lot size (with tolerance)
        remainder = quantity % self.lot_size
        return remainder < 1e-10 or (self.lot_size - remainder) < 1e-10

    def round_quantity(self, quantity: float) -> float:
        """Round a quantity to the nearest valid lot size."""
        if self.fractional:
            return quantity
        if self.lot_size <= 0:
            return quantity
        result = round(quantity / self.lot_size) * self.lot_size
        # Fix floating point precision
        decimals = len(str(self.lot_size).rstrip('0').split('.')[-1]) if '.' in str(self.lot_size) else 0
        return round(result, decimals)

    def validate_price(self, price: float) -> bool:
        """Check if a price respects tick size constraints."""
        if self.min_tick <= 0:
            return True
        remainder = price % self.min_tick
        return remainder < 1e-10 or (self.min_tick - remainder) < 1e-10

    def round_price(self, price: float) -> float:
        """Round a price to the nearest valid tick."""
        if self.min_tick <= 0:
            return price
        result = round(price / self.min_tick) * self.min_tick
        # Fix floating point precision
        decimals = len(str(self.min_tick).rstrip('0').split('.')[-1]) if '.' in str(self.min_tick) else 0
        return round(result, decimals)


def get_constraints(instrument: Instrument) -> MarketConstraints:
    """
    Derive market constraints from an instrument definition.

    Args:
        instrument: The instrument to get constraints for.

    Returns:
        MarketConstraints applicable to this instrument.
    """
    return MarketConstraints(
        shortable=instrument.shortable,
        marginable=instrument.marginable,
        fractional=instrument.fractional,
        min_tick=instrument.tick_size,
        lot_size=instrument.lot_size,
        settlement_days=instrument.settlement_days,
        borrow_required=not instrument.shortable,
        max_leverage=2.0 if instrument.marginable else 1.0,
        day_trade_restricted=instrument.is_equity,
    )
