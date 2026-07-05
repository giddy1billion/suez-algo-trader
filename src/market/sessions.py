"""
Trading Sessions — explicit session types and definitions.

A session represents a distinct trading period within a calendar day.
Different sessions may have different liquidity characteristics, rules,
and risk parameters.
"""

from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Optional


class SessionType(str, Enum):
    """Types of trading sessions."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    ALWAYS_OPEN = "always_open"


@dataclass(frozen=True)
class TradingSession:
    """
    A trading session definition.

    Attributes:
        session_type: The type of this session.
        open_time: Session open time (in exchange local time).
        close_time: Session close time (in exchange local time).
        name: Human-readable session name.
        tradable: Whether orders can be placed during this session.
        full_liquidity: Whether full market depth is available.
    """
    session_type: SessionType
    open_time: time
    close_time: time
    name: str = ""
    tradable: bool = True
    full_liquidity: bool = True

    @property
    def duration_minutes(self) -> int:
        """Duration of the session in minutes."""
        open_mins = self.open_time.hour * 60 + self.open_time.minute
        close_mins = self.close_time.hour * 60 + self.close_time.minute
        if close_mins > open_mins:
            return close_mins - open_mins
        # Wraps midnight
        return (24 * 60 - open_mins) + close_mins


# Standard NYSE sessions
NYSE_PRE_MARKET = TradingSession(
    session_type=SessionType.PRE_MARKET,
    open_time=time(4, 0),
    close_time=time(9, 30),
    name="NYSE Pre-Market",
    tradable=True,
    full_liquidity=False,
)

NYSE_REGULAR = TradingSession(
    session_type=SessionType.REGULAR,
    open_time=time(9, 30),
    close_time=time(16, 0),
    name="NYSE Regular Session",
    tradable=True,
    full_liquidity=True,
)

NYSE_AFTER_HOURS = TradingSession(
    session_type=SessionType.AFTER_HOURS,
    open_time=time(16, 0),
    close_time=time(20, 0),
    name="NYSE After-Hours",
    tradable=True,
    full_liquidity=False,
)

CRYPTO_ALWAYS_OPEN = TradingSession(
    session_type=SessionType.ALWAYS_OPEN,
    open_time=time(0, 0),
    close_time=time(0, 0),  # 24 hours
    name="Crypto 24/7",
    tradable=True,
    full_liquidity=True,
)


def get_nyse_sessions() -> list[TradingSession]:
    """Get all NYSE trading sessions in chronological order."""
    return [NYSE_PRE_MARKET, NYSE_REGULAR, NYSE_AFTER_HOURS]


def get_current_session_type(
    current_time: time,
    sessions: list[TradingSession],
) -> SessionType:
    """
    Determine which session a given time falls into.

    Args:
        current_time: Time to check (in exchange local time).
        sessions: Ordered list of sessions for the exchange.

    Returns:
        SessionType for the matching session, or CLOSED if no match.
    """
    for session in sessions:
        if session.session_type == SessionType.ALWAYS_OPEN:
            return SessionType.ALWAYS_OPEN
        if session.open_time <= current_time < session.close_time:
            return session.session_type
    return SessionType.CLOSED
