"""
Holiday calendar management.

Provides holiday data for exchanges. Designed as an extensible interface
so that calendars can be loaded from configuration, databases, or external
services rather than being hardcoded.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Set


class HolidayCalendar(ABC):
    """
    Abstract interface for holiday calendars.

    Implementations provide holiday dates for a specific exchange.
    A production system should back this with a maintained data source
    rather than hardcoded dates.
    """

    @abstractmethod
    def is_holiday(self, dt: date) -> bool:
        """Check if a given date is a holiday."""
        ...

    @abstractmethod
    def holidays_in_range(self, start: date, end: date) -> Set[date]:
        """Get all holidays in a date range (inclusive)."""
        ...

    @abstractmethod
    def next_holiday(self, after: date) -> date:
        """Get the next holiday after the given date."""
        ...


class NYSEHolidayCalendar(HolidayCalendar):
    """
    NYSE holiday calendar.

    Contains all NYSE-observed holidays from 2024 through 2027.
    For production use beyond this range, integrate with an external
    holiday data provider or extend the dataset.
    """

    # NYSE observed holidays — sourced from NYSE holiday schedule
    _HOLIDAYS: Set[date] = {
        # 2024
        date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
        date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
        date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
        date(2024, 12, 25),
        # 2025
        date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
        date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
        date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
        date(2025, 12, 25),
        # 2026
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
        date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
        date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
        date(2026, 12, 25),
        # 2027
        date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
        date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
        date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
        date(2027, 12, 24),
    }

    def is_holiday(self, dt: date) -> bool:
        """Check if a given date is an NYSE holiday."""
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt in self._HOLIDAYS

    def holidays_in_range(self, start: date, end: date) -> Set[date]:
        """Get all NYSE holidays in a date range (inclusive)."""
        return {h for h in self._HOLIDAYS if start <= h <= end}

    def next_holiday(self, after: date) -> date:
        """Get the next NYSE holiday after the given date."""
        future = sorted(h for h in self._HOLIDAYS if h > after)
        if future:
            return future[0]
        raise ValueError(f"No holidays found after {after} in calendar data")


class NoHolidayCalendar(HolidayCalendar):
    """Calendar with no holidays (e.g., crypto markets)."""

    def is_holiday(self, dt: date) -> bool:
        return False

    def holidays_in_range(self, start: date, end: date) -> Set[date]:
        return set()

    def next_holiday(self, after: date) -> date:
        raise ValueError("No holidays in this calendar")
