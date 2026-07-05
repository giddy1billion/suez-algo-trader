"""
Runtime State — Centralized state accessible to all components.

Provides a thread-safe container for runtime state (pause, trading mode, etc.)
that components can inject and check without circular imports or global variables.

This replaces module-level globals like _bot_paused in telegram_bot.py with
a proper dependency-injected state object.
"""

import threading
from typing import Optional


class RuntimeState:
    """
    Thread-safe container for runtime state that all components can access.
    
    Centralizes state management so that:
    - ExecutionEngine can check if trading is paused
    - TelegramAuditForwarder can check if events should be suppressed
    - Telegram bot can set pause state without exposing globals
    """

    def __init__(self):
        """Initialize runtime state with defaults."""
        self._paused = False
        self._lock = threading.RLock()

    def is_paused(self) -> bool:
        """Check if trading is paused. Thread-safe."""
        with self._lock:
            return self._paused

    def set_paused(self, paused: bool) -> None:
        """Set pause state. Thread-safe."""
        with self._lock:
            self._paused = paused

    def pause(self) -> None:
        """Convenience method to pause trading."""
        self.set_paused(True)

    def resume(self) -> None:
        """Convenience method to resume trading."""
        self.set_paused(False)

    @property
    def paused(self) -> bool:
        """Property accessor for pause state."""
        return self.is_paused()

    @paused.setter
    def paused(self, value: bool) -> None:
        """Property setter for pause state."""
        self.set_paused(value)
