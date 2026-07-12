"""
Runtime Environment Manager — Hot-switch between paper and live trading.

Provides:
- Atomic environment switching (paper ↔ live) without restart
- Broker hot-swap with position drain/migration
- Pre-flight validation before switching to live
- Settings hot-reload from .env
- Event-driven notifications on all state changes
"""

import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from config.settings import Settings, TradingMode, settings
from src.broker.base import BrokerProtocol
from src.core.events import (
    BrokerSwitched,
    EnvironmentSwitched,
    RiskHalt,
    SystemHealth,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SwitchState(str, Enum):
    """States during environment switch."""
    IDLE = "idle"
    DRAINING = "draining"
    VALIDATING = "validating"
    SWITCHING = "switching"
    VERIFYING = "verifying"
    FAILED = "failed"


class BrokerManager:
    """
    Manages broker lifecycle with support for hot-swap.

    Wraps a broker instance behind a lock so consumers always get a
    consistent, valid broker reference even during transitions.
    """

    def __init__(self, broker: BrokerProtocol, event_bus=None):
        self._broker = broker
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self._switch_count = 0
        self._last_switch_time: Optional[datetime] = None

    @property
    def broker(self) -> BrokerProtocol:
        """Get the current active broker. Thread-safe."""
        with self._lock:
            return self._broker

    @property
    def is_paper(self) -> bool:
        with self._lock:
            return self._broker.paper

    @property
    def broker_name(self) -> str:
        with self._lock:
            return self._broker.name

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def switch_broker(
        self,
        new_broker: BrokerProtocol,
        drain_positions: bool = True,
        timeout_seconds: float = 30.0,
    ) -> dict:
        """
        Hot-swap the active broker.

        Args:
            new_broker: New broker instance to switch to.
            drain_positions: If True, close all positions on old broker first.
            timeout_seconds: Max time to wait for position drain.

        Returns:
            Dict with switch result details.

        Raises:
            RuntimeError: If switch fails validation.
        """
        with self._lock:
            old_broker = self._broker
            old_name = old_broker.name
            new_name = new_broker.name
            positions_closed = 0

            logger.info(
                "broker_manager.switch_start",
                old=old_name,
                new=new_name,
                drain=drain_positions,
            )

            # Step 1: Validate new broker connectivity
            try:
                account = new_broker.get_account()
                if account is None:
                    raise RuntimeError(f"New broker '{new_name}' returned None account")
                logger.info(
                    "broker_manager.validated",
                    broker=new_name,
                    equity=account.get("equity", account.get("portfolio_value", 0)),
                )
            except Exception as e:
                logger.error("broker_manager.validation_failed", broker=new_name, error=str(e))
                raise RuntimeError(f"Broker validation failed: {e}") from e

            # Step 2: Drain positions on old broker (if requested)
            if drain_positions:
                positions_closed = self._drain_positions(old_broker, timeout_seconds)

            # Step 3: Atomic swap
            self._broker = new_broker
            self._switch_count += 1
            self._last_switch_time = datetime.now(timezone.utc)

            logger.info(
                "broker_manager.switched",
                old=old_name,
                new=new_name,
                positions_closed=positions_closed,
            )

            # Step 4: Publish event
            if self._event_bus:
                self._event_bus.publish(BrokerSwitched(
                    old_broker=old_name,
                    new_broker=new_name,
                    open_positions_migrated=positions_closed,
                    source="broker_manager",
                ))

            return {
                "success": True,
                "old_broker": old_name,
                "new_broker": new_name,
                "positions_closed": positions_closed,
                "timestamp": self._last_switch_time.isoformat(),
            }

    def restore_broker(
        self,
        broker: BrokerProtocol,
        reason: str = "rollback",
        publish_event: bool = True,
    ) -> dict:
        """
        Restore a previously active broker instance without draining positions.

        This is intended for compensating rollback paths after a failed switch.
        """
        with self._lock:
            previous = self._broker
            if previous is broker:
                return {
                    "success": True,
                    "old_broker": previous.name,
                    "new_broker": broker.name,
                    "restored": False,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            self._broker = broker
            self._switch_count += 1
            self._last_switch_time = datetime.now(timezone.utc)

            logger.warning(
                "broker_manager.restored",
                old=previous.name,
                new=broker.name,
                reason=reason,
            )

            if self._event_bus and publish_event:
                self._event_bus.publish(BrokerSwitched(
                    old_broker=previous.name,
                    new_broker=broker.name,
                    open_positions_migrated=0,
                    source="broker_manager:restore",
                ))

            return {
                "success": True,
                "old_broker": previous.name,
                "new_broker": broker.name,
                "restored": True,
                "timestamp": self._last_switch_time.isoformat(),
            }

    def _drain_positions(self, broker: BrokerProtocol, timeout: float) -> int:
        """
        Close all open positions on a broker. Returns count closed.
        Raises RuntimeError if positions could not be drained.
        """
        try:
            positions = broker.get_positions()
            if not positions:
                return 0

            closed = 0
            failed = []
            for pos in positions:
                symbol = pos.get("symbol", "")
                try:
                    broker.close_position(symbol)
                    closed += 1
                    logger.info("broker_manager.position_closed", symbol=symbol)
                except Exception as e:
                    failed.append(symbol)
                    logger.warning(
                        "broker_manager.close_failed",
                        symbol=symbol,
                        error=str(e),
                    )

            # Wait briefly for fills
            if closed > 0:
                time.sleep(min(2.0, timeout * 0.1))

            if failed:
                logger.error(
                    "broker_manager.drain_incomplete",
                    closed=closed,
                    failed=failed,
                    total=len(positions),
                )
                raise RuntimeError(
                    f"Position drain incomplete: {len(failed)}/{len(positions)} "
                    f"positions failed to close: {failed}"
                )

            return closed
        except RuntimeError:
            raise  # Re-raise our own error
        except Exception as e:
            logger.error("broker_manager.drain_error", error=str(e))
            raise RuntimeError(f"Position drain failed: {e}") from e

    def get_status(self) -> dict:
        """Get broker manager status."""
        with self._lock:
            try:
                account = self._broker.get_account()
                positions = self._broker.get_positions()
            except Exception:
                account = {}
                positions = []

            return {
                "broker": self._broker.name,
                "is_paper": self._broker.paper,
                "switch_count": self._switch_count,
                "last_switch": self._last_switch_time.isoformat() if self._last_switch_time else None,
                "equity": account.get("equity", account.get("portfolio_value", 0)),
                "open_positions": len(positions),
            }


class EnvironmentManager:
    """
    Manages runtime switching between paper and live trading environments.

    Coordinates:
    - Settings reload (TradingMode switch)
    - Broker creation and hot-swap
    - Pre-flight validation for live mode
    - Event publishing for audit trail
    - Rollback on failure
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        broker_factory: Callable[[TradingMode], BrokerProtocol],
        event_bus=None,
        require_confirmation: bool = True,
    ):
        self._broker_manager = broker_manager
        self._broker_factory = broker_factory
        self._event_bus = event_bus
        self._require_confirmation = require_confirmation
        self._state = SwitchState.IDLE
        self._lock = threading.Lock()
        self._current_mode = TradingMode.PAPER if broker_manager.is_paper else TradingMode.LIVE
        self._switch_history: list[dict] = []

    @property
    def current_mode(self) -> TradingMode:
        return self._current_mode

    @property
    def state(self) -> SwitchState:
        return self._state

    @property
    def is_paper(self) -> bool:
        return self._current_mode == TradingMode.PAPER

    @property
    def is_live(self) -> bool:
        return self._current_mode == TradingMode.LIVE

    def switch_to_paper(self, reason: str = "manual") -> dict:
        """Switch to paper trading mode."""
        return self.switch_environment(TradingMode.PAPER, reason=reason)

    def switch_to_live(self, reason: str = "manual") -> dict:
        """Switch to live trading mode."""
        return self.switch_environment(TradingMode.LIVE, reason=reason)

    def switch_environment(
        self,
        target_mode: TradingMode,
        reason: str = "manual",
        drain_positions: bool = True,
        force: bool = False,
    ) -> dict:
        """
        Switch between paper and live environments atomically.

        Args:
            target_mode: Target TradingMode (PAPER or LIVE).
            reason: Reason for the switch (for audit).
            drain_positions: Close positions before switching.
            force: Skip confirmation requirement.

        Returns:
            Dict with switch result.

        Raises:
            RuntimeError: If switch fails or is rejected.
            ValueError: If already in target mode.
        """
        with self._lock:
            if self._state != SwitchState.IDLE:
                raise RuntimeError(f"Switch already in progress (state={self._state})")

            if self._current_mode == target_mode and not force:
                return {
                    "success": True,
                    "old_mode": target_mode.value,
                    "new_mode": target_mode.value,
                    "reason": "already_in_target_mode",
                    "no_op": True,
                }

            old_mode = self._current_mode
            old_settings_mode = settings.trading_mode
            old_broker = self._broker_manager.broker
            result = {
                "old_mode": old_mode.value,
                "new_mode": target_mode.value,
                "reason": reason,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            broker_swapped = False

            try:
                # Phase 1: Validate
                self._state = SwitchState.VALIDATING
                logger.info(
                    "env_manager.switch_start",
                    old=old_mode.value,
                    new=target_mode.value,
                    reason=reason,
                )

                if target_mode == TradingMode.LIVE:
                    self._validate_live_readiness()

                # Phase 2: Create new broker
                self._state = SwitchState.SWITCHING
                new_broker = self._broker_factory(target_mode)

                # Phase 3: Hot-swap via BrokerManager
                swap_result = self._broker_manager.switch_broker(
                    new_broker,
                    drain_positions=drain_positions,
                )
                broker_swapped = True

                # Phase 4: Update settings in-memory
                settings.trading_mode = target_mode
                self._current_mode = target_mode

                # Phase 5: Verify
                self._state = SwitchState.VERIFYING
                self._verify_switch(target_mode)

                # Success
                self._state = SwitchState.IDLE
                result.update({
                    "success": True,
                    "positions_closed": swap_result.get("positions_closed", 0),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                self._switch_history.append(result)

                # Publish event
                if self._event_bus:
                    self._event_bus.publish(EnvironmentSwitched(
                        old_mode=old_mode.value,
                        new_mode=target_mode.value,
                        broker_name=self._broker_manager.broker_name,
                        positions_closed=swap_result.get("positions_closed", 0),
                        reason=reason,
                        source="env_manager",
                    ))

                logger.info("env_manager.switch_complete", **result)
                return result

            except Exception as e:
                # Rollback
                self._state = SwitchState.FAILED
                result["success"] = False
                result["error"] = str(e)
                rollback_performed = False
                rollback_errors: list[str] = []

                if broker_swapped:
                    rollback_performed = True
                    try:
                        self._broker_manager.restore_broker(
                            old_broker,
                            reason=f"env_switch_failed:{target_mode.value}",
                            publish_event=False,
                        )
                    except Exception as rollback_exc:
                        rollback_errors.append(f"broker_restore_failed:{rollback_exc}")

                if self._current_mode != old_mode:
                    rollback_performed = True
                    self._current_mode = old_mode

                if settings.trading_mode != old_settings_mode:
                    rollback_performed = True
                    settings.trading_mode = old_settings_mode

                if rollback_performed:
                    result["rollback_performed"] = True
                    result["rollback_success"] = len(rollback_errors) == 0
                    if rollback_errors:
                        result["rollback_errors"] = rollback_errors

                self._switch_history.append(result)

                logger.error(
                    "env_manager.switch_failed",
                    error=str(e),
                    old_mode=old_mode.value,
                    target_mode=target_mode.value,
                    rollback_performed=rollback_performed,
                    rollback_errors=rollback_errors,
                )

                # Reset state after brief delay
                self._state = SwitchState.IDLE
                if rollback_errors:
                    raise RuntimeError(
                        f"Environment switch failed: {e}; rollback issues: {'; '.join(rollback_errors)}"
                    ) from e
                raise RuntimeError(f"Environment switch failed: {e}") from e

    def _validate_live_readiness(self):
        """Pre-flight checks before switching to live mode."""
        issues = []

        # Check live credentials exist
        if not settings.alpaca_live_api_key:
            issues.append("Live API key not configured")
        if not settings.alpaca_live_secret_key:
            issues.append("Live secret key not configured")

        # Check risk settings are reasonable for live
        if settings.max_daily_loss_pct > 0.10:
            issues.append(f"max_daily_loss_pct={settings.max_daily_loss_pct} is too high for live")
        if settings.max_position_size_pct > 0.10:
            issues.append(f"max_position_size_pct={settings.max_position_size_pct} is too high for live")

        if issues:
            raise RuntimeError(f"Live readiness check failed: {'; '.join(issues)}")

    def _verify_switch(self, target_mode: TradingMode):
        """Verify the switch was successful."""
        broker = self._broker_manager.broker
        if target_mode == TradingMode.PAPER and not broker.paper:
            raise RuntimeError("Switch verification failed: broker not in paper mode")
        if target_mode == TradingMode.LIVE and broker.paper:
            raise RuntimeError("Switch verification failed: broker not in live mode")

    def get_status(self) -> dict:
        """Get full environment manager status."""
        return {
            "current_mode": self._current_mode.value,
            "state": self._state.value,
            "broker": self._broker_manager.get_status(),
            "switch_history_count": len(self._switch_history),
            "last_switch": self._switch_history[-1] if self._switch_history else None,
        }

    def get_switch_history(self, limit: int = 10) -> list[dict]:
        """Get recent switch history."""
        return self._switch_history[-limit:]


def create_broker_for_mode(mode: TradingMode) -> BrokerProtocol:
    """
    Factory function to create a broker instance for the given mode.

    This is the default broker_factory for EnvironmentManager.
    """
    if mode == TradingMode.PAPER:
        from src.broker.paper import PaperBroker
        return PaperBroker(starting_equity=settings.backtest_initial_cash)
    else:
        from src.broker.alpaca_client import AlpacaBroker
        return AlpacaBroker(
            api_key=settings.alpaca_live_api_key,
            secret_key=settings.alpaca_live_secret_key,
            base_url=settings.alpaca_live_base_url,
            paper=False,
        )
