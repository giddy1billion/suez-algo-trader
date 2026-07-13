"""
Protective Exits — Shared risk module ensuring every position has SL/TP.

Every order entry path (signal-driven, manual Telegram, API) MUST flow
through this module so that stop-loss and take-profit levels are:
1. Computed (from strategy hint, ATR, or configurable defaults)
2. Recorded (in the position metadata for audit/reconciliation)
3. Submitted to the broker (as a bracket order when supported)

Design:
    - Single shared module used by ExecutionEngine and Telegram commands
    - Configurable via ProtectiveExitConfig dataclass
    - Never allows a naked position (no protective exits)
    - Falls back to percentage-based defaults when strategy doesn't provide
    - Emits telemetry events for configured/adjusted/executed SL/TP
"""

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProtectiveExitConfig:
    """Configuration for protective exit levels.

    All percentages are expressed as decimals (0.03 = 3%).
    """

    # Default stop-loss distance from entry price (used when strategy doesn't provide)
    default_stop_loss_pct: float = 0.03  # 3%
    # Default take-profit distance from entry price
    default_take_profit_pct: float = 0.06  # 6% (2:1 risk-reward)
    # Maximum allowed stop-loss distance (safety cap)
    max_stop_loss_pct: float = 0.10  # 10%
    # Minimum stop-loss distance (prevent too-tight stops)
    min_stop_loss_pct: float = 0.005  # 0.5%
    # Maximum take-profit distance
    max_take_profit_pct: float = 0.30  # 30%
    # Minimum take-profit distance
    min_take_profit_pct: float = 0.01  # 1%
    # Whether to enforce bracket orders (True) or allow plain market + monitor (False)
    enforce_bracket: bool = True
    # Minimum risk-reward ratio (TP distance / SL distance)
    min_risk_reward: float = 1.5


@dataclass
class ProtectiveExitLevels:
    """Computed protective exit levels for a position."""

    entry_price: float
    side: str  # "buy" or "sell"
    stop_loss: float
    take_profit: float
    stop_loss_pct: float  # Distance from entry as percentage
    take_profit_pct: float  # Distance from entry as percentage
    risk_reward_ratio: float
    source: str  # "strategy" | "atr" | "default" | "decision_contract"
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class ProtectiveExits:
    """
    Shared risk module that computes and validates SL/TP for every position.

    Usage:
        exits = ProtectiveExits(config)

        # For signal-driven trades (may already have SL/TP from strategy):
        levels = exits.compute(
            entry_price=150.0,
            side="buy",
            strategy_stop_loss=145.0,
            strategy_take_profit=160.0,
        )

        # For manual trades (no strategy-provided levels):
        levels = exits.compute(entry_price=150.0, side="buy")

        # Then submit as bracket:
        broker.bracket_order(symbol, qty, side,
                           stop_loss_price=levels.stop_loss,
                           take_profit_price=levels.take_profit)
    """

    def __init__(self, config: Optional[ProtectiveExitConfig] = None, telemetry=None):
        self._config = config or ProtectiveExitConfig()
        self._telemetry = telemetry
        self._adjustments: list[dict] = []  # Track adjustments for telemetry

    @property
    def config(self) -> ProtectiveExitConfig:
        return self._config

    @property
    def recent_adjustments(self) -> list[dict]:
        """Return recent adjustments (for dashboard/reporting)."""
        return list(self._adjustments[-100:])

    def compute(
        self,
        entry_price: float,
        side: str,
        strategy_stop_loss: Optional[float] = None,
        strategy_take_profit: Optional[float] = None,
        atr: Optional[float] = None,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
        symbol: str = "",
        trade_source: str = "signal",
    ) -> ProtectiveExitLevels:
        """
        Compute protective exit levels for a new position.

        Resolution priority for stop-loss:
            1. strategy_stop_loss (from signal/contract) — if valid
            2. ATR-based (if atr provided) — adaptive to volatility
            3. Percentage-based default — always available

        Same priority for take-profit.

        Args:
            entry_price: The fill/entry price of the position.
            side: "buy" (long) or "sell" (short).
            strategy_stop_loss: SL price from strategy/contract (optional).
            strategy_take_profit: TP price from strategy/contract (optional).
            atr: Average True Range value (optional, for volatility-adaptive).
            atr_sl_multiplier: Multiplier for ATR-based stop-loss.
            atr_tp_multiplier: Multiplier for ATR-based take-profit.
            symbol: Symbol for telemetry events (optional).
            trade_source: Origin of the trade ("signal", "manual", "telegram").

        Returns:
            ProtectiveExitLevels with validated SL/TP.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side}")

        source = "default"

        # --- Stop-Loss ---
        stop_loss = self._resolve_stop_loss(
            entry_price, side, strategy_stop_loss, atr, atr_sl_multiplier, symbol
        )
        if strategy_stop_loss and stop_loss == strategy_stop_loss:
            source = "strategy"
        elif atr and source == "default":
            # Check if ATR was used
            atr_sl = entry_price - (atr * atr_sl_multiplier) if side == "buy" else entry_price + (atr * atr_sl_multiplier)
            if abs(stop_loss - atr_sl) < 0.01:
                source = "atr"

        # --- Take-Profit ---
        take_profit = self._resolve_take_profit(
            entry_price, side, strategy_take_profit, atr, atr_tp_multiplier, stop_loss, symbol
        )
        if strategy_take_profit and take_profit == strategy_take_profit:
            source = "strategy" if source == "strategy" else source

        # Compute metrics
        sl_pct = abs(entry_price - stop_loss) / entry_price
        tp_pct = abs(take_profit - entry_price) / entry_price
        rr = tp_pct / sl_pct if sl_pct > 0 else 0.0

        levels = ProtectiveExitLevels(
            entry_price=entry_price,
            side=side,
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            stop_loss_pct=round(sl_pct, 6),
            take_profit_pct=round(tp_pct, 6),
            risk_reward_ratio=round(rr, 2),
            source=source,
        )

        logger.info(
            "protective_exits.computed",
            entry_price=entry_price,
            side=side,
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
            sl_pct=f"{levels.stop_loss_pct:.2%}",
            tp_pct=f"{levels.take_profit_pct:.2%}",
            rr=levels.risk_reward_ratio,
            source=source,
        )

        # Emit telemetry
        self._emit_configured(levels, symbol, trade_source)
        if self._telemetry:
            self._telemetry.increment("protective_exits.configured")

        return levels

    def _resolve_stop_loss(
        self,
        entry_price: float,
        side: str,
        strategy_sl: Optional[float],
        atr: Optional[float],
        atr_multiplier: float,
        symbol: str = "",
    ) -> float:
        """Resolve stop-loss price with validation and clamping."""
        cfg = self._config

        # Try strategy-provided SL first
        if strategy_sl is not None and self._is_valid_sl(entry_price, side, strategy_sl):
            sl_distance_pct = abs(entry_price - strategy_sl) / entry_price
            original_pct = sl_distance_pct
            # Clamp to configured bounds
            sl_distance_pct = max(cfg.min_stop_loss_pct, min(sl_distance_pct, cfg.max_stop_loss_pct))
            if abs(sl_distance_pct - original_pct) > 1e-8:
                reason = "clamped_max" if original_pct > cfg.max_stop_loss_pct else "clamped_min"
                original_sl = strategy_sl
                if side == "buy":
                    adjusted_sl = entry_price * (1 - sl_distance_pct)
                else:
                    adjusted_sl = entry_price * (1 + sl_distance_pct)
                self._record_adjustment(symbol, "stop_loss", original_sl, adjusted_sl, reason)
            if side == "buy":
                return entry_price * (1 - sl_distance_pct)
            else:
                return entry_price * (1 + sl_distance_pct)

        # Try ATR-based SL
        if atr is not None and atr > 0:
            atr_distance = atr * atr_multiplier
            sl_distance_pct = atr_distance / entry_price
            original_pct = sl_distance_pct
            # Clamp
            sl_distance_pct = max(cfg.min_stop_loss_pct, min(sl_distance_pct, cfg.max_stop_loss_pct))
            if abs(sl_distance_pct - original_pct) > 1e-8:
                reason = "clamped_max" if original_pct > cfg.max_stop_loss_pct else "clamped_min"
                if side == "buy":
                    original_val = entry_price * (1 - original_pct)
                    adjusted_val = entry_price * (1 - sl_distance_pct)
                else:
                    original_val = entry_price * (1 + original_pct)
                    adjusted_val = entry_price * (1 + sl_distance_pct)
                self._record_adjustment(symbol, "stop_loss", original_val, adjusted_val, reason)
            if side == "buy":
                return entry_price * (1 - sl_distance_pct)
            else:
                return entry_price * (1 + sl_distance_pct)

        # Default percentage-based
        if side == "buy":
            return entry_price * (1 - cfg.default_stop_loss_pct)
        else:
            return entry_price * (1 + cfg.default_stop_loss_pct)

    def _resolve_take_profit(
        self,
        entry_price: float,
        side: str,
        strategy_tp: Optional[float],
        atr: Optional[float],
        atr_multiplier: float,
        stop_loss: float,
        symbol: str = "",
    ) -> float:
        """Resolve take-profit price with validation and risk-reward enforcement."""
        cfg = self._config

        # Try strategy-provided TP first
        if strategy_tp is not None and self._is_valid_tp(entry_price, side, strategy_tp):
            tp_distance_pct = abs(strategy_tp - entry_price) / entry_price
            original_pct = tp_distance_pct
            # Clamp
            tp_distance_pct = max(cfg.min_take_profit_pct, min(tp_distance_pct, cfg.max_take_profit_pct))
            # Enforce minimum risk-reward
            sl_distance_pct = abs(entry_price - stop_loss) / entry_price
            if sl_distance_pct > 0 and tp_distance_pct / sl_distance_pct < cfg.min_risk_reward:
                pre_rr_pct = tp_distance_pct
                tp_distance_pct = sl_distance_pct * cfg.min_risk_reward
                if side == "buy":
                    self._record_adjustment(
                        symbol, "take_profit",
                        entry_price * (1 + pre_rr_pct),
                        entry_price * (1 + tp_distance_pct),
                        "risk_reward_enforced",
                    )
                else:
                    self._record_adjustment(
                        symbol, "take_profit",
                        entry_price * (1 - pre_rr_pct),
                        entry_price * (1 - tp_distance_pct),
                        "risk_reward_enforced",
                    )
            elif abs(tp_distance_pct - original_pct) > 1e-8:
                reason = "clamped_max" if original_pct > cfg.max_take_profit_pct else "clamped_min"
                if side == "buy":
                    self._record_adjustment(
                        symbol, "take_profit",
                        entry_price * (1 + original_pct),
                        entry_price * (1 + tp_distance_pct),
                        reason,
                    )
                else:
                    self._record_adjustment(
                        symbol, "take_profit",
                        entry_price * (1 - original_pct),
                        entry_price * (1 - tp_distance_pct),
                        reason,
                    )

            if side == "buy":
                return entry_price * (1 + tp_distance_pct)
            else:
                return entry_price * (1 - tp_distance_pct)

        # Try ATR-based TP
        if atr is not None and atr > 0:
            atr_distance = atr * atr_multiplier
            tp_distance_pct = atr_distance / entry_price
            tp_distance_pct = max(cfg.min_take_profit_pct, min(tp_distance_pct, cfg.max_take_profit_pct))
            if side == "buy":
                return entry_price * (1 + tp_distance_pct)
            else:
                return entry_price * (1 - tp_distance_pct)

        # Default percentage-based
        if side == "buy":
            return entry_price * (1 + cfg.default_take_profit_pct)
        else:
            return entry_price * (1 - cfg.default_take_profit_pct)

    def _is_valid_sl(self, entry_price: float, side: str, sl: float) -> bool:
        """Check if a proposed stop-loss is on the correct side of entry."""
        if sl <= 0 or math.isnan(sl) or math.isinf(sl):
            return False
        if side == "buy":
            return sl < entry_price  # SL must be below entry for longs
        else:
            return sl > entry_price  # SL must be above entry for shorts

    def _is_valid_tp(self, entry_price: float, side: str, tp: float) -> bool:
        """Check if a proposed take-profit is on the correct side of entry."""
        if tp <= 0 or math.isnan(tp) or math.isinf(tp):
            return False
        if side == "buy":
            return tp > entry_price  # TP must be above entry for longs
        else:
            return tp < entry_price  # TP must be below entry for shorts

    # --- Telemetry helpers ---

    def _record_adjustment(
        self, symbol: str, field_name: str, original: float, adjusted: float, reason: str
    ) -> None:
        """Record a SL/TP adjustment for telemetry and reporting."""
        adjustment = {
            "symbol": symbol,
            "field": field_name,
            "original": round(original, 4),
            "adjusted": round(adjusted, 4),
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._adjustments.append(adjustment)
        if len(self._adjustments) > 500:
            self._adjustments = self._adjustments[-500:]

        logger.info(
            "protective_exits.adjusted",
            symbol=symbol,
            field=field_name,
            original=round(original, 4),
            adjusted=round(adjusted, 4),
            reason=reason,
        )

        if self._telemetry:
            self._telemetry.increment("protective_exits.adjusted")

    def _emit_configured(
        self, levels: ProtectiveExitLevels, symbol: str, trade_source: str
    ) -> None:
        """Emit a ProtectiveExitConfigured event (if event bus available)."""
        # We emit via telemetry counter; actual event publishing is done
        # by the caller (ExecutionEngine or Telegram bot) who has the EventBus.
        logger.debug(
            "protective_exits.event.configured",
            symbol=symbol,
            trade_source=trade_source,
            source=levels.source,
        )

    def record_execution(
        self,
        symbol: str,
        exit_type: str,
        trigger_price: float,
        fill_price: float,
        qty: float,
        pnl: float,
        order_id: str = "",
        trade_id: str = "",
    ) -> None:
        """Record that a protective exit was executed (SL or TP hit).

        Called by the execution engine or broker monitor when a bracket
        leg fills.
        """
        logger.info(
            "protective_exits.executed",
            symbol=symbol,
            exit_type=exit_type,
            trigger_price=trigger_price,
            fill_price=fill_price,
            qty=qty,
            pnl=pnl,
        )
        if self._telemetry:
            self._telemetry.increment(f"protective_exits.executed.{exit_type}")
            self._telemetry.increment("protective_exits.executed.total")

    def get_telemetry_summary(self) -> dict:
        """Return a summary of protective exit telemetry for dashboards."""
        adjustments_by_reason: dict[str, int] = {}
        adjustments_by_field: dict[str, int] = {}
        for adj in self._adjustments:
            reason = adj.get("reason", "unknown")
            fld = adj.get("field", "unknown")
            adjustments_by_reason[reason] = adjustments_by_reason.get(reason, 0) + 1
            adjustments_by_field[fld] = adjustments_by_field.get(fld, 0) + 1

        return {
            "total_adjustments": len(self._adjustments),
            "adjustments_by_reason": adjustments_by_reason,
            "adjustments_by_field": adjustments_by_field,
            "recent_adjustments": self._adjustments[-10:],
        }
