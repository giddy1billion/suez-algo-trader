"""
Enhanced Telegram Signal Formatter.

Extends the existing signal formatting with actionable command lines.
Given a SignalPackage, appends ready-to-paste commands to the message:
  /buy SYMBOL QTY   (for long signals)
  /sell SYMBOL QTY  (for short signals)

Handles:
- Explicit position_size or auto-sized via risk-sizing
- Fallback source / model-disabled warnings
- Risk-rejection suppression
- Bracket-order vs informational TP/SL display
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalPackage:
    """
    Input package for the enhanced signal formatter.

    Contains all fields needed to produce a complete Telegram signal message
    with actionable command lines.
    """

    # Required fields
    symbol: str = ""
    direction: str = ""       # "BUY" or "SELL" (long/short)
    strength: float = 0.0     # Signal strength / confidence (0-1)
    strategy: str = ""
    source: str = ""          # e.g., "ml_predictor", "fallback", etc.

    # Optional execution parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None

    # Risk/model context
    model_active: bool = True
    risk_approved: bool = True
    risk_rejection_reason: str = ""
    is_fallback_source: bool = False

    # Bracket order support
    bracket_orders_supported: bool = False

    # Auto-sizing context (used when position_size is None)
    auto_sized_qty: Optional[float] = None


def format_signal_message(pkg: SignalPackage) -> str:
    """
    Format a complete Telegram signal message with actionable commands.

    Preserves the existing signal message style and appends command lines.
    Output is plain text, limited to ~8 lines.

    Args:
        pkg: A SignalPackage containing all signal context.

    Returns:
        Formatted plain-text message for Telegram.
    """
    side = pkg.direction.upper() if pkg.direction else ""
    emoji = "📶" if side == "BUY" else "📉" if side == "SELL" else "⏸️"

    # --- Base message (preserves existing style) ---
    lines = [
        f"{emoji} Signal: {side} {pkg.symbol}",
        f"Strength: {pkg.strength:.2f}",
    ]

    if pkg.strategy:
        lines.append(f"Strategy: {pkg.strategy}")

    lines.append(f"Source: {pkg.source}")

    # --- Warning conditions: suppress all commands ---
    warning = _get_warning(pkg)
    if warning:
        lines.insert(0, f"⚠️ {warning}")
        return "\n".join(lines)

    # --- Actionable command ---
    qty, qty_label = _resolve_quantity(pkg)
    if qty is not None and side in ("BUY", "SELL"):
        cmd = "/buy" if side == "BUY" else "/sell"
        qty_str = _format_qty(qty)
        cmd_line = f"{cmd} {pkg.symbol} {qty_str}"
        if qty_label:
            cmd_line += f" {qty_label}"
        lines.append(cmd_line)
    elif side in ("BUY", "SELL"):
        # Cannot determine quantity — suppress command with explanation
        lines.append("⚠️ Command suppressed: position size could not be determined")
        return "\n".join(lines)

    # --- TP/SL information ---
    tp_sl_line = _format_tp_sl(pkg)
    if tp_sl_line:
        lines.append(tp_sl_line)

    return "\n".join(lines)


def _get_warning(pkg: SignalPackage) -> Optional[str]:
    """Check for conditions that require a warning and command suppression."""
    if pkg.is_fallback_source or pkg.source.lower() == "fallback":
        return "FALLBACK SOURCE — no actionable commands (model unavailable)"

    if not pkg.model_active:
        return "NO ACTIVE MODEL — no actionable commands"

    if not pkg.risk_approved:
        reason = pkg.risk_rejection_reason or "risk-control check failed"
        return f"RISK REJECTED — {reason}"

    return None


def _resolve_quantity(pkg: SignalPackage) -> tuple[Optional[float], str]:
    """
    Resolve position quantity.

    Returns:
        (quantity, label) where label is "(auto-sized)" or "" for explicit.
        Returns (None, "") if sizing cannot be determined.
    """
    if pkg.position_size is not None and pkg.position_size > 0:
        return (pkg.position_size, "")

    if pkg.auto_sized_qty is not None and pkg.auto_sized_qty > 0:
        return (pkg.auto_sized_qty, "(auto-sized)")

    # Cannot determine quantity
    return (None, "")


def _format_qty(qty: float) -> str:
    """Format quantity, removing trailing zeros for clean display."""
    if qty == int(qty):
        return str(int(qty))
    return f"{qty:.4f}".rstrip("0").rstrip(".")


def _format_tp_sl(pkg: SignalPackage) -> Optional[str]:
    """Format TP/SL information based on bracket order support."""
    has_sl = pkg.stop_loss is not None
    has_tp = pkg.take_profit is not None

    if not has_sl and not has_tp:
        return None

    if has_sl and has_tp:
        if pkg.bracket_orders_supported:
            return "📋 Native bracket order will be submitted (TP/SL attached)"
        else:
            # Informational TP/SL close commands
            side = pkg.direction.upper()
            close_side = "/sell" if side == "BUY" else "/buy"
            return (
                f"ℹ️ TP: {close_side} {pkg.symbol} @ {pkg.take_profit:.2f} | "
                f"SL: {close_side} {pkg.symbol} @ {pkg.stop_loss:.2f}"
            )

    # Only one of TP/SL present — just display informational
    parts = []
    if has_sl:
        parts.append(f"SL: ${pkg.stop_loss:.2f}")
    if has_tp:
        parts.append(f"TP: ${pkg.take_profit:.2f}")
    return "ℹ️ " + " | ".join(parts)
