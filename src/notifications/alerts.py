"""
Notification system — Telegram (aiogram), Discord, and console alerts for trades and errors.
Supports both simple webhook mode and full interactive Telegram bot mode.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationManager:
    """Send trade alerts and error notifications via multiple channels."""

    def __init__(
        self,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        discord_webhook: str = "",
        notify_trades: bool = True,
        notify_errors: bool = True,
    ):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook
        self.notify_trades = notify_trades
        self.notify_errors = notify_errors

    def notify_trade(self, trade: dict):
        """Send notification about an executed trade."""
        if not self.notify_trades:
            return

        symbol = trade.get('symbol', '?')
        side = trade.get('side', '?').upper()
        qty = trade.get('qty', 0)
        price = trade.get('price', 0)
        confidence = trade.get('signal_confidence', 0)

        emoji = "🟢" if side == "BUY" else "🔴"
        message = (
            f"{emoji} **{side} {symbol}**\n"
            f"Qty: {qty:.4f} @ ${price:.2f}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        if trade.get('stop_loss'):
            message += f"\nSL: ${trade['stop_loss']:.2f}"
        if trade.get('take_profit'):
            message += f"\nTP: ${trade['take_profit']:.2f}"

        self._send(message)

    def notify_exit(self, symbol: str, pnl: float, pnl_pct: float = 0):
        """Notify about a position exit."""
        if not self.notify_trades:
            return

        emoji = "💰" if pnl > 0 else "💸"
        message = (
            f"{emoji} **CLOSED {symbol}**\n"
            f"PnL: ${pnl:.2f} ({pnl_pct:.1%})\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(message)

    def notify_error(self, error: str, context: str = ""):
        """Send error notification."""
        if not self.notify_errors:
            return

        message = (
            f"⚠️ **ERROR**\n"
            f"{error}\n"
            f"Context: {context}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        self._send(message)

    def notify_daily_summary(self, summary: dict):
        """Send end-of-day performance summary."""
        message = (
            f"📊 **Daily Summary**\n"
            f"Trades: {summary.get('trades', 0)}\n"
            f"Win Rate: {summary.get('win_rate', '0%')}\n"
            f"Daily PnL: ${summary.get('daily_pnl', 0):.2f}\n"
            f"Return: {summary.get('daily_return', '0%')}\n"
            f"Halted: {'Yes ⛔' if summary.get('is_halted') else 'No ✅'}"
        )
        self._send(message)

    def notify_startup(self, mode: str, strategy: str, symbols: list[str]):
        """Notify that the bot has started."""
        message = (
            f"🤖 **Bot Started**\n"
            f"Mode: {mode.upper()}\n"
            f"Strategy: {strategy}\n"
            f"Symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._send(message)

    # ──────────────────────────────────────────────────────────────────────
    # Internal Transport
    # ──────────────────────────────────────────────────────────────────────

    def _send(self, message: str):
        """Send message to all configured channels."""
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(message)
        if self.discord_webhook:
            self._send_discord(message)
        # Always log
        logger.info("notification", message=message[:100])

    def _send_telegram(self, message: str):
        """Send via Telegram Bot API."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        # Convert markdown bold to Telegram HTML
        text = message.replace("**", "<b>").replace("**", "</b>")
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    logger.error("telegram.send_failed", status=resp.status_code)
        except Exception as e:
            logger.error("telegram.error", error=str(e))

    def _send_discord(self, message: str):
        """Send via Discord webhook."""
        payload = {"content": message}
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(self.discord_webhook, json=payload)
                if resp.status_code not in (200, 204):
                    logger.error("discord.send_failed", status=resp.status_code)
        except Exception as e:
            logger.error("discord.error", error=str(e))
