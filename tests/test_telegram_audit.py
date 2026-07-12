"""Tests for Telegram audit forwarding system."""

import logging
import time
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

from src.core.events import (
    DecisionContractCreated,
    EventBus,
    OrderAccepted,
    OrderFilled,
    OrderPartialFill,
    OrderRejected,
    OrderSubmitted,
    RiskEvaluated,
    RiskHalt,
    SchedulerEvent,
    SignalGenerated,
    SystemHealth,
    TradeClosed,
    TradeOpened,
)
from src.notifications.telegram_audit_forwarder import (
    TelegramAuditForwarder,
    TelegramLogHandler,
    setup_telegram_full_audit,
    _format_signal,
    _format_risk_evaluated,
    _format_order_submitted,
    _format_order_filled,
    _format_order_rejected,
    _format_trade_opened,
    _format_trade_closed,
    _format_risk_halt,
    _format_system_health,
    _format_scheduler_event,
    _format_order_partial_fill,
    _format_generic,
)


class TestFormatters:
    """Test individual event formatters produce valid output."""

    def test_format_signal_buy(self):
        event = SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.85)
        result = _format_signal(event)
        assert "AAPL" in result
        assert "BUY" in result
        assert "📶" in result

    def test_format_signal_sell(self):
        event = SignalGenerated(symbol="TSLA", signal="SELL", confidence=0.7)
        result = _format_signal(event)
        assert "📉" in result
        assert "SELL" in result

    def test_format_risk_evaluated_rejected(self):
        event = RiskEvaluated(symbol="AAPL", approved=False, reasons=["max exposure"], risk_score=0.9)
        result = _format_risk_evaluated(event)
        assert "REJECTED" in result
        assert "max exposure" in result
        assert "🛡️" in result

    def test_format_risk_evaluated_approved(self):
        event = RiskEvaluated(symbol="MSFT", approved=True, reasons=[], risk_score=0.2)
        result = _format_risk_evaluated(event)
        assert "APPROVED" in result
        assert "✅" in result

    def test_format_order_submitted(self):
        event = OrderSubmitted(symbol="AAPL", side="BUY", qty=100, order_id="ORD123")
        result = _format_order_submitted(event)
        assert "ORD123" in result
        assert "AAPL" in result
        assert "BUY" in result

    def test_format_order_filled(self):
        event = OrderFilled(order_id="ORD123", fill_price=150.25)
        result = _format_order_filled(event)
        assert "ORD123" in result
        assert "150.25" in result
        assert "💰" in result

    def test_format_order_partial_fill(self):
        event = OrderPartialFill(order_id="ORD456", filled_qty=50, fill_price=99.50)
        result = _format_order_partial_fill(event)
        assert "ORD456" in result
        assert "50" in result
        assert "99.50" in result
        assert "⏳" in result

    def test_format_order_rejected(self):
        event = OrderRejected(order_id="ORD789", reason="Insufficient funds")
        result = _format_order_rejected(event)
        assert "ORD789" in result
        assert "Insufficient funds" in result
        assert "⛔" in result

    def test_format_trade_opened(self):
        event = TradeOpened(symbol="NVDA", side="BUY", entry_price=450.0, qty=10, stop_loss=440, take_profit=470)
        result = _format_trade_opened(event)
        assert "NVDA" in result
        assert "BUY" in result
        assert "450.00" in result

    def test_format_trade_closed_profit(self):
        event = TradeClosed(symbol="AAPL", pnl=250.0, pnl_pct=5.0, reason="take_profit")
        result = _format_trade_closed(event)
        assert "💰" in result
        assert "250.00" in result
        assert "take_profit" in result

    def test_format_trade_closed_loss(self):
        event = TradeClosed(symbol="TSLA", pnl=-100.0, pnl_pct=-2.5, reason="stop_loss")
        result = _format_trade_closed(event)
        assert "💸" in result
        assert "-100.00" in result

    def test_format_risk_halt(self):
        event = RiskHalt(reason="Max daily loss exceeded", level="CRITICAL")
        result = _format_risk_halt(event)
        assert "🚨" in result
        assert "CRITICAL" in result
        assert "Max daily loss exceeded" in result

    def test_format_scheduler_event(self):
        event = SchedulerEvent(job_name="backtest", status="completed")
        result = _format_scheduler_event(event)
        assert "backtest" in result
        assert "completed" in result
        assert "✅" in result

    def test_format_system_health_degraded(self):
        event = SystemHealth(component="broker", status="degraded", metrics={"latency_ms": 2500})
        result = _format_system_health(event)
        assert "🟡" in result
        assert "broker" in result
        assert "degraded" in result
        assert "2500" in result

    def test_format_system_health_down(self):
        event = SystemHealth(component="database", status="down", metrics={})
        result = _format_system_health(event)
        assert "🔴" in result
        assert "database" in result

    def test_format_generic_event(self):
        from src.core.events import Event
        event = Event(source="test_module")
        result = _format_generic(event)
        assert "Event" in result


class TestTelegramAuditForwarder:
    """Test the main audit forwarder."""

    def test_handles_all_event_types(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(send_fn)

        events = [
            SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.9),
            RiskEvaluated(symbol="AAPL", approved=True),
            OrderSubmitted(order_id="O1", symbol="AAPL", side="BUY", qty=10),
            OrderAccepted(order_id="O1"),
            OrderPartialFill(order_id="O1", filled_qty=5, fill_price=150.0),
            OrderFilled(order_id="O1", fill_price=150.0),
            TradeOpened(symbol="AAPL", side="BUY", entry_price=150.0, qty=10),
            TradeClosed(symbol="AAPL", pnl=50.0, pnl_pct=3.3, reason="take_profit"),
            RiskHalt(reason="test", level="WARNING"),
            OrderRejected(order_id="O2", reason="insufficient"),
            SchedulerEvent(job_name="train", status="completed"),
            SystemHealth(component="broker", status="degraded"),
        ]

        for event in events:
            forwarder.handle(event)

        # Wait for sender thread to drain
        time.sleep(1.0)
        forwarder.stop()

        assert send_fn.call_count == len(events) - 1  # SignalGenerated waits for risk verdict

    def test_queue_overflow_does_not_crash(self):
        slow_send = MagicMock(side_effect=lambda x: time.sleep(0.5))
        forwarder = TelegramAuditForwarder(slow_send)

        # Flood the queue
        for i in range(6000):
            forwarder.handle(RiskHalt(reason=f"halt-{i}", level="WARNING"))

        # Should not raise
        forwarder.stop()
        assert forwarder._events_dropped > 0  # Some were dropped due to overflow

    def test_register_on_event_bus(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(send_fn)
        bus = EventBus()
        forwarder.register(bus)

        bus.publish(RiskHalt(reason="x", level="WARNING"))
        time.sleep(0.5)
        forwarder.stop()

        assert send_fn.call_count == 1

    def test_stats(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(send_fn)

        forwarder.handle(RiskHalt(reason="test", level="WARNING"))
        time.sleep(0.5)
        forwarder.stop()

        stats = forwarder.stats
        assert stats["events_sent"] == 1
        assert stats["events_dropped"] == 0

    def test_send_error_does_not_crash(self):
        failing_send = MagicMock(side_effect=Exception("Network error"))
        forwarder = TelegramAuditForwarder(failing_send)

        forwarder.handle(RiskHalt(reason="test", level="WARNING"))
        time.sleep(1.5)
        forwarder.stop()

        # Should not crash; event retried MAX_SEND_RETRIES times (Finding 3)
        assert failing_send.call_count == TelegramAuditForwarder.MAX_SEND_RETRIES


class TestTelegramLogHandler:
    """Test the Python logging handler that sends to Telegram."""

    def test_captures_warning(self):
        send_fn = MagicMock()
        handler = TelegramLogHandler(send_fn)

        test_logger = logging.getLogger("test.telegram_handler")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.warning("Something went wrong")
        time.sleep(0.5)
        handler.stop()

        assert send_fn.call_count == 1
        msg = send_fn.call_args[0][0]
        assert "WARNING" in msg
        assert "Something went wrong" in msg

        test_logger.removeHandler(handler)

    def test_captures_error(self):
        send_fn = MagicMock()
        handler = TelegramLogHandler(send_fn)

        test_logger = logging.getLogger("test.telegram_error")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.error("Critical failure")
        time.sleep(0.5)
        handler.stop()

        assert send_fn.call_count == 1
        msg = send_fn.call_args[0][0]
        assert "ERROR" in msg
        assert "🔴" in msg

        test_logger.removeHandler(handler)

    def test_ignores_info(self):
        send_fn = MagicMock()
        handler = TelegramLogHandler(send_fn)

        test_logger = logging.getLogger("test.telegram_info")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.info("Normal operation")
        time.sleep(0.5)
        handler.stop()

        assert send_fn.call_count == 0
        test_logger.removeHandler(handler)

    def test_truncates_long_messages(self):
        send_fn = MagicMock()
        handler = TelegramLogHandler(send_fn)

        test_logger = logging.getLogger("test.telegram_long")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)

        test_logger.warning("X" * 1000)
        time.sleep(0.5)
        handler.stop()

        msg = send_fn.call_args[0][0]
        assert "..." in msg
        assert len(msg) < 1000  # Truncated

        test_logger.removeHandler(handler)


class TestSetupHelper:
    """Test the setup_telegram_full_audit helper."""

    def test_setup_returns_components(self):
        send_fn = MagicMock()
        bus = EventBus()

        result = setup_telegram_full_audit(bus, send_fn, attach_log_handler=True)

        assert "forwarder" in result
        assert "log_handler" in result
        assert isinstance(result["forwarder"], TelegramAuditForwarder)
        assert isinstance(result["log_handler"], TelegramLogHandler)

        # Cleanup
        result["forwarder"].stop()
        result["log_handler"].stop()
        logging.getLogger().removeHandler(result["log_handler"])

    def test_setup_without_log_handler(self):
        send_fn = MagicMock()
        bus = EventBus()

        result = setup_telegram_full_audit(bus, send_fn, attach_log_handler=False)

        assert "forwarder" in result
        assert "log_handler" not in result

        result["forwarder"].stop()

    def test_events_flow_through_after_setup(self):
        send_fn = MagicMock()
        bus = EventBus()

        result = setup_telegram_full_audit(bus, send_fn, attach_log_handler=False)

        # Publish events
        bus.publish(RiskHalt(reason="test halt", level="CRITICAL"))
        bus.publish(SignalGenerated(symbol="BTC/USD", signal="BUY"))

        time.sleep(1.0)
        result["forwarder"].stop()

        assert send_fn.call_count == 1


class TestCanonicalTradeIntentPipeline:
    def test_approved_signal_emits_single_actionable_message_with_dedup(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(
            send_fn,
            bracket_orders_supported_provider=lambda: True,
        )
        forwarder._get_active_model_version = lambda: "v1.2.3"

        signal = SignalGenerated(
            signal_id="sig-1",
            symbol="AAPL",
            signal="BUY",
            side="BUY",
            strategy="ml_momentum",
            strategy_version="v1.2.3",
            signal_strength=0.83,
            features={"strategy_proposed_stop_loss": 145.0, "strategy_proposed_take_profit": 165.0},
            source="engine",
        )
        contract = DecisionContractCreated(
            contract_id="contract-1",
            signal_id="sig-1",
            decision="execute",
            symbol="AAPL",
            side="BUY",
            recommended_stop_loss=144.9,
            recommended_take_profit=165.1,
        )
        risk = RiskEvaluated(
            symbol="AAPL",
            signal_id="sig-1",
            contract_id="contract-1",
            approved=True,
            adjusted_qty=10.7,
        )

        forwarder.handle(signal)
        forwarder.handle(contract)
        forwarder.handle(risk)
        forwarder.handle(risk)  # duplicate delivery

        time.sleep(0.8)
        forwarder.stop()

        assert send_fn.call_count == 2  # contract event + one final approved intent
        final_message = send_fn.call_args_list[-1][0][0]
        assert "/buy AAPL 11" in final_message
        assert "Native bracket order will be submitted" in final_message
        assert _format_signal(signal) in final_message

    def test_no_verdict_timeout_emits_explicit_followup(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(
            send_fn,
            risk_verdict_timeout_seconds=1.0,
            timeout_check_interval=0.5,
        )
        forwarder._get_active_model_version = lambda: "v1"

        forwarder.handle(
            SignalGenerated(
                signal_id="sig-timeout",
                symbol="MSFT",
                signal="BUY",
                side="BUY",
                strategy="s",
                strategy_version="v1",
            )
        )
        time.sleep(2.0)  # Wait for background timer to fire after 1s timeout
        forwarder.stop()

        combined = "\n".join(call[0][0] for call in send_fn.call_args_list)
        assert "NO VERDICT RECEIVED" in combined
        assert "no action taken" in combined

    def test_risk_rejection_suppresses_actionable_command(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(send_fn)
        forwarder._get_active_model_version = lambda: "vA"

        signal = SignalGenerated(
            signal_id="sig-rej",
            symbol="TSLA",
            signal="SELL",
            side="SELL",
            strategy="s",
            strategy_version="vA",
            signal_strength=0.7,
        )
        risk = RiskEvaluated(
            symbol="TSLA",
            signal_id="sig-rej",
            approved=False,
            reasons=["max exposure"],
            adjusted_qty=5.0,
        )

        forwarder.handle(signal)
        forwarder.handle(risk)
        time.sleep(0.6)
        forwarder.stop()

        msg = send_fn.call_args_list[-1][0][0]
        assert "RISK REJECTED" in msg
        assert "/buy" not in msg
        assert "/sell" not in msg

    def test_fallback_provenance_suppresses_command(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(send_fn)
        forwarder._get_active_model_version = lambda: "vA"

        forwarder.handle(
            SignalGenerated(
                signal_id="sig-fallback",
                symbol="AAPL",
                signal="BUY",
                side="BUY",
                strategy="s",
                strategy_version="vA",
                tags=("fallback",),
            )
        )
        forwarder.handle(
            RiskEvaluated(
                symbol="AAPL",
                signal_id="sig-fallback",
                approved=True,
                adjusted_qty=3.0,
            )
        )
        time.sleep(0.6)
        forwarder.stop()

        msg = send_fn.call_args_list[-1][0][0]
        assert "FALLBACK SOURCE" in msg
        assert "/buy" not in msg

    def test_malformed_sizing_input_suppresses_command(self):
        send_fn = MagicMock()
        forwarder = TelegramAuditForwarder(send_fn)
        forwarder._get_active_model_version = lambda: "vA"

        forwarder.handle(
            SignalGenerated(
                signal_id="sig-bad-size",
                symbol="AAPL",
                signal="BUY",
                side="BUY",
                strategy="s",
                strategy_version="vA",
            )
        )
        forwarder.handle(
            RiskEvaluated(
                symbol="AAPL",
                signal_id="sig-bad-size",
                approved=True,
                adjusted_qty=float("inf"),
            )
        )
        time.sleep(0.6)
        forwarder.stop()

        msg = send_fn.call_args_list[-1][0][0]
        assert "position size could not be determined" in msg.lower()
        assert "/buy" not in msg


class TestNotificationSubscriberExpanded:
    """Test that NotificationSubscriber now handles ALL event types."""

    def test_registers_all_handlers(self):
        from src.core.subscribers import NotificationSubscriber

        send_fn = MagicMock()
        sub = NotificationSubscriber(send_fn)
        bus = EventBus()
        sub.register(bus)

        # Publish each event type
        bus.publish(SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.8))
        bus.publish(OrderSubmitted(order_id="O1", symbol="AAPL", side="BUY", qty=10))
        bus.publish(OrderFilled(order_id="O1", fill_price=150.0))
        bus.publish(OrderPartialFill(order_id="O1", filled_qty=5, fill_price=149.0))
        bus.publish(TradeOpened(symbol="AAPL", side="BUY", entry_price=150.0, qty=10))
        bus.publish(TradeClosed(symbol="AAPL", pnl=50.0, pnl_pct=3.3, reason="tp"))
        bus.publish(RiskHalt(reason="test", level="WARNING"))
        bus.publish(OrderRejected(order_id="O2", reason="no funds"))
        bus.publish(SchedulerEvent(job_name="train", status="failed"))
        bus.publish(SystemHealth(component="broker", status="down", metrics={"latency_ms": 5000}))

        # RiskEvaluated only notifies on rejection
        bus.publish(RiskEvaluated(symbol="X", approved=False, reasons=["too risky"]))
        bus.publish(RiskEvaluated(symbol="Y", approved=True, reasons=[]))  # Should NOT trigger

        # 11 notifications total (RiskEvaluated approved=True is silent)
        assert send_fn.call_count == 11
