"""Tests for event bus pub/sub system."""
import pytest
from src.core.events import EventBus, Event, SignalGenerated, OrderFilled


class TestEventBus:
    def test_publish_notifies_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe(SignalGenerated, lambda e: received.append(e))
        
        event = SignalGenerated(symbol="AAPL", signal="BUY", confidence=0.8, strategy="momentum")
        bus.publish(event)
        assert len(received) == 1
        assert received[0].symbol == "AAPL"

    def test_multiple_subscribers(self):
        bus = EventBus()
        r1, r2 = [], []
        bus.subscribe(SignalGenerated, lambda e: r1.append(e))
        bus.subscribe(SignalGenerated, lambda e: r2.append(e))
        
        bus.publish(SignalGenerated(symbol="MSFT", signal="SELL", confidence=0.7, strategy="test"))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_subscriber_error_doesnt_crash(self):
        bus = EventBus()
        good_received = []
        
        def bad_handler(e):
            raise RuntimeError("handler failed")
        
        bus.subscribe(SignalGenerated, bad_handler)
        bus.subscribe(SignalGenerated, lambda e: good_received.append(e))
        
        # Should not raise
        bus.publish(SignalGenerated(symbol="X", signal="BUY", confidence=0.5, strategy="t"))
        assert len(good_received) == 1  # Good handler still called

    def test_different_event_types_isolated(self):
        bus = EventBus()
        signal_received = []
        fill_received = []
        
        bus.subscribe(SignalGenerated, lambda e: signal_received.append(e))
        bus.subscribe(OrderFilled, lambda e: fill_received.append(e))
        
        bus.publish(SignalGenerated(symbol="A", signal="BUY", confidence=0.9, strategy="t"))
        assert len(signal_received) == 1
        assert len(fill_received) == 0
