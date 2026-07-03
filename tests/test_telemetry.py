"""Tests for telemetry module and event schema versioning."""

import threading
import time

import pytest

from src.monitoring.telemetry import (
    Counter,
    LatencyStats,
    LatencyTracker,
    Telemetry,
    TimerContext,
)
from src.core.events import Event, SignalGenerated, SCHEMA_VERSION


class TestLatencyTracker:
    def test_empty_stats(self):
        tracker = LatencyTracker()
        stats = tracker.get_stats()
        assert stats.count == 0
        assert stats.min_ms == 0.0

    def test_record_and_stats(self):
        tracker = LatencyTracker()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            tracker.record(v)
        stats = tracker.get_stats()
        assert stats.count == 5
        assert stats.min_ms == 10.0
        assert stats.max_ms == 50.0
        assert stats.mean_ms == 30.0
        assert stats.last_ms == 50.0
        assert stats.p50_ms == 30.0

    def test_reset(self):
        tracker = LatencyTracker()
        tracker.record(5.0)
        tracker.reset()
        assert tracker.get_stats().count == 0

    def test_window_size(self):
        tracker = LatencyTracker(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record(v)
        stats = tracker.get_stats()
        assert stats.count == 3
        assert stats.min_ms == 3.0


class TestCounter:
    def test_increment_and_total(self):
        c = Counter()
        c.increment()
        c.increment(5)
        assert c.total == 6

    def test_rate_per_minute(self):
        c = Counter()
        c.increment(3)
        # All recent increments happened just now, so rate should be > 0
        assert c.rate_per_minute > 0

    def test_reset(self):
        c = Counter()
        c.increment(10)
        c.reset()
        assert c.total == 0


class TestTelemetry:
    def test_record_latency(self):
        t = Telemetry()
        t.record_latency("broker.order", 12.5)
        t.record_latency("broker.order", 15.0)
        stats = t.get_latency_stats("broker.order")
        assert stats.count == 2
        assert stats.min_ms == 12.5

    def test_increment_counter(self):
        t = Telemetry()
        t.increment("events.published")
        t.increment("events.published", 4)
        assert t.get_counter("events.published") == 5

    def test_set_and_get_gauge(self):
        t = Telemetry()
        t.set_gauge("queue.depth", 42.0)
        assert t.get_gauge("queue.depth") == 42.0
        assert t.get_gauge("nonexistent") == 0.0

    def test_timer_context(self):
        t = Telemetry()
        with t.timer("strategy.momentum"):
            time.sleep(0.01)
        stats = t.get_latency_stats("strategy.momentum")
        assert stats.count == 1
        assert stats.last_ms >= 5.0  # at least 5ms

    def test_get_report(self):
        t = Telemetry()
        t.record_latency("broker.order", 20.0)
        t.increment("orders.submitted")
        t.set_gauge("memory.mb", 256.0)
        report = t.get_report()
        assert "uptime_seconds" in report
        assert "latencies" in report
        assert "counters" in report
        assert "gauges" in report
        assert "timestamp" in report
        assert "broker.order" in report["latencies"]
        assert report["counters"]["orders.submitted"]["total"] == 1
        assert report["gauges"]["memory.mb"] == 256.0

    def test_reset(self):
        t = Telemetry()
        t.record_latency("x", 1.0)
        t.increment("y")
        t.set_gauge("z", 5.0)
        t.reset()
        assert t.get_latency_stats("x").count == 0
        assert t.get_counter("y") == 0
        assert t.get_gauge("z") == 0.0


class TestThreadSafety:
    def test_concurrent_recording(self):
        t = Telemetry()
        errors = []

        def worker(tid):
            try:
                for i in range(100):
                    t.record_latency("concurrent", float(i))
                    t.increment("ops")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors
        assert t.get_counter("ops") == 1000


class TestEventVersioning:
    def test_to_dict_includes_schema_version(self):
        event = Event(source="test")
        data = event.to_dict()
        assert "_schema_version" in data
        assert data["_schema_version"] == SCHEMA_VERSION
        assert data["_type"] == "Event"

    def test_from_dict_handles_versioned_event(self):
        event = Event(source="test")
        data = event.to_dict()
        restored = Event.from_dict(data)
        assert restored.source == "test"
        assert restored.event_id == event.event_id

    def test_signal_generated_versioning(self):
        sig = SignalGenerated(symbol="BTCUSDT", signal="BUY", confidence=0.9)
        data = sig.to_dict()
        assert data["_schema_version"] == SCHEMA_VERSION
        assert data["_type"] == "SignalGenerated"
        restored = SignalGenerated.from_dict(data)
        assert restored.symbol == "BTCUSDT"
        assert restored.confidence == 0.9

    def test_from_dict_ignores_unknown_version(self):
        data = {
            "_type": "Event",
            "_schema_version": "99.0.0",
            "source": "old",
            "event_id": "abc123",
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
        event = Event.from_dict(data)
        assert event.source == "old"
        assert event.event_id == "abc123"
