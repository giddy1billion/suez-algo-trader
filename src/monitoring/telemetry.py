"""
Production Telemetry — Granular metrics for operational diagnosis.

Tracks:
- Per-strategy latency (signal generation time)
- Broker latency distributions (order submission → confirmation)
- Event queue depth (events pending processing)
- Projection lag (time between event publish and projection update)
- Snapshot duration (how long snapshots take)
- Reconciliation frequency and results
- Cache hit ratios
- Memory/CPU trends over time
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""
    count: int = 0
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    last_ms: float = 0.0


class LatencyTracker:
    """Tracks latency measurements with windowed statistics."""
    
    def __init__(self, window_size: int = 1000):
        self._samples: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
    
    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._samples.append(latency_ms)
    
    def get_stats(self) -> LatencyStats:
        """Compute statistics over the sample window."""
        with self._lock:
            if not self._samples:
                return LatencyStats()
            
            import numpy as np
            arr = np.array(list(self._samples))
            return LatencyStats(
                count=len(arr),
                min_ms=float(arr.min()),
                max_ms=float(arr.max()),
                mean_ms=float(arr.mean()),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                last_ms=float(arr[-1]),
            )
    
    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


class Counter:
    """Thread-safe counter with time-windowed rate computation."""
    
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._recent: deque = deque(maxlen=100)  # timestamps of recent increments
    
    def increment(self, n: int = 1) -> None:
        with self._lock:
            self._count += n
            self._recent.append(time.time())
    
    @property
    def total(self) -> int:
        with self._lock:
            return self._count
    
    @property
    def rate_per_minute(self) -> float:
        """Compute rate over the last minute."""
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            recent_in_window = sum(1 for t in self._recent if t > cutoff)
            return recent_in_window
    
    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._recent.clear()
            self._start_time = time.time()


class Telemetry:
    """
    Central telemetry collector for the trading platform.
    
    Usage:
        telemetry = Telemetry()
        
        # Record strategy latency
        with telemetry.timer("strategy.momentum"):
            signals = strategy.generate_signals(data)
        
        # Record broker latency
        telemetry.record_latency("broker.submit_order", 45.2)
        
        # Increment counters
        telemetry.increment("events.published")
        telemetry.increment("orders.submitted")
        
        # Get full report
        report = telemetry.get_report()
    """
    
    def __init__(self):
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def record_latency(self, name: str, latency_ms: float) -> None:
        """Record a latency measurement for a named metric."""
        tracker = self._get_or_create_latency(name)
        tracker.record(latency_ms)
    
    def timer(self, name: str) -> 'TimerContext':
        """Context manager for timing operations."""
        return TimerContext(self, name)
    
    def increment(self, name: str, n: int = 1) -> None:
        """Increment a named counter."""
        counter = self._get_or_create_counter(name)
        counter.increment(n)
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value (point-in-time measurement)."""
        with self._lock:
            self._gauges[name] = value
    
    def get_latency_stats(self, name: str) -> LatencyStats:
        """Get latency statistics for a metric."""
        tracker = self._get_or_create_latency(name)
        return tracker.get_stats()
    
    def get_counter(self, name: str) -> int:
        """Get a counter's total value."""
        counter = self._get_or_create_counter(name)
        return counter.total
    
    def get_gauge(self, name: str) -> float:
        """Get a gauge value."""
        with self._lock:
            return self._gauges.get(name, 0.0)
    
    def get_report(self) -> dict:
        """Get comprehensive telemetry report."""
        with self._lock:
            latencies = {}
            for name, tracker in self._latency_trackers.items():
                stats = tracker.get_stats()
                if stats.count > 0:
                    latencies[name] = {
                        "count": stats.count,
                        "mean_ms": round(stats.mean_ms, 2),
                        "p50_ms": round(stats.p50_ms, 2),
                        "p95_ms": round(stats.p95_ms, 2),
                        "p99_ms": round(stats.p99_ms, 2),
                        "max_ms": round(stats.max_ms, 2),
                    }
            
            counters = {}
            for name, counter in self._counters.items():
                counters[name] = {
                    "total": counter.total,
                    "rate_per_min": round(counter.rate_per_minute, 1),
                }
            
            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "latencies": latencies,
                "counters": counters,
                "gauges": dict(self._gauges),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for tracker in self._latency_trackers.values():
                tracker.reset()
            for counter in self._counters.values():
                counter.reset()
            self._gauges.clear()
            self._start_time = time.time()
    
    def _get_or_create_latency(self, name: str) -> LatencyTracker:
        with self._lock:
            if name not in self._latency_trackers:
                self._latency_trackers[name] = LatencyTracker()
            return self._latency_trackers[name]
    
    def _get_or_create_counter(self, name: str) -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter()
            return self._counters[name]


class TimerContext:
    """Context manager that records elapsed time as latency."""
    
    def __init__(self, telemetry: Telemetry, name: str):
        self._telemetry = telemetry
        self._name = name
        self._start = 0.0
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed_ms = (time.time() - self._start) * 1000
        self._telemetry.record_latency(self._name, elapsed_ms)
