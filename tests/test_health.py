"""Tests for the HealthMonitor and OpsCommandHandler."""

import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.monitoring.health import HealthMonitor, ComponentHealth


class TestComponentHealth:
    """Test ComponentHealth dataclass."""

    def test_default_values(self):
        ch = ComponentHealth(name="test")
        assert ch.name == "test"
        assert ch.status == "unknown"
        assert ch.last_heartbeat is None
        assert ch.latency_ms == 0.0
        assert ch.error_count == 0
        assert ch.metadata == {}


class TestHealthMonitorHeartbeat:
    """Test heartbeat recording."""

    def test_heartbeat_basic(self):
        hm = HealthMonitor()
        hm.heartbeat("broker")
        status = hm.get_status("broker")
        assert status.status == "healthy"
        assert status.last_heartbeat is not None

    def test_heartbeat_with_latency(self):
        hm = HealthMonitor()
        hm.heartbeat("broker", latency_ms=42.5)
        status = hm.get_status("broker")
        assert status.latency_ms == 42.5

    def test_heartbeat_with_metadata(self):
        hm = HealthMonitor()
        hm.heartbeat("broker", metadata={"version": "2.0"})
        status = hm.get_status("broker")
        assert status.metadata["version"] == "2.0"

    def test_heartbeat_updates_timestamp(self):
        hm = HealthMonitor()
        hm.heartbeat("broker")
        first = hm.get_status("broker").last_heartbeat
        time.sleep(0.01)
        hm.heartbeat("broker")
        second = hm.get_status("broker").last_heartbeat
        assert second > first

    def test_heartbeat_unknown_component(self):
        hm = HealthMonitor()
        hm.heartbeat("custom_component", latency_ms=10.0)
        status = hm.get_status("custom_component")
        assert status.status == "healthy"
        assert status.name == "custom_component"


class TestHealthMonitorErrors:
    """Test error counting and status escalation."""

    def test_single_error_degrades(self):
        hm = HealthMonitor()
        hm.heartbeat("broker")
        hm.report_error("broker", "connection timeout")
        status = hm.get_status("broker")
        assert status.status == "degraded"
        assert status.error_count == 1

    def test_multiple_errors_mark_down(self):
        hm = HealthMonitor()
        hm.heartbeat("broker")
        for i in range(5):
            hm.report_error("broker", f"error {i}")
        status = hm.get_status("broker")
        assert status.status == "down"
        assert status.error_count == 5

    def test_error_records_last_error(self):
        hm = HealthMonitor()
        hm.report_error("database", "disk full")
        status = hm.get_status("database")
        assert status.metadata["last_error"] == "disk full"

    def test_error_on_unknown_component(self):
        hm = HealthMonitor()
        hm.report_error("new_service", "timeout")
        status = hm.get_status("new_service")
        assert status.error_count == 1
        assert status.status == "degraded"


class TestHealthMonitorStatusDetection:
    """Test degraded/down status detection."""

    def test_overall_healthy_when_all_healthy(self):
        hm = HealthMonitor()
        for comp in ['broker', 'database', 'ml_model', 'event_bus',
                     'risk_engine', 'telegram', 'scheduler']:
            hm.heartbeat(comp)
        overall, issues = hm.check_component_health()
        assert overall == "healthy"
        assert issues == []

    def test_overall_degraded_when_one_degraded(self):
        hm = HealthMonitor()
        hm.heartbeat("broker")
        hm.report_error("broker", "slow response")
        overall, issues = hm.check_component_health()
        assert overall == "degraded"
        assert len(issues) >= 1

    def test_overall_down_when_critical_down(self):
        hm = HealthMonitor()
        hm.heartbeat("broker")
        for i in range(5):
            hm.report_error("broker", f"err {i}")
        overall, issues = hm.check_component_health()
        assert overall == "down"

    def test_unknown_components_not_flagged(self):
        hm = HealthMonitor()
        # Default components never heartbeated — should be "unknown" not issues
        overall, issues = hm.check_component_health()
        assert overall == "healthy"
        assert issues == []

    def test_get_status_unknown_component(self):
        hm = HealthMonitor()
        status = hm.get_status("nonexistent")
        assert status.status == "unknown"
        assert status.name == "nonexistent"


class TestHealthMonitorSystemMetrics:
    """Test system metrics retrieval."""

    def test_system_metrics_returns_dict(self):
        hm = HealthMonitor()
        metrics = hm.get_system_metrics()
        assert isinstance(metrics, dict)
        # Should have these keys (or error key)
        if "error" not in metrics:
            assert "memory_mb" in metrics
            assert "cpu_percent" in metrics
            assert "disk_percent" in metrics
            assert "disk_free_gb" in metrics

    def test_system_metrics_values_reasonable(self):
        hm = HealthMonitor()
        metrics = hm.get_system_metrics()
        if "error" not in metrics:
            assert metrics["memory_mb"] > 0
            assert 0 <= metrics["cpu_percent"] <= 100
            assert 0 <= metrics["disk_percent"] <= 100
            assert metrics["disk_free_gb"] >= 0


class TestHealthMonitorFullReport:
    """Test full report generation."""

    def test_full_report_structure(self):
        hm = HealthMonitor()
        hm.heartbeat("broker", latency_ms=50.0)
        report = hm.get_full_report()
        assert "timestamp" in report
        assert "uptime_seconds" in report
        assert "overall_status" in report
        assert "issues" in report
        assert "system" in report
        assert "components" in report

    def test_full_report_components(self):
        hm = HealthMonitor()
        hm.heartbeat("broker", latency_ms=25.0)
        report = hm.get_full_report()
        assert "broker" in report["components"]
        assert report["components"]["broker"]["status"] == "healthy"
        assert report["components"]["broker"]["latency_ms"] == 25.0

    def test_full_report_uptime_positive(self):
        hm = HealthMonitor()
        time.sleep(0.01)
        report = hm.get_full_report()
        assert report["uptime_seconds"] > 0


class TestHealthMonitorThreadSafety:
    """Test thread safety."""

    def test_concurrent_heartbeats(self):
        hm = HealthMonitor()
        errors = []

        def send_heartbeats(comp, n):
            try:
                for i in range(n):
                    hm.heartbeat(comp, latency_ms=float(i))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=send_heartbeats, args=("broker", 100)),
            threading.Thread(target=send_heartbeats, args=("database", 100)),
            threading.Thread(target=send_heartbeats, args=("ml_model", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert hm.get_status("broker").status == "healthy"
        assert hm.get_status("database").status == "healthy"


class TestOpsCommandHandler:
    """Test OpsCommandHandler format methods."""

    def test_format_health_no_monitor(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        ops = OpsCommandHandler(health_monitor=None)
        result = ops.format_health()
        assert "not configured" in result.lower() or "⚠️" in result

    def test_format_health_with_monitor(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        hm = HealthMonitor()
        hm.heartbeat("broker", latency_ms=30.0)
        ops = OpsCommandHandler(health_monitor=hm)
        result = ops.format_health()
        assert "System Health" in result
        assert "broker" in result

    def test_format_risk_report_no_manager(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        ops = OpsCommandHandler()
        result = ops.format_risk_report()
        assert "not configured" in result.lower()

    def test_format_risk_report_with_engine(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        mock_rm = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.gross_exposure = 50000.0
        mock_metrics.net_exposure = 30000.0
        mock_metrics.portfolio_var = 1500.0
        mock_metrics.drawdown = -2.5
        mock_metrics.daily_pnl = 350.0
        mock_metrics.open_positions = 5
        mock_metrics.cash_ratio = 0.4
        mock_metrics.portfolio_heat = 0.15
        mock_rm.get_metrics.return_value = mock_metrics
        mock_rm.is_halted = False

        ops = OpsCommandHandler(risk_manager=mock_rm)
        result = ops.format_risk_report()
        assert "Risk Report" in result
        assert "50,000" in result

    def test_format_events_no_bus(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        ops = OpsCommandHandler()
        result = ops.format_events()
        assert "not configured" in result.lower()

    def test_format_performance_no_metrics(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        ops = OpsCommandHandler()
        result = ops.format_performance()
        assert "not configured" in result.lower()

    def test_format_system(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        hm = HealthMonitor()
        ops = OpsCommandHandler(health_monitor=hm)
        result = ops.format_system()
        assert "System Info" in result
        assert "Python" in result
        assert "Threads" in result

    def test_format_latency_no_broker(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        hm = HealthMonitor()
        ops = OpsCommandHandler(health_monitor=hm, broker=None)
        result = ops.format_latency()
        assert "Latency" in result
        assert "N/A" in result

    def test_format_uptime_helper(self):
        from src.monitoring.ops_commands import OpsCommandHandler
        assert OpsCommandHandler._format_uptime(90061) == "1d 1h 1m"
        assert OpsCommandHandler._format_uptime(3661) == "1h 1m 1s"
        assert OpsCommandHandler._format_uptime(65) == "1m 5s"
        assert OpsCommandHandler._format_uptime(5) == "5s"
