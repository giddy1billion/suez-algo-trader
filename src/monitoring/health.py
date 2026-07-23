"""
System Health Monitor — monitors health across all components.

Thread-safe, lightweight (<100ms per full check).
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import psutil

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Boot time for uptime calculation
_BOOT_TIME = datetime.now(timezone.utc)

# Threshold (seconds) after which a missing heartbeat marks component degraded/down
_DEGRADED_THRESHOLD = 60
_DOWN_THRESHOLD = 180


@dataclass
class ComponentHealth:
    """Health state for a single component."""
    name: str
    status: str = "unknown"  # "healthy", "degraded", "down", "unknown"
    last_heartbeat: Optional[datetime] = None
    latency_ms: float = 0.0
    error_count: int = 0
    metadata: dict = field(default_factory=dict)


class HealthMonitor:
    """Monitors system health across all components."""

    def __init__(self):
        self._heartbeats: dict[str, datetime] = {}
        self._component_status: dict[str, str] = {}
        self._components: dict[str, ComponentHealth] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)
        # Register default components
        for name in ['broker', 'database', 'ml_model', 'event_bus',
                     'risk_engine', 'telegram', 'scheduler']:
            self._components[name] = ComponentHealth(name=name, status="unknown")

    def heartbeat(self, component: str, latency_ms: float = 0.0, metadata: dict = None):
        """Record a heartbeat from a component."""
        with self._lock:
            now = datetime.now(timezone.utc)
            self._heartbeats[component] = now
            self._component_status[component] = "ok"
            # Update ComponentHealth entry
            if component not in self._components:
                self._components[component] = ComponentHealth(name=component)
            ch = self._components[component]
            ch.last_heartbeat = now
            ch.latency_ms = latency_ms
            ch.status = "healthy"
            if metadata:
                ch.metadata.update(metadata)

    def report_error(self, component: str, error: str):
        """Record an error for a component."""
        with self._lock:
            if component not in self._components:
                self._components[component] = ComponentHealth(name=component)
            ch = self._components[component]
            ch.error_count += 1
            ch.metadata["last_error"] = error
            # Escalate status based on error count
            if ch.error_count >= 5:
                ch.status = "down"
            elif ch.error_count >= 1:
                ch.status = "degraded"
            self._component_status[component] = "error" if ch.error_count >= 5 else "warning"
        logger.warning("health.error_reported", component=component, error=error)

    def get_status(self, component: str) -> ComponentHealth:
        """Get status of a specific component."""
        with self._lock:
            if component in self._components:
                self._refresh_component_status(self._components[component])
                return self._components[component]
            return ComponentHealth(name=component, status="unknown")

    def get_full_report(self) -> dict:
        """Get comprehensive health report including system metrics."""
        with self._lock:
            # Refresh all component statuses
            for ch in self._components.values():
                self._refresh_component_status(ch)

            components = {}
            for name, ch in self._components.items():
                components[name] = {
                    "status": ch.status,
                    "last_heartbeat": ch.last_heartbeat.isoformat() if ch.last_heartbeat else None,
                    "latency_ms": ch.latency_ms,
                    "error_count": ch.error_count,
                }

        system = self.get_system_metrics()
        overall, issues = self.check_component_health()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "overall_status": overall,
            "issues": issues,
            "system": system,
            "components": components,
        }

    def get_system_metrics(self) -> dict:
        """Get OS-level metrics: memory, CPU, disk."""
        try:
            proc = psutil.Process()
            memory_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
            cpu_pct = psutil.cpu_percent(interval=None)
            disk = psutil.disk_usage("C:\\") if _is_windows() else psutil.disk_usage("/")
            return {
                "memory_mb": memory_mb,
                "cpu_percent": round(cpu_pct, 1),
                "disk_percent": round(disk.percent, 1),
                "disk_free_gb": round(disk.free / (1024 ** 3), 1),
            }
        except Exception as e:
            return {"error": str(e)}

    def check_component_health(self) -> tuple[str, list[str]]:
        """Quick health check across ComponentHealth entries.

        Returns (overall_status, list_of_issues).
        """
        issues: list[str] = []
        has_degraded = False
        has_down = False

        with self._lock:
            for ch in self._components.values():
                self._refresh_component_status(ch)
                if ch.status == "down":
                    has_down = True
                    issues.append(f"{ch.name}: DOWN (errors={ch.error_count})")
                elif ch.status == "degraded":
                    has_degraded = True
                    issues.append(f"{ch.name}: degraded (errors={ch.error_count})")

        if has_down:
            return "down", issues
        elif has_degraded:
            return "degraded", issues
        return "healthy", issues

    def _refresh_component_status(self, ch: ComponentHealth):
        """Update component status based on heartbeat freshness (must hold lock)."""
        if ch.status == "unknown" and ch.last_heartbeat is None:
            return  # Never received a heartbeat — keep unknown
        if ch.last_heartbeat is None:
            return
        elapsed = (datetime.now(timezone.utc) - ch.last_heartbeat).total_seconds()
        if elapsed > _DOWN_THRESHOLD:
            if ch.status != "down":
                ch.status = "down"
        elif elapsed > _DEGRADED_THRESHOLD:
            if ch.status != "degraded":
                ch.status = "degraded"
        else:
            # P1-13: Recover from "down"/"degraded" when heartbeat resumes,
            # but only if no errors are outstanding.
            if ch.error_count == 0 and ch.status in ("down", "degraded"):
                ch.status = "healthy"

    def set_status(self, component: str, status: str):
        """Manually set component status (ok, warning, error)."""
        with self._lock:
            self._component_status[component] = status

    @property
    def uptime_seconds(self) -> float:
        """Seconds since monitor was created."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    def check_all(self, broker=None, db=None, strategy=None, bot_manager=None, scheduler=None) -> dict:
        """Returns full health report across all components."""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "system": self.check_system(),
            "broker": self.check_broker(broker) if broker else {"status": "not_configured"},
            "database": self.check_database(db) if db else {"status": "not_configured"},
            "ml": self.check_ml(strategy) if strategy else {"status": "not_configured"},
            "telegram": self.check_telegram(bot_manager) if bot_manager else {"status": "not_configured"},
            "scheduler": self.check_scheduler(scheduler) if scheduler else {"status": "not_configured"},
        }

        # Determine overall status
        statuses = []
        for key in ("system", "broker", "database", "ml", "telegram", "scheduler"):
            s = report[key].get("status", "unknown")
            if s != "not_configured":
                statuses.append(s)

        if any(s == "error" for s in statuses):
            report["overall"] = "unhealthy"
        elif any(s == "warning" for s in statuses):
            report["overall"] = "degraded"
        else:
            report["overall"] = "healthy"

        return report

    def check_system(self) -> dict:
        """Check system resources: CPU, memory, disk, uptime."""
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/") if not _is_windows() else psutil.disk_usage("C:\\")
            uptime_secs = time.time() - psutil.boot_time()

            status = "ok"
            if cpu_pct > 90 or mem.percent > 90:
                status = "warning"
            if cpu_pct > 98 or mem.percent > 95:
                status = "error"

            return {
                "status": status,
                "cpu_percent": round(cpu_pct, 1),
                "memory_mb": round(mem.used / (1024 * 1024), 1),
                "memory_percent": round(mem.percent, 1),
                "disk_percent": round(disk.percent, 1),
                "uptime_seconds": round(uptime_secs),
            }
        except Exception as e:
            logger.warning(f"System health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_broker(self, broker) -> dict:
        """Check broker connectivity and latency."""
        if broker is None:
            return {"status": "not_configured"}

        try:
            start = time.perf_counter()
            # Try to get account info as a connectivity test
            account = None
            if hasattr(broker, "get_account"):
                account = broker.get_account()
            elif hasattr(broker, "account"):
                account = broker.account
            latency_ms = round((time.perf_counter() - start) * 1000, 1)

            connected = account is not None
            status = "ok" if connected else "error"
            if latency_ms > 2000:
                status = "warning"

            result = {
                "status": status,
                "connected": connected,
                "latency_ms": latency_ms,
                "last_response": datetime.now(timezone.utc).isoformat(),
            }

            # Check API status if available
            if hasattr(broker, "api") and hasattr(broker.api, "get_clock"):
                try:
                    clock = broker.api.get_clock()
                    result["market_open"] = clock.is_open if hasattr(clock, "is_open") else None
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.warning(f"Broker health check failed: {e}")
            return {"status": "error", "connected": False, "error": str(e)}

    def check_database(self, db) -> dict:
        """Check database connectivity and size."""
        if db is None:
            return {"status": "not_configured"}

        try:
            start = time.perf_counter()

            # Try a simple query as connectivity test
            connected = False
            size_mb = 0.0

            if hasattr(db, "conn") and db.conn:
                cursor = db.conn.execute("SELECT 1")
                cursor.fetchone()
                connected = True

                # Get DB file size
                if hasattr(db, "db_path"):
                    import os
                    if os.path.exists(db.db_path):
                        size_mb = round(os.path.getsize(db.db_path) / (1024 * 1024), 2)

            latency_ms = round((time.perf_counter() - start) * 1000, 1)

            status = "ok" if connected else "error"
            if latency_ms > 100:
                status = "warning"

            return {
                "status": status,
                "connected": connected,
                "size_mb": size_mb,
                "query_latency_ms": latency_ms,
            }
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return {"status": "error", "connected": False, "error": str(e)}

    def check_ml(self, strategy) -> dict:
        """Check ML model status."""
        if strategy is None:
            return {"status": "not_configured"}

        try:
            model_loaded = False
            model_age_hours = None
            last_prediction_time = None
            accuracy = None

            # Check if strategy has an ML model loaded
            if hasattr(strategy, "model") and strategy.model is not None:
                model_loaded = True

            if hasattr(strategy, "model_trained_at") and strategy.model_trained_at:
                age = (datetime.now(timezone.utc) - strategy.model_trained_at).total_seconds()
                model_age_hours = round(age / 3600, 1)

            if hasattr(strategy, "last_prediction_time"):
                last_prediction_time = strategy.last_prediction_time

            if hasattr(strategy, "accuracy"):
                accuracy = strategy.accuracy
            elif hasattr(strategy, "metrics") and isinstance(strategy.metrics, dict):
                accuracy = strategy.metrics.get("accuracy")

            status = "ok" if model_loaded else "warning"
            if model_age_hours and model_age_hours > 72:
                status = "warning"

            return {
                "status": status,
                "model_loaded": model_loaded,
                "model_age_hours": model_age_hours,
                "last_prediction_time": str(last_prediction_time) if last_prediction_time else None,
                "accuracy": round(accuracy, 4) if accuracy else None,
            }
        except Exception as e:
            logger.warning(f"ML health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_telegram(self, bot_manager) -> dict:
        """Check Telegram bot status."""
        if bot_manager is None:
            return {"status": "not_configured"}

        try:
            connected = False
            last_message_time = None
            polling_active = False

            if hasattr(bot_manager, "bot") and bot_manager.bot is not None:
                connected = True

            if hasattr(bot_manager, "_last_message_time"):
                last_message_time = bot_manager._last_message_time

            if hasattr(bot_manager, "_polling_active"):
                polling_active = bot_manager._polling_active
            else:
                # Assume active if bot exists
                polling_active = connected

            status = "ok" if connected and polling_active else "warning"
            if not connected:
                status = "error"

            result = {
                "status": status,
                "connected": connected,
                "polling_active": polling_active,
            }
            if last_message_time:
                result["last_message_time"] = str(last_message_time)

            return result
        except Exception as e:
            logger.warning(f"Telegram health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_scheduler(self, scheduler) -> dict:
        """Check APScheduler status."""
        if scheduler is None:
            return {"status": "not_configured"}

        try:
            running = False
            next_jobs = []
            missed_jobs = 0

            if hasattr(scheduler, "running"):
                running = scheduler.running
            elif hasattr(scheduler, "state"):
                running = scheduler.state == 1  # STATE_RUNNING

            if hasattr(scheduler, "get_jobs"):
                jobs = scheduler.get_jobs()
                for job in jobs[:10]:
                    job_info = {"name": job.name}
                    if hasattr(job, "next_run_time") and job.next_run_time:
                        job_info["next_run"] = job.next_run_time.isoformat()
                    next_jobs.append(job_info)

                # Count missed jobs (next_run_time is in the past)
                now = datetime.now(timezone.utc)
                for job in jobs:
                    if hasattr(job, "next_run_time") and job.next_run_time:
                        if job.next_run_time.replace(tzinfo=timezone.utc) < now:
                            missed_jobs += 1

            status = "ok" if running else "warning"
            if missed_jobs > 3:
                status = "warning"

            return {
                "status": status,
                "running": running,
                "next_jobs": next_jobs,
                "missed_jobs": missed_jobs,
            }
        except Exception as e:
            logger.warning(f"Scheduler health check failed: {e}")
            return {"status": "error", "error": str(e)}


def _is_windows() -> bool:
    """Check if running on Windows."""
    import platform
    return platform.system() == "Windows"
