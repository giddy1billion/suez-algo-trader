"""
System Health Monitor — monitors health across all components.

Thread-safe, lightweight (<100ms per full check).
"""

import threading
import time
from datetime import datetime, timezone

import psutil

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Boot time for uptime calculation
_BOOT_TIME = datetime.now(timezone.utc)


class HealthMonitor:
    """Monitors system health across all components."""

    def __init__(self):
        self._heartbeats: dict[str, datetime] = {}
        self._component_status: dict[str, str] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)

    def heartbeat(self, component: str):
        """Record a heartbeat from a component."""
        with self._lock:
            self._heartbeats[component] = datetime.now(timezone.utc)
            self._component_status[component] = "ok"

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
