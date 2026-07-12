"""
Startup Preflight Check — validates critical runtime dependencies and configuration.

Provides a structured, machine-readable result object so callers can decide
how to handle failures (abort, warn, degrade gracefully). Each check is a
small, independent callable that returns a CheckResult — making it trivial
to extend with new checks (database reachability, broker ping, Redis, etc.)
without touching main.py.

Usage:
    from src.core.preflight import run_preflight, PreflightOutcome
    result = run_preflight(settings)
    if result.outcome == PreflightOutcome.FAIL:
        for check in result.failed:
            logger.error("preflight.FAILED", check=check.name, issue=check.message)
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────────
# Result Models
# ──────────────────────────────────────────────────────────────────────────


class CheckSeverity(str, Enum):
    """How critical a failed check is."""
    CRITICAL = "critical"  # Abort startup
    WARNING = "warning"    # Log and continue


class PreflightOutcome(str, Enum):
    """Overall preflight result."""
    PASS = "pass"
    WARN = "warn"  # Warnings present but no critical failures
    FAIL = "fail"  # At least one critical failure


@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    passed: bool
    severity: CheckSeverity
    message: str = ""

    @property
    def failed(self) -> bool:
        return not self.passed


@dataclass
class PreflightReport:
    """Aggregated results of all preflight checks."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def outcome(self) -> PreflightOutcome:
        if any(c.failed and c.severity == CheckSeverity.CRITICAL for c in self.checks):
            return PreflightOutcome.FAIL
        if any(c.failed and c.severity == CheckSeverity.WARNING for c in self.checks):
            return PreflightOutcome.WARN
        return PreflightOutcome.PASS

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.failed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.failed and c.severity == CheckSeverity.WARNING]

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.failed and c.severity == CheckSeverity.CRITICAL]

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)


# ──────────────────────────────────────────────────────────────────────────
# Individual Checks
# ──────────────────────────────────────────────────────────────────────────


def check_timezone_data() -> CheckResult:
    """Verify IANA timezone database is available (critical for market-time logic)."""
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo("US/Eastern")
        ZoneInfo("America/New_York")
        return CheckResult(
            name="timezone_data",
            passed=True,
            severity=CheckSeverity.CRITICAL,
        )
    except Exception as e:
        return CheckResult(
            name="timezone_data",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message=(
                f"Timezone data unavailable ({type(e).__name__}: {e}). "
                "Install 'tzdata' package or use a non-slim base image."
            ),
        )


def check_alpaca_credentials(settings) -> CheckResult:
    """Verify Alpaca API credentials are set and non-placeholder."""
    from config.settings import TradingMode

    is_live = settings.trading_mode == TradingMode.LIVE
    api_key = settings.alpaca_live_api_key if is_live else settings.alpaca_paper_api_key
    secret = settings.alpaca_live_secret_key if is_live else settings.alpaca_paper_secret_key
    mode_label = "LIVE" if is_live else "PAPER"

    if not api_key or not secret or api_key.startswith("your_"):
        return CheckResult(
            name="alpaca_credentials",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message=f"Alpaca {mode_label} API credentials are missing or placeholder.",
        )

    return CheckResult(
        name="alpaca_credentials",
        passed=True,
        severity=CheckSeverity.CRITICAL,
    )


def check_telegram_config(settings) -> CheckResult:
    """Verify Telegram bot credentials (warning-only — bot can run without it)."""
    if not settings.telegram_bot_token:
        return CheckResult(
            name="telegram_config",
            passed=False,
            severity=CheckSeverity.WARNING,
            message="Telegram bot disabled — no TELEGRAM_BOT_TOKEN set.",
        )
    if not settings.telegram_chat_id:
        return CheckResult(
            name="telegram_config",
            passed=False,
            severity=CheckSeverity.WARNING,
            message="Telegram notifications disabled — no TELEGRAM_CHAT_ID set.",
        )
    return CheckResult(
        name="telegram_config",
        passed=True,
        severity=CheckSeverity.WARNING,
    )


def check_critical_imports() -> CheckResult:
    """Verify essential libraries are importable."""
    missing = []
    for module_name in ("pandas", "numpy", "httpx", "sqlalchemy"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        return CheckResult(
            name="critical_imports",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message=f"Missing critical dependencies: {', '.join(missing)}",
        )

    return CheckResult(
        name="critical_imports",
        passed=True,
        severity=CheckSeverity.CRITICAL,
    )


def check_database_writable(settings) -> CheckResult:
    """Verify database path is writable (for SQLite) or reachable."""
    import os

    db_url = settings.database_url
    if db_url.startswith("sqlite"):
        # Extract path from sqlite:///path
        db_path = db_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path) or "."
        if not os.path.isdir(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except OSError as e:
                return CheckResult(
                    name="database_writable",
                    passed=False,
                    severity=CheckSeverity.CRITICAL,
                    message=f"Database directory not writable: {db_dir} ({e})",
                )
        # Verify write permission
        test_file = os.path.join(db_dir, ".preflight_write_test")
        try:
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except OSError as e:
            return CheckResult(
                name="database_writable",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Cannot write to database directory: {db_dir} ({e})",
            )

    return CheckResult(
        name="database_writable",
        passed=True,
        severity=CheckSeverity.CRITICAL,
    )


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────

# Default check registry — order matters for early exit on critical failures
_DEFAULT_CHECKS: list[Callable] = [
    check_timezone_data,
    check_critical_imports,
    # Settings-dependent checks are added dynamically in run_preflight
]


def run_preflight(settings) -> PreflightReport:
    """Execute all preflight checks and return structured report.

    Args:
        settings: Application settings object (config.settings.Settings).

    Returns:
        PreflightReport with all check results.
    """
    report = PreflightReport()

    # Static checks (no settings dependency)
    report.checks.append(check_timezone_data())
    report.checks.append(check_critical_imports())

    # Settings-dependent checks
    report.checks.append(check_alpaca_credentials(settings))
    report.checks.append(check_telegram_config(settings))
    report.checks.append(check_database_writable(settings))

    return report
