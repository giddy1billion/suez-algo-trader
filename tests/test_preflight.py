"""Unit tests for src.core.preflight — startup health checks."""

import os
import sys
import tempfile
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from src.core.preflight import (
    CheckResult,
    CheckSeverity,
    PreflightOutcome,
    PreflightReport,
    check_alpaca_credentials,
    check_critical_imports,
    check_database_writable,
    check_telegram_config,
    check_timezone_data,
    run_preflight,
)


# ──────────────────────────────────────────────────────────────────────────
# CheckResult / PreflightReport model tests
# ──────────────────────────────────────────────────────────────────────────


class TestPreflightReport:
    def test_all_pass(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=True, severity=CheckSeverity.CRITICAL),
            CheckResult(name="b", passed=True, severity=CheckSeverity.WARNING),
        ])
        assert report.outcome == PreflightOutcome.PASS
        assert report.passed_count == 2
        assert report.failed == []

    def test_warning_only(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=True, severity=CheckSeverity.CRITICAL),
            CheckResult(name="b", passed=False, severity=CheckSeverity.WARNING, message="minor"),
        ])
        assert report.outcome == PreflightOutcome.WARN
        assert len(report.warnings) == 1
        assert report.critical_failures == []

    def test_critical_failure(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=False, severity=CheckSeverity.CRITICAL, message="boom"),
            CheckResult(name="b", passed=True, severity=CheckSeverity.WARNING),
        ])
        assert report.outcome == PreflightOutcome.FAIL
        assert len(report.critical_failures) == 1
        assert report.critical_failures[0].name == "a"

    def test_mixed_failures(self):
        report = PreflightReport(checks=[
            CheckResult(name="a", passed=False, severity=CheckSeverity.CRITICAL, message="x"),
            CheckResult(name="b", passed=False, severity=CheckSeverity.WARNING, message="y"),
            CheckResult(name="c", passed=True, severity=CheckSeverity.CRITICAL),
        ])
        assert report.outcome == PreflightOutcome.FAIL
        assert len(report.failed) == 2
        assert report.total_count == 3
        assert report.passed_count == 1

    def test_empty_report_passes(self):
        report = PreflightReport()
        assert report.outcome == PreflightOutcome.PASS


# ──────────────────────────────────────────────────────────────────────────
# Individual check tests
# ──────────────────────────────────────────────────────────────────────────


class TestCheckTimezoneData:
    def test_passes_when_tzdata_available(self):
        result = check_timezone_data()
        # On dev machines and Docker with tzdata, this should pass
        assert result.passed is True
        assert result.name == "timezone_data"

    def test_fails_when_zoneinfo_raises(self):
        """Simulate missing tzdata by patching ZoneInfo at its source."""
        with patch.dict(sys.modules, {"zoneinfo": None}):
            # Force re-import failure path
            result = check_timezone_data()
        # When zoneinfo can't be imported, the check should fail
        assert result.failed is True
        assert result.severity == CheckSeverity.CRITICAL
        assert "tzdata" in result.message.lower() or "timezone" in result.message.lower()


class TestCheckAlpacaCredentials:
    @dataclass
    class _MockSettings:
        trading_mode: str = "paper"
        alpaca_paper_api_key: str = "PK_VALID_KEY"
        alpaca_paper_secret_key: str = "SK_VALID_SECRET"
        alpaca_live_api_key: str = ""
        alpaca_live_secret_key: str = ""

    def test_passes_with_valid_paper_keys(self):
        settings = self._MockSettings()
        result = check_alpaca_credentials(settings)
        assert result.passed is True

    def test_fails_with_empty_keys(self):
        settings = self._MockSettings(alpaca_paper_api_key="", alpaca_paper_secret_key="")
        result = check_alpaca_credentials(settings)
        assert result.passed is False
        assert "PAPER" in result.message

    def test_fails_with_placeholder_keys(self):
        settings = self._MockSettings(alpaca_paper_api_key="your_api_key_here")
        result = check_alpaca_credentials(settings)
        assert result.passed is False

    def test_checks_live_keys_in_live_mode(self):
        settings = self._MockSettings(trading_mode="live")
        result = check_alpaca_credentials(settings)
        assert result.passed is False
        assert "LIVE" in result.message


class TestCheckTelegramConfig:
    @dataclass
    class _MockSettings:
        telegram_bot_token: str = "123:ABC"
        telegram_chat_id: str = "456789"

    def test_passes_with_both_set(self):
        result = check_telegram_config(self._MockSettings())
        assert result.passed is True

    def test_warns_without_token(self):
        settings = self._MockSettings(telegram_bot_token="")
        result = check_telegram_config(settings)
        assert result.passed is False
        assert result.severity == CheckSeverity.WARNING
        assert "TELEGRAM_BOT_TOKEN" in result.message

    def test_warns_without_chat_id(self):
        settings = self._MockSettings(telegram_chat_id="")
        result = check_telegram_config(settings)
        assert result.passed is False
        assert result.severity == CheckSeverity.WARNING
        assert "TELEGRAM_CHAT_ID" in result.message


class TestCheckCriticalImports:
    def test_passes_when_all_available(self):
        result = check_critical_imports()
        assert result.passed is True
        assert result.name == "critical_imports"


class TestCheckDatabaseWritable:
    @dataclass
    class _MockSettings:
        database_url: str = ""

    def test_passes_with_writable_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            settings = self._MockSettings(database_url=f"sqlite:///{db_path}")
            result = check_database_writable(settings)
            assert result.passed is True

    def test_fails_with_unwritable_dir(self):
        # Use a path that definitely doesn't exist and can't be created
        if sys.platform == "win32":
            bad_path = "Z:\\nonexistent_drive\\no_way\\test.db"
        else:
            bad_path = "/proc/nonexistent/test.db"
        settings = self._MockSettings(database_url=f"sqlite:///{bad_path}")
        result = check_database_writable(settings)
        assert result.passed is False
        assert result.severity == CheckSeverity.CRITICAL


# ──────────────────────────────────────────────────────────────────────────
# Integration: run_preflight
# ──────────────────────────────────────────────────────────────────────────


class TestRunPreflight:
    @dataclass
    class _FullSettings:
        trading_mode: str = "paper"
        alpaca_paper_api_key: str = "PK_TEST"
        alpaca_paper_secret_key: str = "SK_TEST"
        alpaca_live_api_key: str = ""
        alpaca_live_secret_key: str = ""
        telegram_bot_token: str = "123:ABC"
        telegram_chat_id: str = "999"
        database_url: str = "sqlite:///data_cache/trading.db"

    def test_full_preflight_passes(self):
        report = run_preflight(self._FullSettings())
        assert report.outcome in (PreflightOutcome.PASS, PreflightOutcome.WARN)
        assert report.total_count == 5  # 5 checks registered

    def test_full_preflight_reports_credential_failure(self):
        settings = self._FullSettings(alpaca_paper_api_key="")
        report = run_preflight(settings)
        assert report.outcome == PreflightOutcome.FAIL
        crit = [c for c in report.critical_failures if c.name == "alpaca_credentials"]
        assert len(crit) == 1
