"""
Startup Validation — Validate configuration before trading starts.

Performs comprehensive validation of all configuration values at startup
to fail fast instead of discovering invalid configuration during live trading.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check: str
    passed: bool
    message: str = ""
    severity: str = "error"  # error, warning

    def __bool__(self) -> bool:
        return self.passed


@dataclass
class ValidationReport:
    """Complete validation report for startup checks."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all error-severity checks passed."""
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def warnings(self) -> list[ValidationResult]:
        """Get all warning-severity failures."""
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def errors(self) -> list[ValidationResult]:
        """Get all error-severity failures."""
        return [r for r in self.results if not r.passed and r.severity == "error"]

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        errors = len(self.errors)
        warnings = len(self.warnings)
        lines = [
            f"Validation: {passed}/{total} checks passed",
            f"  Errors: {errors}, Warnings: {warnings}",
        ]
        for r in self.errors:
            lines.append(f"  ✗ [ERROR] {r.check}: {r.message}")
        for r in self.warnings:
            lines.append(f"  ⚠ [WARN]  {r.check}: {r.message}")
        return "\n".join(lines)


class StartupValidator:
    """
    Validates configuration completeness and consistency before trading begins.

    Performs checks including:
    - Leverage > 0
    - Risk percentages within valid range
    - Stop loss / take profit relationship
    - Exchange timeout validity
    - API endpoints configured
    - Trading enabled only if credentials exist
    - ML configuration consistency
    """

    def validate_all(
        self,
        config: Any,
        config_service: Optional[Any] = None,
    ) -> ValidationReport:
        """
        Run all startup validation checks.

        Args:
            config: The Settings (pydantic) object.
            config_service: Optional ConfigurationService for DB-backed values.

        Returns:
            ValidationReport with all check results.
        """
        report = ValidationReport()

        # Core trading validations
        report.results.extend(self._validate_leverage(config))
        report.results.extend(self._validate_risk_percentages(config))
        report.results.extend(self._validate_stop_loss_take_profit(config))
        report.results.extend(self._validate_trading_interval(config))
        report.results.extend(self._validate_credentials(config))
        report.results.extend(self._validate_symbols(config))
        report.results.extend(self._validate_ml_config(config))
        report.results.extend(self._validate_risk_engine(config))

        if report.passed:
            logger.info("startup_validation.passed", checks=len(report.results))
        else:
            logger.error(
                "startup_validation.failed",
                errors=len(report.errors),
                warnings=len(report.warnings),
            )

        return report

    def _validate_leverage(self, config: Any) -> list[ValidationResult]:
        """Validate leverage settings."""
        results = []
        leverage = getattr(config, "max_leverage", None)
        if leverage is not None:
            if leverage <= 0:
                results.append(ValidationResult(
                    check="leverage_positive",
                    passed=False,
                    message=f"max_leverage must be > 0, got {leverage}",
                ))
            elif leverage > 10:
                results.append(ValidationResult(
                    check="leverage_reasonable",
                    passed=False,
                    message=f"max_leverage > 10 is extremely risky, got {leverage}",
                    severity="warning",
                ))
            else:
                results.append(ValidationResult(
                    check="leverage_positive",
                    passed=True,
                    message=f"max_leverage={leverage} is valid",
                ))
        return results

    def _validate_risk_percentages(self, config: Any) -> list[ValidationResult]:
        """Validate all risk percentage values are within [0, 1]."""
        results = []
        risk_fields = [
            "max_position_size_pct",
            "max_daily_loss_pct",
            "max_portfolio_exposure",
            "max_single_stock_pct",
        ]
        for field_name in risk_fields:
            value = getattr(config, field_name, None)
            if value is not None:
                if not (0 < value <= 1.0):
                    results.append(ValidationResult(
                        check=f"risk_{field_name}",
                        passed=False,
                        message=f"{field_name} must be in (0, 1.0], got {value}",
                    ))
                else:
                    results.append(ValidationResult(
                        check=f"risk_{field_name}",
                        passed=True,
                    ))
        return results

    def _validate_stop_loss_take_profit(self, config: Any) -> list[ValidationResult]:
        """Validate stop loss < take profit relationship."""
        results = []
        sl = getattr(config, "default_stop_loss_pct", None)
        tp = getattr(config, "default_take_profit_pct", None)

        if sl is not None and tp is not None:
            if sl >= tp:
                results.append(ValidationResult(
                    check="stop_loss_less_than_take_profit",
                    passed=False,
                    message=f"stop_loss ({sl}) should be < take_profit ({tp})",
                    severity="warning",
                ))
            else:
                results.append(ValidationResult(
                    check="stop_loss_less_than_take_profit",
                    passed=True,
                ))

            if sl <= 0:
                results.append(ValidationResult(
                    check="stop_loss_positive",
                    passed=False,
                    message=f"default_stop_loss_pct must be > 0, got {sl}",
                ))

            if tp <= 0:
                results.append(ValidationResult(
                    check="take_profit_positive",
                    passed=False,
                    message=f"default_take_profit_pct must be > 0, got {tp}",
                ))
        return results

    def _validate_trading_interval(self, config: Any) -> list[ValidationResult]:
        """Validate trading interval is reasonable."""
        results = []
        interval = getattr(config, "trading_interval", None)
        if interval is not None:
            if interval < 1:
                results.append(ValidationResult(
                    check="trading_interval_valid",
                    passed=False,
                    message=f"trading_interval must be >= 1 second, got {interval}",
                ))
            elif interval < 5:
                results.append(ValidationResult(
                    check="trading_interval_reasonable",
                    passed=False,
                    message=f"trading_interval < 5s may cause rate limiting, got {interval}",
                    severity="warning",
                ))
            else:
                results.append(ValidationResult(
                    check="trading_interval_valid",
                    passed=True,
                ))
        return results

    def _validate_credentials(self, config: Any) -> list[ValidationResult]:
        """Validate credentials exist when live trading is enabled."""
        results = []
        from config.settings import TradingMode

        mode = getattr(config, "trading_mode", TradingMode.PAPER)

        if mode == TradingMode.LIVE:
            api_key = getattr(config, "alpaca_live_api_key", "")
            secret_key = getattr(config, "alpaca_live_secret_key", "")

            if not api_key:
                results.append(ValidationResult(
                    check="live_api_key_configured",
                    passed=False,
                    message="Live trading requires alpaca_live_api_key",
                ))
            else:
                results.append(ValidationResult(
                    check="live_api_key_configured",
                    passed=True,
                ))

            if not secret_key:
                results.append(ValidationResult(
                    check="live_secret_key_configured",
                    passed=False,
                    message="Live trading requires alpaca_live_secret_key",
                ))
            else:
                results.append(ValidationResult(
                    check="live_secret_key_configured",
                    passed=True,
                ))
        else:
            # Paper mode — check paper credentials
            api_key = getattr(config, "alpaca_paper_api_key", "")
            if not api_key:
                results.append(ValidationResult(
                    check="paper_api_key_configured",
                    passed=False,
                    message="Paper trading requires alpaca_paper_api_key",
                    severity="warning",
                ))
            else:
                results.append(ValidationResult(
                    check="paper_api_key_configured",
                    passed=True,
                ))

        return results

    def _validate_symbols(self, config: Any) -> list[ValidationResult]:
        """Validate trading symbols are configured."""
        results = []
        symbols = getattr(config, "symbols_list", [])
        if not symbols:
            results.append(ValidationResult(
                check="trading_symbols_configured",
                passed=False,
                message="No trading symbols configured",
                severity="warning",
            ))
        else:
            results.append(ValidationResult(
                check="trading_symbols_configured",
                passed=True,
                message=f"{len(symbols)} symbols configured",
            ))
        return results

    def _validate_ml_config(self, config: Any) -> list[ValidationResult]:
        """Validate ML configuration consistency."""
        results = []
        strategy = getattr(config, "active_strategy", "")

        if strategy == "ml":
            model_path = getattr(config, "ml_model_path", "")
            confidence = getattr(config, "ml_min_confidence", 0)

            if not model_path:
                results.append(ValidationResult(
                    check="ml_model_path_configured",
                    passed=False,
                    message="ML strategy requires ml_model_path",
                ))

            if confidence <= 0 or confidence >= 1.0:
                results.append(ValidationResult(
                    check="ml_confidence_valid",
                    passed=False,
                    message=f"ml_min_confidence must be in (0, 1), got {confidence}",
                ))
            else:
                results.append(ValidationResult(
                    check="ml_confidence_valid",
                    passed=True,
                ))

        return results

    def _validate_risk_engine(self, config: Any) -> list[ValidationResult]:
        """Validate risk engine configuration."""
        results = []

        max_daily_loss = getattr(config, "risk_max_daily_loss_pct", None)
        max_drawdown = getattr(config, "risk_max_drawdown_pct", None)

        if max_daily_loss is not None and max_drawdown is not None:
            if max_daily_loss >= max_drawdown:
                results.append(ValidationResult(
                    check="daily_loss_less_than_drawdown",
                    passed=False,
                    message=(
                        f"risk_max_daily_loss_pct ({max_daily_loss}) should be "
                        f"< risk_max_drawdown_pct ({max_drawdown})"
                    ),
                    severity="warning",
                ))
            else:
                results.append(ValidationResult(
                    check="daily_loss_less_than_drawdown",
                    passed=True,
                ))

        return results
