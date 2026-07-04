"""
Configuration Dependency Validation — Validates inter-dependent settings.

Ensures that when a feature is enabled, all required dependent settings
are properly configured.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.config.validation import ValidationResult, ValidationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DependencyRule:
    """
    Defines a configuration dependency relationship.

    When `condition` is True, all `required_keys` must be properly configured.
    """

    name: str
    description: str
    condition: Callable[[Any], bool]
    required_keys: list[str]
    validate_key: Optional[Callable[[Any, str], bool]] = None

    def check(self, config: Any) -> list[ValidationResult]:
        """Evaluate this dependency rule against the configuration."""
        results = []

        if not self.condition(config):
            return results  # Condition not met, no validation needed

        for key in self.required_keys:
            value = _get_nested_attr(config, key)

            if value is None or value == "":
                results.append(ValidationResult(
                    check=f"dependency.{self.name}.{key}",
                    passed=False,
                    message=f"{self.name}: requires '{key}' to be configured",
                ))
            elif self.validate_key and not self.validate_key(config, key):
                results.append(ValidationResult(
                    check=f"dependency.{self.name}.{key}",
                    passed=False,
                    message=f"{self.name}: '{key}' has an invalid value",
                ))
            else:
                results.append(ValidationResult(
                    check=f"dependency.{self.name}.{key}",
                    passed=True,
                ))

        return results


def _get_nested_attr(obj: Any, key: str) -> Any:
    """Get a potentially nested attribute using dot notation."""
    parts = key.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


# ─── Built-in Dependency Rules ────────────────────────────────────────────────

DEFAULT_DEPENDENCY_RULES: list[DependencyRule] = [
    # ML strategy requires model configuration
    DependencyRule(
        name="ml_strategy",
        description="ML strategy requires model path and confidence threshold",
        condition=lambda cfg: getattr(cfg, "active_strategy", "") == "ml",
        required_keys=[
            "ml_model_path",
            "ml_min_confidence",
            "ml_retrain_interval_hours",
        ],
    ),
    # Live trading requires exchange credentials
    DependencyRule(
        name="live_trading",
        description="Live trading requires exchange API credentials",
        condition=lambda cfg: getattr(cfg, "trading_mode", "paper") != "paper"
        if hasattr(cfg, "trading_mode")
        else str(getattr(cfg, "trading_mode", "paper")) != "paper",
        required_keys=[
            "alpaca_live_api_key",
            "alpaca_live_secret_key",
        ],
    ),
    # Telegram notifications require bot token and chat ID
    DependencyRule(
        name="telegram_notifications",
        description="Telegram notifications require bot token and chat ID",
        condition=lambda cfg: bool(getattr(cfg, "telegram_bot_token", "")),
        required_keys=[
            "telegram_bot_token",
            "telegram_chat_id",
        ],
    ),
    # Intelligence layer requires minimum configuration
    DependencyRule(
        name="intelligence_layer",
        description="Intelligence layer requires trade score threshold",
        condition=lambda cfg: getattr(cfg, "intelligence_enabled", False),
        required_keys=[
            "intelligence_min_trade_score",
            "intelligence_drift_window",
            "intelligence_drift_min_samples",
        ],
    ),
    # Multi-strategy mode requires config string
    DependencyRule(
        name="multi_strategy",
        description="Multi-strategy mode requires strategy configuration",
        condition=lambda cfg: getattr(cfg, "active_strategy", "") == "multi",
        required_keys=[
            "multi_strategy_config",
        ],
    ),
]


class DependencyValidator:
    """
    Validates configuration dependencies.

    Ensures that when features are enabled, all their required dependencies
    are properly configured.
    """

    def __init__(self, rules: Optional[list[DependencyRule]] = None):
        self._rules = rules if rules is not None else list(DEFAULT_DEPENDENCY_RULES)

    def add_rule(self, rule: DependencyRule) -> None:
        """Add a custom dependency rule."""
        self._rules.append(rule)

    def validate(self, config: Any) -> ValidationReport:
        """
        Validate all dependency rules against the given configuration.

        Returns a ValidationReport with results from all applicable rules.
        """
        report = ValidationReport()

        for rule in self._rules:
            try:
                results = rule.check(config)
                report.results.extend(results)
            except Exception as e:
                logger.error(
                    "dependency_validator.rule_error",
                    rule=rule.name,
                    error=str(e),
                )
                report.results.append(ValidationResult(
                    check=f"dependency.{rule.name}",
                    passed=False,
                    message=f"Rule evaluation failed: {e}",
                ))

        if report.passed:
            logger.debug("dependency_validation.passed", rules=len(self._rules))
        else:
            logger.warning(
                "dependency_validation.failed",
                errors=len(report.errors),
            )

        return report
