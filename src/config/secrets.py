"""
Secrets Management — Reference-based secrets handling.

Ensures sensitive secrets (API keys, tokens, credentials) remain in
environment variables or dedicated secret managers rather than being
stored in the database. The database stores references to secrets,
not the secrets themselves.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Secret keys that must NEVER be stored in the database
PROTECTED_SECRETS = frozenset([
    "alpaca_paper_api_key",
    "alpaca_paper_secret_key",
    "alpaca_live_api_key",
    "alpaca_live_secret_key",
    "telegram_bot_token",
    "discord_webhook_url",
    "database_url",
    "jwt_secret",
    "encryption_key",
    "smtp_password",
])


@dataclass
class SecretReference:
    """
    A reference to a secret stored outside the database.

    The database stores the reference (env var name or vault path),
    not the actual secret value.
    """

    name: str
    source: str  # "env", "vault", "aws_ssm", "gcp_secrets"
    reference: str  # Environment variable name or vault path
    description: str = ""

    def resolve(self) -> Optional[str]:
        """Resolve the secret reference to its actual value."""
        if self.source == "env":
            return os.environ.get(self.reference)
        # Extensible for vault, AWS SSM, GCP Secrets, etc.
        logger.warning(
            "secrets.unsupported_source",
            name=self.name,
            source=self.source,
        )
        return None


class SecretsManager:
    """
    Manages secret references and resolution.

    Ensures secrets are never stored in the database, only references
    to their location in environment variables or secret managers.

    Features:
    - Validates that protected keys are not stored in the config DB
    - Resolves secret references from environment variables
    - Supports pluggable secret backends
    - Redacts secrets in logs and exports
    """

    def __init__(self):
        self._references: dict[str, SecretReference] = {}
        self._register_default_references()

    def _register_default_references(self) -> None:
        """Register default secret references from environment variables."""
        default_mappings = {
            "alpaca_paper_api_key": "ALPACA_PAPER_API_KEY",
            "alpaca_paper_secret_key": "ALPACA_PAPER_SECRET_KEY",
            "alpaca_live_api_key": "ALPACA_LIVE_API_KEY",
            "alpaca_live_secret_key": "ALPACA_LIVE_SECRET_KEY",
            "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
            "discord_webhook_url": "DISCORD_WEBHOOK_URL",
            "database_url": "DATABASE_URL",
            "jwt_secret": "JWT_SECRET",
            "encryption_key": "ENCRYPTION_KEY",
            "smtp_password": "SMTP_PASSWORD",
        }
        for name, env_var in default_mappings.items():
            self._references[name] = SecretReference(
                name=name,
                source="env",
                reference=env_var,
                description=f"Secret resolved from ${env_var}",
            )

    def register(self, reference: SecretReference) -> None:
        """Register a secret reference."""
        self._references[reference.name] = reference

    def resolve(self, name: str) -> Optional[str]:
        """
        Resolve a secret by name.

        Returns the secret value if found, None otherwise.
        """
        ref = self._references.get(name)
        if ref is None:
            logger.warning("secrets.no_reference", name=name)
            return None
        return ref.resolve()

    def is_protected(self, key: str) -> bool:
        """Check if a key is a protected secret that shouldn't be in the DB."""
        return key in PROTECTED_SECRETS

    def validate_no_secrets_in_config(self, entries: list[dict]) -> list[str]:
        """
        Validate that no protected secrets are stored in configuration entries.

        Returns list of violation messages.
        """
        violations = []
        for entry in entries:
            key = entry.get("key", "")
            if self.is_protected(key) and not entry.get("is_secret", False):
                violations.append(
                    f"Protected secret '{key}' should not be stored in config DB"
                )
            # Check if value looks like a secret (heuristic)
            value = entry.get("value", "")
            if self._looks_like_secret(value) and not entry.get("is_secret", False):
                violations.append(
                    f"Value for '{key}' looks like a secret - "
                    f"consider using environment variables instead"
                )
        return violations

    def redact_value(self, key: str, value: str) -> str:
        """Redact a secret value for safe logging/display."""
        if self.is_protected(key) or self._looks_like_secret(value):
            if len(value) > 8:
                return value[:4] + "****" + value[-4:]
            return "****"
        return value

    def get_all_references(self) -> dict[str, dict[str, str]]:
        """Get all registered secret references (without values)."""
        return {
            name: {
                "source": ref.source,
                "reference": ref.reference,
                "description": ref.description,
                "is_configured": ref.resolve() is not None,
            }
            for name, ref in self._references.items()
        }

    @staticmethod
    def _looks_like_secret(value: str) -> bool:
        """Heuristic check if a value looks like a secret."""
        if not value or len(value) < 16:
            return False
        # Check for common patterns
        secret_prefixes = ("sk-", "pk-", "ak-", "xoxb-", "ghp_", "gho_")
        if any(value.startswith(prefix) for prefix in secret_prefixes):
            return True
        # High entropy check (many uppercase, digits, special chars)
        special_count = sum(1 for c in value if not c.isalnum())
        upper_count = sum(1 for c in value if c.isupper())
        digit_count = sum(1 for c in value if c.isdigit())
        complexity = (special_count + upper_count + digit_count) / len(value)
        return complexity > 0.5
