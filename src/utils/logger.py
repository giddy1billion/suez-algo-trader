"""
Structured logging configuration using structlog.
Outputs JSON in production, colored in dev.
Includes sanitization to prevent credential leakage in logs.
"""

import logging
import re
import sys
from pathlib import Path

import structlog


# Patterns that may contain credentials in exception messages
_SENSITIVE_PATTERNS = [
    # API keys / tokens
    re.compile(r'(api[_-]?key|token|secret|password|authorization|bearer)\s*[=:]\s*\S+', re.IGNORECASE),
    # URLs with embedded credentials
    re.compile(r'https?://[^@\s]+:[^@\s]+@', re.IGNORECASE),
    # Common key formats (alphanumeric strings 20+ chars following a key-like word)
    re.compile(r'(key|secret|token)\s*[=:]\s*[A-Za-z0-9+/]{20,}', re.IGNORECASE),
]

_REDACTION = "[REDACTED]"


def sanitize_log_value(value: str) -> str:
    """Remove potential secrets from a log value string."""
    if not isinstance(value, str):
        return value
    result = value
    for pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub(_REDACTION, result)
    return result


def _sanitize_processor(logger, method_name, event_dict):
    """Structlog processor that sanitizes sensitive data from log output."""
    for key in list(event_dict.keys()):
        if key in ("error", "exception", "exc_info", "message", "msg"):
            val = event_dict[key]
            if isinstance(val, str):
                event_dict[key] = sanitize_log_value(val)
    return event_dict


def setup_logging(log_level: str = "INFO", log_file: str = "logs/trader.log"):
    """Configure structured logging for the entire application."""

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Standard library logging config
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _sanitize_processor,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None):
    """Get a structured logger instance."""
    return structlog.get_logger(name)
