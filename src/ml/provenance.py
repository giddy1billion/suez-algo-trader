"""
Centralized provenance resolution — single source of truth for commit metadata.

All components that need the build-time commit hash, branch, or timestamp
MUST import from this module.  No other module should directly access
environment variables, Git CLI, .git directories, or .git_commit files
for commit resolution.

Resolution order (strict, no fallback past build_info in production):
1. ``src/ml/build_info.py`` constants (injected at Docker/CI build time)

If build_info is empty (local development only), raises or returns empty
depending on the ``strict`` flag.  Legacy fallbacks (env vars, git CLI,
.git_commit file) are intentionally **removed** to guarantee that a
production artifact carries the identical embedded commit hash everywhere.

Usage:
    from src.ml.provenance import get_commit_hash, get_branch, get_build_timestamp
"""

from __future__ import annotations

import src.ml.build_info as _build_info


class ProvenanceError(Exception):
    """Raised when commit metadata cannot be resolved in strict mode."""


def get_commit_hash(*, strict: bool = False) -> str:
    """Return the build-time injected commit hash.

    Args:
        strict: If True, raise ProvenanceError when build_info is empty
                (i.e. the artifact was not properly stamped).

    Returns:
        Full commit SHA string, or "" if not stamped and strict=False.
    """
    commit = _build_info.GIT_COMMIT
    if commit:
        return commit
    if strict:
        raise ProvenanceError(
            "Build-time commit hash not found in build_info.py. "
            "Ensure scripts/inject_build_info.py ran during the build."
        )
    return ""


def get_branch() -> str:
    """Return the build-time injected branch name, or '' if not stamped."""
    return _build_info.GIT_BRANCH


def get_build_timestamp() -> str:
    """Return the ISO-8601 build timestamp, or '' if not stamped."""
    return _build_info.BUILD_TIMESTAMP


def get_short_commit(*, length: int = 7, strict: bool = False) -> str:
    """Return abbreviated commit hash (first *length* chars)."""
    full = get_commit_hash(strict=strict)
    return full[:length] if full else ""
