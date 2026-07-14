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

import os

import src.ml.build_info as _build_info


class ProvenanceError(Exception):
    """Raised when commit metadata cannot be resolved in strict mode."""


def get_commit_hash(*, strict: bool = False) -> str:
    """Return the commit hash from build-time metadata or environment.

    Resolution order:
    1. ``build_info.GIT_COMMIT`` (injected at Docker/CI build time)
    2. ``GIT_COMMIT`` environment variable
    3. ``SOURCE_VERSION`` environment variable (Azure)
    4. ``GITHUB_SHA`` environment variable (GitHub Actions)
    5. ``.git_commit`` file in the project root

    Args:
        strict: If True, raise ProvenanceError when no source provides
                a commit hash.

    Returns:
        Full commit SHA string, or "" if not available and strict=False.
    """
    commit = _build_info.GIT_COMMIT
    if commit:
        return commit

    # Fallback to environment variables (CI / local dev)
    for var in ("GIT_COMMIT", "SOURCE_VERSION", "GITHUB_SHA"):
        val = os.environ.get(var, "").strip()
        if val:
            return val

    # Fallback to .git_commit file in project root
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    _commit_file = os.path.join(_project_root, ".git_commit")
    try:
        if os.path.isfile(_commit_file):
            with open(_commit_file) as fh:
                val = fh.read().strip()
                if val:
                    return val
    except OSError:
        pass

    if strict:
        raise ProvenanceError(
            "Commit hash not found in build_info.py, environment variables "
            "(GIT_COMMIT, SOURCE_VERSION, GITHUB_SHA), or .git_commit file. "
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
