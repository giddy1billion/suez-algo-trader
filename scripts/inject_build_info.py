#!/usr/bin/env python3
"""
Build-time script that stamps ``src/ml/build_info.py`` with the current
git commit hash, branch, and ISO-8601 build timestamp.

Usage (typically invoked by CI or Dockerfile):
    python scripts/inject_build_info.py [--commit SHA] [--branch NAME]

If ``--commit`` is not supplied the script resolves it from:
  1. $GIT_COMMIT / $SOURCE_VERSION / $GITHUB_SHA environment variables
  2. ``git rev-parse HEAD``

Exit codes:
  0  — build_info.py successfully stamped
  1  — unable to determine commit hash (caller should treat as fatal in CI)
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BUILD_INFO_PATH = Path(__file__).resolve().parent.parent / "src" / "ml" / "build_info.py"

TEMPLATE = '''\
"""
Build-time metadata module for model provenance.

This file is overwritten at build time by ``scripts/inject_build_info.py``
to embed the exact git commit hash (and optional branch/timestamp) into the
production artifact.  At runtime the governance layer reads the constants
below as the **highest-confidence** provenance source — no .git directory or
environment variables required.

If the file has NOT been stamped (i.e. during local development), the
sentinel values remain and the governance system falls through to its
other resolution strategies (env vars → git CLI → .git_commit file).
"""

# Injected at build time — DO NOT edit manually.
GIT_COMMIT: str = "{commit}"
GIT_BRANCH: str = "{branch}"
BUILD_TIMESTAMP: str = "{timestamp}"
'''


def _resolve_commit() -> str:
    """Resolve commit hash from environment or git CLI."""
    for var in ("GIT_COMMIT", "SOURCE_VERSION", "GITHUB_SHA"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _resolve_branch() -> str:
    """Resolve branch from environment or git CLI."""
    for var in ("GIT_BRANCH", "GITHUB_REF_NAME"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject build metadata into build_info.py")
    parser.add_argument("--commit", default="", help="Explicit commit SHA to embed")
    parser.add_argument("--branch", default="", help="Explicit branch name to embed")
    args = parser.parse_args()

    commit = args.commit or _resolve_commit()
    branch = args.branch or _resolve_branch()
    timestamp = datetime.now(timezone.utc).isoformat()

    if not commit:
        print("ERROR: Unable to determine git commit hash.", file=sys.stderr)
        print("  Set GIT_COMMIT env var or pass --commit SHA.", file=sys.stderr)
        return 1

    content = TEMPLATE.format(commit=commit, branch=branch, timestamp=timestamp)
    BUILD_INFO_PATH.write_text(content)
    print(f"build_info.py stamped: commit={commit[:12]}, branch={branch}, ts={timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
