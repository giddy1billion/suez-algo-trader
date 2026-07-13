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
GIT_COMMIT: str = ""
GIT_BRANCH: str = ""
BUILD_TIMESTAMP: str = ""
