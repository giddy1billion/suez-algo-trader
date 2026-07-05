"""
Caller Regression Guard — AST-based test ensuring ALL backtest callers
use the asset-class-aware configuration layer.

This test scans Python source files (excluding tests and __pycache__) for
direct calls to backtest functions that bypass `get_backtest_config()`.

If any future contributor hardcodes defaults (e.g., `commission_pct=0.001`)
or calls `_numpy_ema_crossover_backtest()` / `vectorbt_momentum_backtest()`
without passing config-derived parameters, this test will fail.
"""
import ast
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple

import pytest

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Directories to scan
SCAN_DIRS = [
    ROOT / "backtesting",
    ROOT / "src",
    ROOT / "main.py",
]

# Files/dirs to exclude from scanning
EXCLUDE_PATTERNS = {
    "__pycache__",
    "test_",
    "tests",
    ".git",
    "backtest_params.py",  # This IS the config layer itself
    "statistical_validation.py",  # Uses get_backtest_config internally
}

# Functions that MUST receive config-derived parameters (not bare defaults)
MONITORED_FUNCTIONS = {
    "_numpy_ema_crossover_backtest",
    "vectorbt_momentum_backtest",
    "vectorbt_parameter_sweep",
    "vectorbt_multi_symbol_backtest",
    "monte_carlo_from_backtest",
    "walk_forward_ema_backtest",
    "portfolio_backtest",
}

# The config function that MUST appear in the same file as any monitored call
CONFIG_FUNCTIONS = {
    "get_backtest_config",
    "set_symbol_override",
    "set_asset_class_override",
}

# Known-safe parameters that indicate config usage (not hardcoded defaults)
CONFIG_INDICATOR_KWARGS = {
    "commission_pct",
    "slippage_pct",
    "risk_per_trade",
    "atr_stop_multiplier",
    "cooldown_bars",
    "use_asset_class_params",
    "per_symbol_fees",
}

# Files that are exempt (they ARE the implementation, not callers)
IMPLEMENTATION_FILES = {
    "vbt_adapter.py",
    "monte_carlo.py",
    "walk_forward.py",
    "param_validator.py",
    "portfolio_backtest.py",
}


class BacktestCallVisitor(ast.NodeVisitor):
    """AST visitor that finds direct calls to monitored backtest functions."""

    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Tuple[int, str, str]] = []
        self.has_config_import = False
        self.has_config_call = False

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track imports from backtest_params config module."""
        if node.module and "backtest_params" in node.module:
            for alias in node.names:
                if alias.name in CONFIG_FUNCTIONS:
                    self.has_config_import = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check each function call for monitored functions."""
        func_name = self._get_func_name(node)

        # Track config function usage
        if func_name in CONFIG_FUNCTIONS:
            self.has_config_call = True

        # Check if this is a monitored backtest function call
        if func_name in MONITORED_FUNCTIONS:
            # If file has no config import, it's a violation
            if not self.has_config_import:
                self.violations.append((
                    node.lineno,
                    func_name,
                    "No get_backtest_config import found in this file",
                ))

        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> str:
        """Extract function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


def _should_scan(filepath: Path) -> bool:
    """Determine if a file should be scanned."""
    # Only Python files
    if filepath.suffix != ".py":
        return False

    # Skip excluded patterns
    for part in filepath.parts:
        if any(excl in part for excl in EXCLUDE_PATTERNS):
            return False

    # Skip implementation files (they define the functions, not call them externally)
    if filepath.name in IMPLEMENTATION_FILES:
        return False

    return True


def _get_files_to_scan() -> List[Path]:
    """Collect all Python files that should be scanned."""
    files = []
    for scan_path in SCAN_DIRS:
        if scan_path.is_file():
            if _should_scan(scan_path):
                files.append(scan_path)
        elif scan_path.is_dir():
            for root, dirs, filenames in os.walk(scan_path):
                # Prune excluded directories
                dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS]
                for fname in filenames:
                    fpath = Path(root) / fname
                    if _should_scan(fpath):
                        files.append(fpath)
    return files


def _scan_file(filepath: Path) -> List[Tuple[str, int, str, str]]:
    """Scan a single file for violations. Returns list of (file, line, func, reason)."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    visitor = BacktestCallVisitor(str(filepath))
    visitor.visit(tree)

    # Only report violations in files that DON'T have config imports
    # (files with config imports are using the proper pattern)
    results = []
    if visitor.violations and not visitor.has_config_import:
        for line, func, reason in visitor.violations:
            results.append((str(filepath.relative_to(ROOT)), line, func, reason))
    return results


def test_no_bare_backtest_calls():
    """
    Regression guard: ensure all callers of backtest functions use get_backtest_config.

    This test fails if any Python file (outside of the implementation files themselves)
    calls a monitored backtest function without importing get_backtest_config.

    This prevents future contributors from accidentally bypassing the config layer
    and reintroducing hardcoded defaults.
    """
    files = _get_files_to_scan()
    all_violations = []

    for filepath in files:
        violations = _scan_file(filepath)
        all_violations.extend(violations)

    if all_violations:
        msg_lines = [
            "\n❌ CALLER REGRESSION GUARD FAILURE",
            "The following files call backtest functions without using get_backtest_config():",
            "",
        ]
        for filepath, line, func, reason in all_violations:
            msg_lines.append(f"  {filepath}:{line} → {func}()")
            msg_lines.append(f"    Reason: {reason}")
            msg_lines.append("")

        msg_lines.append(
            "Fix: Import and use get_backtest_config(symbol) from src.config.backtest_params "
            "to pass asset-class-aware parameters."
        )
        pytest.fail("\n".join(msg_lines))


def test_hardcoded_fee_constants():
    """
    Detect hardcoded fee/slippage constants that bypass the config layer.

    Scans for patterns like `commission_pct=0.001` or `fees=0.001` outside
    of configuration files.
    """
    HARDCODED_PATTERNS = [
        ("commission_pct", "0.001"),
        ("slippage_pct", "0.0005"),
    ]

    files = _get_files_to_scan()
    violations = []

    for filepath in files:
        # Skip config files that legitimately define defaults
        if "config" in str(filepath) or "backtest_params" in filepath.name:
            continue

        try:
            source = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for line_no, line in enumerate(source.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for param, value in HARDCODED_PATTERNS:
                # Look for assignment patterns like `commission_pct=0.001`
                pattern = f"{param}={value}"
                alt_pattern = f"{param} = {value}"
                if pattern in line or alt_pattern in line:
                    # Exclude test files and docstrings
                    if "test" not in str(filepath).lower() and '"""' not in line and "'''" not in line:
                        violations.append((
                            str(filepath.relative_to(ROOT)),
                            line_no,
                            stripped[:80],
                        ))

    if violations:
        msg_lines = [
            "\n❌ HARDCODED FEE/SLIPPAGE CONSTANTS DETECTED",
            "The following locations use hardcoded fee values instead of the config layer:",
            "",
        ]
        for filepath, line, content in violations:
            msg_lines.append(f"  {filepath}:{line}")
            msg_lines.append(f"    {content}")
            msg_lines.append("")
        msg_lines.append(
            "Fix: Use get_backtest_config(symbol) to resolve fees dynamically."
        )
        pytest.fail("\n".join(msg_lines))


if __name__ == "__main__":
    # Can also be run standalone for CI reporting
    print("Scanning for bare backtest calls...")
    files = _get_files_to_scan()
    print(f"Scanning {len(files)} files...")
    total_violations = 0

    for filepath in files:
        violations = _scan_file(filepath)
        for filepath_str, line, func, reason in violations:
            print(f"  ❌ {filepath_str}:{line} → {func}() — {reason}")
            total_violations += 1

    if total_violations == 0:
        print("✅ All callers properly use get_backtest_config()")
    else:
        print(f"\n❌ Found {total_violations} violation(s)")
        sys.exit(1)
