"""
P2-08: Basic integration smoke test.

Verifies that key modules can be imported and core classes instantiated
without error, ensuring the dependency graph is intact.
"""

import sys


def test_imports():
    """Verify key modules are importable."""
    from src.broker import alpaca_client
    from src.execution import engine
    from src.risk import engine as risk_engine
    from src.core import events

    assert hasattr(alpaca_client, "AlpacaBroker")
    assert hasattr(engine, "ExecutionEngine")
    assert hasattr(risk_engine, "RiskEngine")
    assert hasattr(events, "EventBus")
    assert hasattr(events, "Event")


def test_event_bus_instantiation():
    """Verify EventBus can be created."""
    from src.core.events import EventBus

    bus = EventBus()
    assert bus is not None
    assert hasattr(bus, "publish")
    assert hasattr(bus, "subscribe")


def test_risk_engine_instantiation():
    """Verify RiskEngine can be created with defaults."""
    from src.risk.engine import RiskEngine

    re = RiskEngine()
    assert re is not None


if __name__ == "__main__":
    test_imports()
    test_event_bus_instantiation()
    test_risk_engine_instantiation()
    print("[OK] All integration smoke tests passed.")
    sys.exit(0)
