"""Tests for Operational Modes (Phase 6)."""

from unittest.mock import MagicMock, patch

import pytest

from config.settings import OperationalMode
from src.core.events import OperationalModeChanged


class TestOperationalModes:
    """Test operational mode transitions."""

    @pytest.fixture
    def runtime_manager(self):
        """Create a RuntimeManager with mocked dependencies."""
        with patch("src.core.runtime.ModelRegistry"), \
             patch("src.core.runtime.ModelGovernance"), \
             patch("src.core.runtime.ModelPredictor"), \
             patch("src.core.runtime.TrainingPipeline"), \
             patch("src.core.runtime.ABTestManager"), \
             patch("src.core.runtime.BacktestRunner"), \
             patch("src.core.runtime.EnvironmentManager"):
            from src.core.runtime import RuntimeManager
            from src.core.environment import BrokerManager

            broker_manager = MagicMock(spec=BrokerManager)
            broker_manager.broker = MagicMock()
            broker_manager.get_status.return_value = {"name": "paper"}

            event_bus = MagicMock()
            rm = RuntimeManager(
                broker_manager=broker_manager,
                event_bus=event_bus,
                operational_mode=OperationalMode.PAPER,
            )
            return rm

    def test_initial_mode(self, runtime_manager):
        assert runtime_manager.operational_mode == OperationalMode.PAPER

    def test_switch_to_research(self, runtime_manager):
        result = runtime_manager.switch_operational_mode("research", reason="testing")
        assert result["new_mode"] == "research"
        assert result["old_mode"] == "paper"
        assert runtime_manager.operational_mode == OperationalMode.RESEARCH

    def test_switch_to_live_from_paper(self, runtime_manager):
        result = runtime_manager.switch_operational_mode("live", reason="go live")
        assert result["new_mode"] == "live"

    def test_cannot_switch_research_to_live(self, runtime_manager):
        runtime_manager.switch_operational_mode("research")
        with pytest.raises(RuntimeError, match="Cannot transition directly"):
            runtime_manager.switch_operational_mode("live")

    def test_switch_unchanged(self, runtime_manager):
        result = runtime_manager.switch_operational_mode("paper")
        assert result["status"] == "unchanged"

    def test_invalid_mode_raises(self, runtime_manager):
        with pytest.raises(ValueError, match="Invalid operational mode"):
            runtime_manager.switch_operational_mode("invalid_mode")

    def test_is_research_mode(self, runtime_manager):
        assert runtime_manager.is_research_mode is False
        runtime_manager.switch_operational_mode("research")
        assert runtime_manager.is_research_mode is True

    def test_can_execute_trades(self, runtime_manager):
        assert runtime_manager.can_execute_trades is True
        runtime_manager.switch_operational_mode("research")
        assert runtime_manager.can_execute_trades is False

    def test_event_published_on_switch(self, runtime_manager):
        runtime_manager.switch_operational_mode("research", reason="test")
        runtime_manager._event_bus.publish.assert_called()
        call_args = runtime_manager._event_bus.publish.call_args
        event = call_args[0][0]
        assert isinstance(event, OperationalModeChanged)
        assert event.old_mode == "paper"
        assert event.new_mode == "research"

    def test_status_includes_operational_mode(self, runtime_manager):
        status = runtime_manager.get_status()
        assert "operational_mode" in status["environment"]
