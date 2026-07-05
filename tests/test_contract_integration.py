"""
Integration tests for the Decision Contract pipeline.

Validates the full flow:
1. DecisionOrchestrator produces contracts
2. ContractStore persists and replays contracts
3. TradeRequest carries contract to RiskEngine
4. RiskEngine respects contract decisions
5. Contract outcome recording on trade close
6. Replay capability (given contract_id → full decision context)
7. Analytics (accuracy, calibration, stage diagnostics)
"""
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.intelligence.confidence.decision_contract import (
    Decision,
    DecisionContract,
    DecisionContractBuilder,
    DecisionProvenance,
    StageAssessment,
    StageSeverity,
    VetoAuthority,
)
from src.intelligence.confidence.contract_store import ContractStore
from src.risk.models import TradeRequest, RiskDecision


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_store():
    """Create a temporary ContractStore for testing."""
    test_dir = os.path.join(tempfile.gettempdir(), f"contract_test_{os.getpid()}")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    store = ContractStore(storage_path=test_dir)
    yield store
    store.close()
    shutil.rmtree(test_dir, ignore_errors=True)


def _build_execute_contract(symbol: str = "BTC/USD", confidence: float = 0.85) -> DecisionContract:
    """Helper: build a contract that results in EXECUTE decision."""
    builder = DecisionContractBuilder(validity_minutes=5.0)
    builder.set_symbol(symbol, "BUY")
    builder.set_provenance(model_version="v14.3", feature_set_version="fs_82")
    builder.set_sizing(position_pct=3.2, kelly=0.08, risk_grade="A")
    builder.add_stage(StageAssessment(stage="data_quality", score=0.96, passed=True, weight=0.15))
    builder.add_stage(StageAssessment(stage="market_regime", score=confidence, passed=True, weight=0.20))
    builder.add_stage(StageAssessment(stage="model_reliability", score=confidence, passed=True, weight=0.25))
    builder.add_stage(StageAssessment(stage="risk_alignment", score=0.84, passed=True, weight=0.20))
    builder.add_stage(StageAssessment(stage="execution_feasibility", score=0.93, passed=True, weight=0.20))
    return builder.build()


def _build_rejected_contract(symbol: str = "ETH/USD") -> DecisionContract:
    """Helper: build a contract that results in REJECT (veto)."""
    builder = DecisionContractBuilder()
    builder.set_symbol(symbol, "BUY")
    builder.add_stage(StageAssessment(
        stage="data_quality", score=0.12, passed=False, weight=0.15,
        veto=True, veto_reason="Data feed interrupted",
        severity=StageSeverity.CRITICAL,
    ))
    builder.add_stage(StageAssessment(stage="model_reliability", score=0.92, passed=True, weight=0.20))
    return builder.build()


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Contract flows through TradeRequest to RiskEngine
# ──────────────────────────────────────────────────────────────────────────────


class TestContractFlowThroughRisk:
    """DecisionContract flows as single source of truth through risk evaluation."""

    def test_trade_request_carries_contract(self):
        """TradeRequest stores the DecisionContract."""
        contract = _build_execute_contract()
        request = TradeRequest(
            symbol="BTC/USD",
            side="buy",
            qty=0.5,
            price=67000.0,
            decision_contract=contract,
        )
        assert request.has_contract
        assert request.effective_confidence == contract.final_confidence
        assert request.decision_contract.contract_id == contract.contract_id

    def test_effective_confidence_priority(self):
        """DecisionContract confidence takes priority over scalar."""
        contract = _build_execute_contract(confidence=0.90)
        request = TradeRequest(
            symbol="BTC/USD",
            side="buy",
            qty=1.0,
            price=100.0,
            confidence=0.60,  # Lower scalar
            decision_contract=contract,
        )
        # Contract confidence should win
        assert request.effective_confidence > 0.85

    def test_risk_engine_rejects_non_executable_contract(self):
        """RiskEngine rejects when contract is not executable."""
        from src.risk.engine import RiskEngine

        engine = RiskEngine()
        rejected_contract = _build_rejected_contract()

        request = TradeRequest(
            symbol="ETH/USD",
            side="buy",
            qty=1.0,
            price=3500.0,
            decision_contract=rejected_contract,
        )

        decision = engine.evaluate(
            request=request,
            portfolio_value=100000.0,
            cash=50000.0,
        )

        assert not decision.approved
        assert "Decision contract rejected" in decision.reasons[0]

    def test_risk_engine_accepts_executable_contract(self):
        """RiskEngine proceeds with evaluation for executable contracts."""
        from src.risk.engine import RiskEngine

        engine = RiskEngine()
        contract = _build_execute_contract()

        request = TradeRequest(
            symbol="BTC/USD",
            side="buy",
            qty=0.5,
            price=67000.0,
            stop_loss=65000.0,
            decision_contract=contract,
        )

        decision = engine.evaluate(
            request=request,
            portfolio_value=100000.0,
            cash=80000.0,
        )

        # Should pass confidence check (contract is executable)
        # May still be rejected by other risk layers (position size, etc.)
        # but it should NOT be rejected for confidence reasons
        if not decision.approved:
            for reason in decision.reasons:
                assert "confidence" not in reason.lower()
                assert "Decision contract rejected" not in reason


# ──────────────────────────────────────────────────────────────────────────────
# Tests — ContractStore Persistence
# ──────────────────────────────────────────────────────────────────────────────


class TestContractStorePersistence:
    """ContractStore persists all contracts for audit trail."""

    def test_store_and_retrieve(self, temp_store):
        contract = _build_execute_contract()
        temp_store.store(contract)

        replay = temp_store.replay(contract.contract_id)
        assert replay is not None
        assert replay["contract_id"] == contract.contract_id
        assert replay["decision"] == "execute"
        assert abs(replay["final_confidence"] - contract.final_confidence) < 0.001

    def test_store_rejected_contract(self, temp_store):
        contract = _build_rejected_contract()
        temp_store.store(contract)

        replay = temp_store.replay(contract.contract_id)
        assert replay is not None
        assert replay["decision"] == "reject"
        assert replay["vetoed"] == True

    def test_mark_executed_updates_status(self, temp_store):
        contract = _build_execute_contract()
        temp_store.store(contract)
        temp_store.mark_executed(contract.contract_id, "T-001")

        replay = temp_store.replay(contract.contract_id)
        assert replay["_replay_metadata"]["execution_status"] == "executed"
        assert replay["_replay_metadata"]["trade_id"] == "T-001"

    def test_record_outcome(self, temp_store):
        contract = _build_execute_contract()
        temp_store.store(contract)
        temp_store.mark_executed(contract.contract_id, "T-002")
        temp_store.record_outcome(
            contract_id=contract.contract_id,
            trade_id="T-002",
            symbol="BTC/USD",
            side="buy",
            entry_price=67000.0,
            exit_price=68500.0,
            pnl=750.0,
            pnl_pct=2.24,
            holding_minutes=120,
            exit_reason="take_profit",
        )

        replay = temp_store.replay(contract.contract_id)
        assert "_outcome" in replay
        assert replay["_outcome"]["pnl_pct"] == 2.24
        assert replay["_outcome"]["actual_profitable"] == True
        assert replay["_outcome"]["exit_reason"] == "take_profit"

    def test_stores_stage_details(self, temp_store):
        contract = _build_execute_contract()
        temp_store.store(contract)

        replay = temp_store.replay(contract.contract_id)
        assert "_stage_details" in replay
        assert len(replay["_stage_details"]) == 5
        stage_names = [s["stage"] for s in replay["_stage_details"]]
        assert "data_quality" in stage_names
        assert "model_reliability" in stage_names


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Replay Capability
# ──────────────────────────────────────────────────────────────────────────────


class TestReplayCapability:
    """Given a contract_id, reconstruct the full decision context."""

    def test_replay_by_contract_id(self, temp_store):
        contract = _build_execute_contract("SOL/USD")
        temp_store.store(contract)

        replay = temp_store.replay(contract.contract_id)
        assert replay["symbol"] == "SOL/USD"
        assert replay["direction"] == "BUY"
        assert "stages" in replay
        assert "provenance" in replay
        assert replay["provenance"]["model_version"] == "v14.3"

    def test_replay_by_trade_id(self, temp_store):
        contract = _build_execute_contract("AAPL")
        temp_store.store(contract)
        temp_store.mark_executed(contract.contract_id, "T-AAPL-001")

        replay = temp_store.replay_by_trade("T-AAPL-001")
        assert replay is not None
        assert replay["contract_id"] == contract.contract_id
        assert replay["symbol"] == "AAPL"

    def test_replay_nonexistent_returns_none(self, temp_store):
        assert temp_store.replay("DC-NONEXISTENT") is None
        assert temp_store.replay_by_trade("T-NONEXISTENT") is None

    def test_replay_includes_full_provenance(self, temp_store):
        builder = DecisionContractBuilder()
        builder.set_symbol("NVDA", "BUY")
        builder.set_provenance(
            model_version="v15.0",
            feature_set_version="fs_99",
            dataset_version="ds_2026_07",
            walk_forward_passed=True,
            monte_carlo_passed=True,
            model_health_score=92.0,
        )
        builder.add_stage(StageAssessment(stage="x", score=0.9, passed=True, weight=1.0))
        contract = builder.build()
        temp_store.store(contract)

        replay = temp_store.replay(contract.contract_id)
        prov = replay["provenance"]
        assert prov["model_version"] == "v15.0"
        assert prov["walk_forward_passed"] == True


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Analytics and Querying
# ──────────────────────────────────────────────────────────────────────────────


class TestContractAnalytics:
    """Analytics queries for model governance."""

    def test_query_by_symbol(self, temp_store):
        temp_store.store(_build_execute_contract("BTC/USD"))
        temp_store.store(_build_execute_contract("ETH/USD"))
        temp_store.store(_build_execute_contract("BTC/USD"))

        results = temp_store.query_contracts(symbol="BTC/USD")
        assert len(results) == 2
        assert all(r["symbol"] == "BTC/USD" for r in results)

    def test_query_by_decision(self, temp_store):
        temp_store.store(_build_execute_contract())
        temp_store.store(_build_rejected_contract())

        executed = temp_store.query_contracts(decision="execute")
        assert len(executed) == 1
        rejected = temp_store.query_contracts(decision="reject")
        assert len(rejected) == 1

    def test_accuracy_calculation(self, temp_store):
        # Store 3 executed contracts with outcomes
        for i, pnl in enumerate([2.5, -1.0, 3.0]):
            c = _build_execute_contract(f"SYM{i}")
            temp_store.store(c)
            temp_store.mark_executed(c.contract_id, f"T-{i}")
            temp_store.record_outcome(
                contract_id=c.contract_id,
                trade_id=f"T-{i}",
                symbol=f"SYM{i}",
                side="buy",
                entry_price=100.0,
                exit_price=100.0 + pnl,
                pnl=pnl * 10,
                pnl_pct=pnl,
            )

        acc = temp_store.get_contract_accuracy()
        assert acc["total"] == 3
        assert acc["wins"] == 2  # 2 profitable, 1 loss
        assert abs(acc["win_rate"] - 2/3) < 0.01

    def test_stage_diagnostics(self, temp_store):
        temp_store.store(_build_execute_contract())
        temp_store.store(_build_rejected_contract())

        diag = temp_store.get_stage_diagnostics()
        assert len(diag) > 0

        # data_quality should appear (used in both contracts)
        dq = next((d for d in diag if d["stage"] == "data_quality"), None)
        assert dq is not None
        assert dq["evaluations"] == 2
        assert dq["veto_count"] == 1  # One veto from rejected contract

    def test_count(self, temp_store):
        temp_store.store(_build_execute_contract())
        temp_store.store(_build_rejected_contract())

        counts = temp_store.count()
        assert counts["total"] == 2
        assert counts["rejected"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Contract Immutability Through Pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestContractImmutabilityThroughPipeline:
    """Contract cannot be mutated at any point in the pipeline."""

    def test_contract_survives_trade_request(self):
        """Contract stored in TradeRequest remains unchanged."""
        contract = _build_execute_contract()
        original_confidence = contract.final_confidence
        original_id = contract.contract_id

        request = TradeRequest(
            symbol="BTC/USD",
            side="buy",
            qty=1.0,
            price=100.0,
            decision_contract=contract,
        )

        # Try modifying through request (should not affect contract)
        request.confidence = 0.1
        assert request.decision_contract.final_confidence == original_confidence
        assert request.decision_contract.contract_id == original_id

    def test_serialized_contract_matches_original(self, temp_store):
        """Contract serialized to store matches original."""
        contract = _build_execute_contract()
        temp_store.store(contract)

        replay = temp_store.replay(contract.contract_id)
        assert replay["final_confidence"] == contract.final_confidence
        assert replay["integrity_hash"] == contract.integrity_hash
        assert replay["decision"] == contract.decision.value


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Contract Expiry
# ──────────────────────────────────────────────────────────────────────────────


class TestContractExpiry:
    """Expired contracts cannot authorize execution."""

    def test_expired_contract_rejected_by_risk(self):
        """RiskEngine rejects expired contracts."""
        from src.risk.engine import RiskEngine

        # Build a contract that expires immediately
        builder = DecisionContractBuilder(validity_minutes=0.0)
        builder.set_symbol("BTC/USD", "BUY")
        builder.add_stage(StageAssessment(stage="x", score=0.95, passed=True, weight=1.0))
        contract = builder.build()
        time.sleep(0.01)

        assert contract.is_expired
        assert not contract.is_executable

        engine = RiskEngine()
        request = TradeRequest(
            symbol="BTC/USD",
            side="buy",
            qty=1.0,
            price=67000.0,
            decision_contract=contract,
        )

        decision = engine.evaluate(request=request, portfolio_value=100000.0, cash=50000.0)
        assert not decision.approved
        assert "Decision contract rejected" in decision.reasons[0]
