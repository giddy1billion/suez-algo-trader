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



# ???????????????????????????????????????????????????????????????????????????????
# System-Wide Contract Integration Tests
# ???????????????????????????????????????????????????????????????????????????????


class TestContractIdFlowsSystemWide:
    """Verify contract_id flows through ALL system components end-to-end."""

    def test_trade_opened_event_carries_contract_id(self):
        """TradeOpened event carries contract_id field."""
        from src.core.events import TradeOpened

        event = TradeOpened(
            trade_id="T-001",
            symbol="BTC/USD",
            side="BUY",
            entry_price=67000.0,
            qty=0.1,
            contract_id="DC-abc123",
        )
        assert event.contract_id == "DC-abc123"
        # Verify serializable
        d = event.to_dict()
        assert d["contract_id"] == "DC-abc123"

    def test_trade_closed_event_carries_contract_id(self):
        """TradeClosed event carries contract_id for feedback loop."""
        from src.core.events import TradeClosed

        event = TradeClosed(
            trade_id="T-001",
            symbol="BTC/USD",
            exit_price=68000.0,
            pnl=100.0,
            pnl_pct=1.49,
            reason="take_profit",
            contract_id="DC-abc123",
        )
        assert event.contract_id == "DC-abc123"
        d = event.to_dict()
        assert d["contract_id"] == "DC-abc123"

    def test_risk_evaluated_event_carries_contract_id(self):
        """RiskEvaluated event carries contract_id."""
        from src.core.events import RiskEvaluated

        event = RiskEvaluated(
            symbol="BTC/USD",
            approved=True,
            risk_score=0.85,
            contract_id="DC-xyz789",
        )
        assert event.contract_id == "DC-xyz789"

    def test_trade_scorecard_carries_contract_fields(self):
        """TradeScorecard has contract_id, contract_decision, contract_confidence."""
        from src.ml.feedback_loop import TradeScorecard
        from datetime import datetime, timezone

        scorecard = TradeScorecard(
            trade_id="T-001",
            signal_id="SIG-001",
            symbol="ETH/USD",
            timestamp=datetime.now(timezone.utc),
            predicted_direction="buy",
            actual_profitable=True,
            predicted_confidence=0.82,
            predicted_win_probability=0.75,
            predicted_return_pct=3.0,
            actual_return_pct=2.5,
            predicted_risk_reward=2.0,
            actual_risk_reward=1.8,
            predicted_duration_minutes=120,
            actual_duration_minutes=90,
            entry_price=3500.0,
            exit_price=3587.5,
            stop_loss_price=3400.0,
            stop_loss_hit=False,
            take_profit_prices=[3600.0],
            take_profit_reached=[False],
            max_favorable_excursion=3600.0,
            max_adverse_excursion=3480.0,
            model_version="v14.3",
            strategy_name="momentum",
            market_regime="trending",
            volatility_level="medium",
            contract_id="DC-eth001",
            contract_decision="execute",
            contract_confidence=0.82,
        )
        assert scorecard.contract_id == "DC-eth001"
        assert scorecard.contract_decision == "execute"
        assert scorecard.contract_confidence == 0.82

    def test_audit_trail_carries_contract_id(self):
        """TradeAuditTrail has contract_id field for full traceability."""
        from src.core.audit_log import TradeAuditTrail

        trail = TradeAuditTrail(
            trade_id="T-001",
            signal_id="SIG-001",
            contract_id="DC-audit001",
            prediction_id="PRED-001",
            model_version="v14.3",
        )
        assert trail.contract_id == "DC-audit001"
        d = trail.to_dict()
        assert d["contract_id"] == "DC-audit001"
        assert trail.is_complete()

    def test_journal_entry_model_has_contract_id(self):
        """JournalEntry SQLAlchemy model has contract_id column."""
        from src.data.store import JournalEntry

        assert hasattr(JournalEntry, "contract_id")

    def test_trade_model_has_contract_id(self):
        """Trade SQLAlchemy model has contract_id column."""
        from src.data.store import Trade

        assert hasattr(Trade, "contract_id")

    def test_experience_db_stores_contract_id(self):
        """ExperienceDatabase tables include contract_id columns."""
        import tempfile, shutil
        from src.ml.feedback_loop import ExperienceDatabase, TradeScorecard
        from datetime import datetime, timezone

        tmp = tempfile.mkdtemp()
        try:
            db = ExperienceDatabase(storage_path=tmp)
            # Record a trade with contract_id
            scorecard = TradeScorecard(
                trade_id="T-exp001",
                signal_id="SIG-exp001",
                symbol="SOL/USD",
                timestamp=datetime.now(timezone.utc),
                predicted_direction="buy",
                actual_profitable=True,
                predicted_confidence=0.78,
                predicted_win_probability=0.70,
                predicted_return_pct=5.0,
                actual_return_pct=4.2,
                predicted_risk_reward=2.5,
                actual_risk_reward=2.1,
                predicted_duration_minutes=60,
                actual_duration_minutes=45,
                entry_price=150.0,
                exit_price=156.3,
                stop_loss_price=145.0,
                stop_loss_hit=False,
                take_profit_prices=[160.0],
                take_profit_reached=[False],
                max_favorable_excursion=157.0,
                max_adverse_excursion=148.5,
                model_version="v2.1",
                strategy_name="momentum",
                market_regime="trending",
                volatility_level="high",
                contract_id="DC-sol-test",
                contract_decision="execute",
                contract_confidence=0.78,
            )
            db.record_trade(scorecard)

            # Verify stored in predictions table
            result = db._conn.execute(
                "SELECT contract_id FROM predictions WHERE trade_id = ?",
                ["T-exp001"]
            ).fetchone()
            assert result[0] == "DC-sol-test"

            # Verify stored in outcomes table
            result = db._conn.execute(
                "SELECT contract_id, contract_decision, contract_confidence FROM outcomes WHERE trade_id = ?",
                ["T-exp001"]
            ).fetchone()
            assert result[0] == "DC-sol-test"
            assert result[1] == "execute"
            assert result[2] == 0.78

            # Verify stored in scorecards table
            result = db._conn.execute(
                "SELECT contract_id FROM scorecards WHERE trade_id = ?",
                ["T-exp001"]
            ).fetchone()
            assert result[0] == "DC-sol-test"

            db._conn.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_notification_includes_contract_id(self):
        """NotificationManager.notify_trade() includes contract_id in message."""
        pytest.importorskip("httpx")
        from src.notifications.alerts import NotificationManager
        from unittest.mock import patch

        mgr = NotificationManager(notify_trades=True)
        sent_messages = []

        with patch.object(mgr, "_send", side_effect=lambda msg: sent_messages.append(msg)):
            mgr.notify_trade({
                "symbol": "BTC/USD",
                "side": "BUY",
                "qty": 0.5,
                "price": 67000.0,
                "signal_confidence": 0.82,
                "contract_id": "DC-notif-test-12345678",
            })

        assert len(sent_messages) == 1
        assert "DC-notif-test" in sent_messages[0]

    def test_backward_compat_no_contract_id(self):
        """All components work fine without contract_id (backward compatible)."""
        from src.core.events import TradeOpened, TradeClosed, RiskEvaluated
        from src.core.audit_log import TradeAuditTrail

        # Events work without contract_id
        opened = TradeOpened(trade_id="T-001", symbol="AAPL", side="BUY")
        assert opened.contract_id == ""

        closed = TradeClosed(trade_id="T-001", symbol="AAPL")
        assert closed.contract_id == ""

        risk = RiskEvaluated(symbol="AAPL", approved=True)
        assert risk.contract_id == ""

        # AuditTrail works without contract_id
        trail = TradeAuditTrail(trade_id="T-001")
        assert trail.contract_id == ""

    def test_backward_compat_notifications_no_contract(self):
        """Notifications work without contract_id."""
        pytest.importorskip("httpx")
        from src.notifications.alerts import NotificationManager
        from unittest.mock import patch

        mgr = NotificationManager(notify_trades=True)
        sent = []
        with patch.object(mgr, "_send", side_effect=lambda msg: sent.append(msg)):
            mgr.notify_trade({"symbol": "AAPL", "side": "BUY", "qty": 10, "price": 150.0, "signal_confidence": 0.7})
        assert len(sent) == 1
        assert "Contract" not in sent[0]  # No contract line when absent
