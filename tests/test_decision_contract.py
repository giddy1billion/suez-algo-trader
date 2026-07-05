"""
Tests for the Decision Contract system.

Validates:
1. Builder produces frozen (immutable) contracts
2. Veto authority halts execution regardless of other scores
3. Confidence decay / expiry
4. Weighted scoring
5. Serialization round-trip
6. Orchestrator integration (gate → contract)
7. Stage assessments carry evidence
"""
import time
from datetime import datetime, timezone, timedelta

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


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _healthy_builder() -> DecisionContractBuilder:
    """A builder with typical healthy stages (all passing)."""
    builder = DecisionContractBuilder(validity_minutes=5.0)
    builder.set_symbol("BTC/USD", "BUY")
    builder.set_provenance(
        model_version="v14.3",
        feature_set_version="fs_82",
        dataset_version="ds_2026_07_05",
        walk_forward_passed=True,
        model_health_score=89.0,
    )
    builder.set_sizing(position_pct=3.2, kelly=0.08, risk_grade="A")

    builder.add_stage(StageAssessment(
        stage="data_quality", score=0.96, passed=True, weight=0.15,
        evidence={"completeness": 0.99, "candles": 500},
    ))
    builder.add_stage(StageAssessment(
        stage="market_regime", score=0.88, passed=True, weight=0.18,
        severity=StageSeverity.LOW,
        evidence={"regime": "TRENDING", "adx": 32},
        warnings=["Volatility elevated"],
    ))
    builder.add_stage(StageAssessment(
        stage="model_reliability", score=0.79, passed=True, weight=0.20,
        evidence={"accuracy": 0.72},
    ))
    builder.add_stage(StageAssessment(
        stage="risk_alignment", score=0.84, passed=True, weight=0.17,
        evidence={"exposure": 0.45},
    ))
    builder.add_stage(StageAssessment(
        stage="execution_feasibility", score=0.93, passed=True, weight=0.15,
        evidence={"spread_bps": 3.2},
    ))
    builder.add_stage(StageAssessment(
        stage="historical_similarity", score=0.81, passed=True, weight=0.15,
        evidence={"winrate": 0.68},
    ))
    return builder


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Contract Builder
# ──────────────────────────────────────────────────────────────────────────────


class TestDecisionContractBuilder:
    """Builder produces valid, frozen contracts."""

    def test_build_healthy_contract(self):
        builder = _healthy_builder()
        contract = builder.build()

        assert contract.decision == Decision.EXECUTE
        assert contract.is_executable
        assert 0.8 < contract.final_confidence < 0.9
        assert contract.symbol == "BTC/USD"
        assert contract.direction == "BUY"
        assert contract.recommended_position_pct == 3.2
        assert contract.risk_grade == "A"
        assert len(contract.stages) == 6

    def test_contract_id_format(self):
        contract = _healthy_builder().build()
        assert contract.contract_id.startswith("DC-")
        assert len(contract.contract_id) == 15  # "DC-" + 12 hex chars

    def test_builder_consumed_after_build(self):
        builder = _healthy_builder()
        builder.build()
        with pytest.raises(RuntimeError, match="already consumed"):
            builder.build()

    def test_builder_reset_allows_rebuild(self):
        builder = _healthy_builder()
        c1 = builder.build()
        builder.reset()
        builder.set_symbol("ETH/USD", "SELL")
        builder.add_stage(StageAssessment(stage="data_quality", score=0.90, passed=True, weight=1.0))
        c2 = builder.build()
        assert c1.contract_id != c2.contract_id
        assert c2.symbol == "ETH/USD"


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Immutability
# ──────────────────────────────────────────────────────────────────────────────


class TestImmutability:
    """DecisionContract is frozen — cannot be mutated."""

    def test_cannot_mutate_confidence(self):
        contract = _healthy_builder().build()
        with pytest.raises(Exception):  # FrozenInstanceError
            contract.final_confidence = 0.0

    def test_cannot_mutate_decision(self):
        contract = _healthy_builder().build()
        with pytest.raises(Exception):
            contract.decision = Decision.REJECT

    def test_cannot_mutate_symbol(self):
        contract = _healthy_builder().build()
        with pytest.raises(Exception):
            contract.symbol = "HACKED"

    def test_stage_assessment_frozen(self):
        stage = StageAssessment(stage="test", score=0.5, passed=True)
        with pytest.raises(Exception):
            stage.score = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Veto Authority
# ──────────────────────────────────────────────────────────────────────────────


class TestVetoAuthority:
    """Any stage with veto=True halts execution unconditionally."""

    def test_single_veto_rejects(self):
        builder = DecisionContractBuilder()
        builder.set_symbol("ETH/USD", "BUY")
        builder.add_stage(StageAssessment(
            stage="data_quality", score=0.12, passed=False, weight=0.15,
            severity=StageSeverity.CRITICAL,
            veto=True,
            veto_reason="Data feed interrupted",
            blockers=["Missing candles"],
        ))
        # Other stages are fine
        builder.add_stage(StageAssessment(
            stage="model_reliability", score=0.95, passed=True, weight=0.20,
        ))
        builder.add_stage(StageAssessment(
            stage="execution_feasibility", score=0.99, passed=True, weight=0.15,
        ))
        contract = builder.build()

        assert contract.decision == Decision.REJECT
        assert contract.vetoed is True
        assert contract.vetoed_by == VetoAuthority.DATA_QUALITY
        assert "interrupted" in contract.veto_reason
        assert not contract.is_executable

    def test_veto_overrides_high_scores(self):
        """Even if average confidence is 95%, a veto rejects."""
        builder = DecisionContractBuilder()
        builder.set_symbol("SOL/USD", "BUY")
        # 5 stages all scoring 0.95+
        for name in ["market_regime", "model_reliability", "risk_alignment", "execution_feasibility", "historical"]:
            builder.add_stage(StageAssessment(stage=name, score=0.95, passed=True, weight=0.18))
        # One veto
        builder.add_stage(StageAssessment(
            stage="data_quality", score=0.50, passed=False, weight=0.10,
            veto=True, veto_reason="Stale data",
        ))
        contract = builder.build()
        assert contract.decision == Decision.REJECT
        assert contract.vetoed
        assert contract.final_confidence == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Confidence Scoring
# ──────────────────────────────────────────────────────────────────────────────


class TestConfidenceScoring:
    """Weighted average scoring and decision thresholds."""

    def test_weighted_average(self):
        builder = DecisionContractBuilder()
        builder.set_symbol("AAPL", "BUY")
        builder.add_stage(StageAssessment(stage="a", score=1.0, passed=True, weight=0.7))
        builder.add_stage(StageAssessment(stage="b", score=0.0, passed=True, weight=0.3))
        contract = builder.build()
        assert abs(contract.final_confidence - 0.70) < 0.01

    def test_low_confidence_rejects(self):
        builder = DecisionContractBuilder()
        builder.set_symbol("MSFT", "BUY")
        builder.add_stage(StageAssessment(stage="a", score=0.3, passed=True, weight=0.5))
        builder.add_stage(StageAssessment(stage="b", score=0.4, passed=True, weight=0.5))
        contract = builder.build()
        assert contract.decision == Decision.REJECT
        assert not contract.is_executable

    def test_medium_confidence_reduces(self):
        builder = DecisionContractBuilder()
        builder.set_symbol("NVDA", "BUY")
        builder.add_stage(StageAssessment(stage="a", score=0.65, passed=True, weight=0.5))
        builder.add_stage(StageAssessment(stage="b", score=0.70, passed=True, weight=0.5))
        contract = builder.build()
        assert contract.decision == Decision.REDUCE

    def test_failing_stage_with_high_confidence_reduces(self):
        builder = DecisionContractBuilder()
        builder.set_symbol("GOOGL", "BUY")
        builder.add_stage(StageAssessment(stage="a", score=0.90, passed=True, weight=0.7))
        builder.add_stage(StageAssessment(stage="b", score=0.50, passed=False, weight=0.3))
        contract = builder.build()
        # 0.9*0.7 + 0.5*0.3 = 0.78 → above 0.65 threshold → REDUCE (not REJECT)
        assert contract.decision == Decision.REDUCE


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Expiry / Decay
# ──────────────────────────────────────────────────────────────────────────────


class TestExpiry:
    """Contracts expire after their validity window."""

    def test_fresh_contract_not_expired(self):
        contract = _healthy_builder().build()
        assert not contract.is_expired

    def test_expired_contract_not_executable(self):
        builder = DecisionContractBuilder(validity_minutes=0.0)  # expires immediately
        builder.set_symbol("AAPL", "BUY")
        builder.add_stage(StageAssessment(stage="x", score=0.95, passed=True, weight=1.0))
        contract = builder.build()
        # Wait a tiny bit for expiry
        time.sleep(0.01)
        assert contract.is_expired
        assert not contract.is_executable

    def test_age_seconds(self):
        contract = _healthy_builder().build()
        time.sleep(0.05)
        assert contract.age_seconds >= 0.04


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Provenance
# ──────────────────────────────────────────────────────────────────────────────


class TestProvenance:
    """Full reproducibility metadata."""

    def test_provenance_attached(self):
        contract = _healthy_builder().build()
        prov = contract.provenance
        assert prov.model_version == "v14.3"
        assert prov.feature_set_version == "fs_82"
        assert prov.dataset_version == "ds_2026_07_05"
        assert prov.walk_forward_passed is True
        assert prov.model_health_score == 89.0

    def test_provenance_in_serialization(self):
        contract = _healthy_builder().build()
        d = contract.to_dict()
        assert d["provenance"]["model_version"] == "v14.3"


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Serialization
# ──────────────────────────────────────────────────────────────────────────────


class TestSerialization:
    """to_dict() produces complete audit records."""

    def test_to_dict_complete(self):
        contract = _healthy_builder().build()
        d = contract.to_dict()
        assert d["contract_id"].startswith("DC-")
        assert d["decision"] == "execute"
        assert 0.8 < d["final_confidence"] < 0.9
        assert d["symbol"] == "BTC/USD"
        assert len(d["stages"]) == 6
        assert d["integrity_hash"]

    def test_to_dict_stages_have_evidence(self):
        contract = _healthy_builder().build()
        d = contract.to_dict()
        stage0 = d["stages"][0]
        assert stage0["stage"] == "data_quality"
        assert stage0["evidence"]["completeness"] == 0.99

    def test_to_audit_dict_same_as_to_dict(self):
        contract = _healthy_builder().build()
        assert contract.to_dict() == contract.to_audit_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Integrity Hash
# ──────────────────────────────────────────────────────────────────────────────


class TestIntegrityHash:
    """Hash provides tamper detection."""

    def test_hash_present(self):
        contract = _healthy_builder().build()
        assert contract.integrity_hash
        assert len(contract.integrity_hash) == 16  # SHA256[:16]

    def test_different_contracts_different_hashes(self):
        c1 = _healthy_builder().build()
        builder2 = _healthy_builder()
        builder2._built = False  # Reset
        builder2 = DecisionContractBuilder()
        builder2.set_symbol("ETH/USD", "SELL")
        builder2.add_stage(StageAssessment(stage="x", score=0.5, passed=True, weight=1.0))
        c2 = builder2.build()
        assert c1.integrity_hash != c2.integrity_hash


# ──────────────────────────────────────────────────────────────────────────────
# Tests — Properties
# ──────────────────────────────────────────────────────────────────────────────


class TestProperties:
    """Computed properties aggregate stage data."""

    def test_stage_scores(self):
        contract = _healthy_builder().build()
        scores = contract.stage_scores
        assert scores["data_quality"] == 0.96
        assert scores["model_reliability"] == 0.79

    def test_warnings_aggregation(self):
        contract = _healthy_builder().build()
        assert len(contract.warnings) == 1
        assert "Volatility elevated" in contract.warnings[0]

    def test_blockers_empty_when_healthy(self):
        contract = _healthy_builder().build()
        assert contract.blockers == []

    def test_blockers_present_when_veto(self):
        builder = DecisionContractBuilder()
        builder.set_symbol("X", "BUY")
        builder.add_stage(StageAssessment(
            stage="data_quality", score=0.1, passed=False, weight=0.15,
            veto=True, blockers=["Feed down"],
        ))
        contract = builder.build()
        assert "Feed down" in contract.blockers[0]
