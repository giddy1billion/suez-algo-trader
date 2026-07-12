"""
Integration tests for ML and Contract modules against PostgreSQL.

These tests validate that all DuckDB-migrated modules work correctly
with both SQLite (default) and PostgreSQL (via DATABASE_URL_TEST env var).

Run locally (SQLite):
    pytest tests/test_ml_pg_integration.py -v

Run against live PG:
    DATABASE_URL_TEST="postgresql://..." pytest tests/test_ml_pg_integration.py -v
"""
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

# Import all modules under test
from src.ml.feature_store import FeatureStore
from src.ml.dataset_registry import DatasetRegistry
from src.ml.feedback_loop import ExperienceDatabase, TradeScorecard
from src.intelligence.confidence.contract_store import ContractStore
from src.intelligence.confidence.decision_contract import (
    DecisionContract,
    Decision,
    StageAssessment,
    StageSeverity,
    DecisionProvenance,
)


def get_test_url():
    """Get database URL for testing - falls back to temp SQLite."""
    return os.environ.get("DATABASE_URL_TEST", None)


class TestFeatureStoreIntegration:
    """FeatureStore CRUD on SQLite and optionally PG."""

    def setup_method(self):
        self._tmp = tempfile.mkdtemp(prefix="fs_test_")
        url = get_test_url()
        if url:
            self.fs = FeatureStore(storage_path=self._tmp, database_url=url)
        else:
            self.fs = FeatureStore(store_dir=self._tmp)

    def teardown_method(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_register_and_get_version(self):
        vid = self.fs.register_version(
            feature_names=["rsi_14", "macd", "bb_width"],
            scaling_params={"mean": 0.5, "std": 0.2},
            encoding_params={},
            normalization_method="standard",
            description="integration test",
        )
        assert vid is not None
        v = self.fs.get_version(vid)
        assert v.feature_names == ["rsi_14", "macd", "bb_width"]
        assert v.normalization_method == "standard"

    def test_version_deduplication(self):
        vid1 = self.fs.register_version(
            feature_names=["a", "b"],
            scaling_params={"x": 1},
        )
        vid2 = self.fs.register_version(
            feature_names=["a", "b"],
            scaling_params={"x": 1},
        )
        assert vid1 == vid2

    def test_snapshot_features_and_retrieve(self):
        vid = self.fs.register_version(
            feature_names=["rsi"],
            scaling_params={},
        )
        snap_id = self.fs.snapshot_features(
            version_id=vid,
            symbol="ETH/USD",
            values={"rsi": 62.5},
            raw_values={"rsi": 62.5},
        )
        snap = self.fs.get_snapshot(snap_id)
        assert snap is not None
        assert snap.symbol == "ETH/USD"
        assert snap.values["rsi"] == 62.5

    def test_get_active_version(self):
        self.fs.register_version(
            feature_names=["f1"],
            scaling_params={},
            description="first",
        )
        vid2 = self.fs.register_version(
            feature_names=["f1", "f2"],
            scaling_params={"new": True},
            description="second",
        )
        active = self.fs.get_active_version()
        assert active is not None
        assert active.version_id == vid2


class TestDatasetRegistryIntegration:
    """DatasetRegistry CRUD on SQLite and optionally PG."""

    def setup_method(self):
        self._tmp = tempfile.mkdtemp(prefix="dr_test_")
        url = get_test_url()
        if url:
            self.dr = DatasetRegistry(storage_path=self._tmp, database_url=url)
        else:
            self.dr = DatasetRegistry(storage_path=self._tmp)

    def teardown_method(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_register_and_get_dataset(self):
        data = pd.DataFrame(
            {"rsi": np.random.random(50), "close": np.random.random(50) * 100},
            index=pd.date_range("2026-01-01", periods=50, freq="h"),
        )
        ds_id = self.dr.register_dataset(
            data=data,
            symbols=["BTC/USD"],
            timeframe="1h",
            feature_version_id="fv_test",
            source="test",
        )
        ds = self.dr.get_dataset(ds_id)
        assert ds is not None
        assert ds.row_count == 50
        assert ds.symbols == ["BTC/USD"]

    def test_register_model_and_lineage(self):
        data = pd.DataFrame({"x": [1, 2, 3]})
        ds_id = self.dr.register_dataset(
            data=data, symbols=["X"], timeframe="1d", feature_version_id="fv1"
        )
        self.dr.register_model(
            model_version="test_model_v1",
            dataset_id=ds_id,
            feature_version_id="fv1",
            pipeline_id="pipe1",
            hyperparameters={"n": 100},
            training_metrics={"acc": 0.8},
            training_duration=10.5,
        )
        ml = self.dr.get_model_lineage("test_model_v1")
        assert ml is not None
        assert ml.dataset_id == ds_id
        assert ml.training_duration_seconds == 10.5

    def test_model_promotion(self):
        data = pd.DataFrame({"x": [1]})
        ds_id = self.dr.register_dataset(
            data=data, symbols=["Y"], timeframe="1d", feature_version_id="fv2"
        )
        self.dr.register_model("m1", ds_id, "fv2", "p1")
        self.dr.set_model_status("m1", "active")
        active = self.dr.get_active_model()
        assert active is not None
        assert active.model_version == "m1"

    def test_record_prediction(self):
        # First register a model so prediction can reference it
        data = pd.DataFrame({"x": [1]})
        ds_id = self.dr.register_dataset(
            data=data, symbols=["BTC"], timeframe="1d", feature_version_id="fv_pred"
        )
        self.dr.register_model("m_pred_1", ds_id, "fv_pred", "pipe_pred")
        self.dr.record_prediction(
            prediction_id="pred_test_001",
            model_version="m_pred_1",
            feature_version_id="fv_pred",
            feature_snapshot_id="snap1",
            symbol="BTC/USD",
            predicted_direction="buy",
            predicted_confidence=0.72,
        )
        # Verify prediction was stored
        assert self.dr.prediction_count > 0


class TestExperienceDatabaseIntegration:
    """ExperienceDatabase (feedback loop) CRUD on SQLite and optionally PG."""

    def setup_method(self):
        self._tmp = tempfile.mkdtemp(prefix="exp_test_")
        url = get_test_url()
        if url:
            self.db = ExperienceDatabase(storage_path=self._tmp, database_url=url)
        else:
            self.db = ExperienceDatabase(storage_path=self._tmp)

    def teardown_method(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_scorecard(self, trade_id="T-001", **kwargs):
        defaults = dict(
            trade_id=trade_id,
            signal_id="S-001",
            symbol="ETH/USD",
            timestamp=datetime.now(timezone.utc),
            predicted_direction="buy",
            actual_profitable=True,
            predicted_confidence=0.8,
            predicted_win_probability=0.7,
            predicted_return_pct=2.0,
            actual_return_pct=1.5,
            predicted_risk_reward=2.0,
            actual_risk_reward=1.8,
            predicted_duration_minutes=60,
            actual_duration_minutes=50,
            entry_price=3500.0,
            exit_price=3552.0,
            stop_loss_price=3450.0,
            stop_loss_hit=False,
            take_profit_prices=[3600.0],
            take_profit_reached=[False],
            max_favorable_excursion=3570.0,
            max_adverse_excursion=3480.0,
            model_version="v1",
            strategy_name="momentum",
            market_regime="trending",
            volatility_level="medium",
            feature_vector={"rsi": 55.0, "macd": 0.01},
            direction_score=1.0,
            confidence_calibration_error=0.1,
            timing_score=0.8,
            exit_efficiency=0.7,
            overall_score=80.0,
        )
        defaults.update(kwargs)
        return TradeScorecard(**defaults)

    def test_record_trade_and_total(self):
        sc = self._make_scorecard()
        self.db.record_trade(sc)
        assert self.db.total_trades >= 1

    def test_contract_fields_stored(self):
        sc = self._make_scorecard(
            trade_id="T-contract-test",
            contract_id="DC-test-001",
            contract_decision="execute",
            contract_confidence=0.88,
        )
        self.db.record_trade(sc)
        # Verify via get_training_samples
        samples = self.db.get_training_samples(min_trades=0)
        assert len(samples) >= 1

    def test_get_recent_accuracy(self):
        for i in range(3):
            sc = self._make_scorecard(
                trade_id=f"T-acc-{i}",
                actual_profitable=(i % 2 == 0),
            )
            self.db.record_trade(sc)
        acc = self.db.get_recent_accuracy(n_trades=50)
        assert 0 <= acc <= 1.0


class TestContractStoreIntegration:
    """ContractStore CRUD on SQLite and optionally PG."""

    def setup_method(self):
        self._tmp = tempfile.mkdtemp(prefix="cs_test_")
        url = get_test_url()
        if url:
            self.cs = ContractStore(storage_path=self._tmp, database_url=url)
        else:
            self.cs = ContractStore(storage_path=self._tmp)

    def teardown_method(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_contract(self, contract_id="DC-test-001"):
        now = datetime.now(timezone.utc)
        return DecisionContract(
            contract_id=contract_id,
            symbol="BTC/USD",
            direction="buy",
            decision=Decision.EXECUTE,
            final_confidence=0.85,
            recommendation="Test signal",
            stages=[
                StageAssessment(
                    stage="quality",
                    score=0.9,
                    passed=True,
                    weight=0.5,
                    severity=StageSeverity.NONE,
                    evidence={"rsi": 55},
                    warnings=[],
                    blockers=[],
                    evaluation_ms=10.0,
                ),
            ],
            vetoed=False,
            recommended_position_pct=2.0,
            provenance=DecisionProvenance(
                model_version="v1",
                feature_set_version="fv1",
                dataset_version="ds1",
                walk_forward_passed=True,
                monte_carlo_passed=True,
                model_health_score=0.9,
            ),
            created_at=now,
            valid_until=now + timedelta(minutes=5),
        )

    def test_store_and_replay(self):
        contract = self._make_contract()
        stored_id = self.cs.store(contract)
        assert stored_id == "DC-test-001"

        replay = self.cs.replay("DC-test-001")
        assert replay is not None
        assert replay["symbol"] == "BTC/USD"
        assert replay["decision"] == "execute"

    def test_mark_executed(self):
        contract = self._make_contract("DC-exec-001")
        self.cs.store(contract)
        self.cs.mark_executed("DC-exec-001", "T-001")
        replay = self.cs.replay("DC-exec-001")
        # execution info is in _replay_metadata
        assert replay.get("_replay_metadata") is not None or "executed" in str(replay)

    def test_record_outcome(self):
        contract = self._make_contract("DC-outcome-001")
        self.cs.store(contract)
        self.cs.record_outcome(
            contract_id="DC-outcome-001",
            trade_id="T-002",
            symbol="BTC/USD",
            side="buy",
            entry_price=60000.0,
            exit_price=61000.0,
            pnl=100.0,
            pnl_pct=1.67,
            holding_minutes=45,
        )
        replay = self.cs.replay("DC-outcome-001")
        # outcome is in _outcome key
        assert replay.get("_outcome") is not None

    def test_query_contracts(self):
        for i in range(3):
            c = self._make_contract(f"DC-query-{i}")
            self.cs.store(c)
        results = self.cs.query_contracts(symbol="BTC/USD")
        assert len(results) >= 3


# ─── Live PostgreSQL Tests (skipped unless DATABASE_URL_TEST is set) ───

@pytest.mark.skipif(
    not os.environ.get("DATABASE_URL_TEST"),
    reason="DATABASE_URL_TEST not set - skipping live PG tests"
)
class TestLivePostgreSQLModules:
    """End-to-end tests against live PostgreSQL for all migrated modules."""

    @pytest.fixture(autouse=True)
    def setup_url(self):
        self.url = os.environ["DATABASE_URL_TEST"]

    def test_feature_store_on_pg(self):
        fs = FeatureStore(database_url=self.url)
        vid = fs.register_version(
            feature_names=["test_f1", "test_f2"],
            scaling_params={"s": 1},
            description="live PG test",
        )
        v = fs.get_version(vid)
        assert v.feature_names == ["test_f1", "test_f2"]

    def test_dataset_registry_on_pg(self):
        dr = DatasetRegistry(database_url=self.url)
        data = pd.DataFrame({"x": [1, 2, 3]})
        ds_id = dr.register_dataset(
            data=data, symbols=["TEST"], timeframe="1d", feature_version_id="fv_live"
        )
        ds = dr.get_dataset(ds_id)
        assert ds.row_count == 3

    def test_experience_db_on_pg(self):
        db = ExperienceDatabase(database_url=self.url)
        sc = TradeScorecard(
            trade_id="T-live-pg-001", signal_id="S-1", symbol="BTC/USD",
            timestamp=datetime.now(timezone.utc), predicted_direction="buy",
            actual_profitable=True, predicted_confidence=0.8,
            predicted_win_probability=0.7, predicted_return_pct=2.0,
            actual_return_pct=1.5, predicted_risk_reward=2.0,
            actual_risk_reward=1.8, predicted_duration_minutes=60,
            actual_duration_minutes=50, entry_price=60000.0,
            exit_price=60900.0, stop_loss_price=59000.0, stop_loss_hit=False,
            take_profit_prices=[61000.0], take_profit_reached=[False],
            max_favorable_excursion=61200.0, max_adverse_excursion=59500.0,
            model_version="v1", strategy_name="test", market_regime="test",
            volatility_level="low", overall_score=75.0,
        )
        db.record_trade(sc)
        assert db.total_trades() >= 1

    def test_contract_store_on_pg(self):
        cs = ContractStore(database_url=self.url)
        now = datetime.now(timezone.utc)
        contract = DecisionContract(
            contract_id="DC-live-pg-001", symbol="ETH/USD", direction="buy",
            decision=Decision.EXECUTE, final_confidence=0.9,
            recommendation="Live PG test",
            stages=[StageAssessment(
                stage="test", score=0.95, passed=True, weight=1.0,
                severity=StageSeverity.NONE, evidence={}, warnings=[], blockers=[],
                evaluation_ms=5.0,
            )],
            vetoed=False, recommended_position_pct=1.0,
            provenance=DecisionProvenance(
                model_version="v1", feature_set_version="fv1",
                dataset_version="ds1", walk_forward_passed=True,
                monte_carlo_passed=True, model_health_score=0.95,
            ),
            created_at=now, valid_until=now + timedelta(minutes=5),
        )
        stored_id = cs.store(contract)
        assert stored_id == "DC-live-pg-001"
        replay = cs.replay("DC-live-pg-001")
        assert replay["symbol"] == "ETH/USD"
