"""
SQLAlchemy ORM models for ML subsystem tables.
"""
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Index
from sqlalchemy.orm import DeclarativeBase


class MLBase(DeclarativeBase):
    pass


# ─── Feature Store ────────────────────────────────────────────

class MLFeatureVersion(MLBase):
    __tablename__ = "ml_feature_versions"

    version_id = Column(String(64), primary_key=True)
    feature_names = Column(Text, nullable=False)  # JSON list
    feature_hash = Column(String(64), nullable=False, unique=True)
    scaling_params = Column(Text)  # JSON
    encoding_params = Column(Text)  # JSON
    normalization_method = Column(String(50), default="standard")
    description = Column(Text, default="")
    parent_version = Column(String(64))
    created_at = Column(DateTime)


class MLFeatureSnapshot(MLBase):
    __tablename__ = "ml_feature_snapshots"
    __table_args__ = (
        Index("idx_ml_snapshots_symbol_time", "symbol", "timestamp"),
    )

    snapshot_id = Column(String(64), primary_key=True)
    version_id = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    prediction_id = Column(String(64))
    timestamp = Column(DateTime, nullable=False)
    values_json = Column(Text, nullable=False)
    raw_values_json = Column(Text)
    created_at = Column(DateTime)


class MLTransformationLog(MLBase):
    __tablename__ = "ml_transformation_log"

    log_id = Column(String(64), primary_key=True)
    version_id = Column(String(64), nullable=False)
    operation = Column(String(100), nullable=False)
    params = Column(Text)  # JSON
    timestamp = Column(DateTime)


# ─── Dataset Registry ─────────────────────────────────────────

class MLDataset(MLBase):
    __tablename__ = "ml_datasets"

    dataset_id = Column(String(64), primary_key=True)
    version = Column(Integer, nullable=False)
    symbols = Column(Text, nullable=False)  # JSON list
    timeframe = Column(String(20), nullable=False)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    row_count = Column(Integer, nullable=False)
    feature_version_id = Column(String(64))
    data_hash = Column(String(64), nullable=False)
    source = Column(String(50), default="broker_historical")
    description = Column(Text, default="")
    parent_dataset_id = Column(String(64))
    parquet_path = Column(Text)
    created_at = Column(DateTime)


class MLModelLineage(MLBase):
    __tablename__ = "ml_model_lineage"
    __table_args__ = (
        Index("idx_ml_lineage_status", "status"),
    )

    model_version = Column(String(64), primary_key=True)
    dataset_id = Column(String(64), nullable=False)
    feature_version_id = Column(String(64), nullable=False)
    training_pipeline_id = Column(String(100))
    parent_model_version = Column(String(64))
    hyperparameters = Column(Text)  # JSON
    training_metrics = Column(Text)  # JSON
    training_duration_seconds = Column(Float, default=0.0)
    training_timestamp = Column(DateTime)
    promotion_timestamp = Column(DateTime)
    demotion_timestamp = Column(DateTime)
    status = Column(String(20), default="registered")


class MLPredictionRecord(MLBase):
    __tablename__ = "ml_prediction_records"
    __table_args__ = (
        Index("idx_ml_predictions_model", "model_version"),
        Index("idx_ml_predictions_trade", "trade_id"),
    )

    prediction_id = Column(String(64), primary_key=True)
    model_version = Column(String(64), nullable=False)
    feature_version_id = Column(String(64), nullable=False)
    feature_snapshot_id = Column(String(64))
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    predicted_direction = Column(String(10))
    predicted_confidence = Column(Float)
    trade_id = Column(String(64))
    outcome_profitable = Column(Boolean)
    created_at = Column(DateTime)


# ─── Feedback Loop (Experience Database) ──────────────────────

class MLPrediction(MLBase):
    __tablename__ = "ml_predictions"

    prediction_id = Column(String(128), primary_key=True)
    trade_id = Column(String(64))
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    model_version = Column(String(64))
    strategy_name = Column(String(64))
    predicted_direction = Column(String(10))
    predicted_confidence = Column(Float)
    predicted_win_probability = Column(Float)
    predicted_return_pct = Column(Float)
    predicted_duration_minutes = Column(Integer)
    predicted_risk_reward = Column(Float)
    market_regime = Column(String(32))
    volatility_level = Column(String(32))
    feature_hash = Column(String(32))
    contract_id = Column(String(64))
    created_at = Column(DateTime)


class MLOutcome(MLBase):
    __tablename__ = "ml_outcomes"

    trade_id = Column(String(64), primary_key=True)
    symbol = Column(String(32), nullable=False)
    actual_profitable = Column(Boolean)
    actual_return_pct = Column(Float)
    actual_duration_minutes = Column(Integer)
    actual_risk_reward = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    stop_loss_price = Column(Float)
    stop_loss_hit = Column(Boolean)
    max_favorable_excursion = Column(Float)
    max_adverse_excursion = Column(Float)
    slippage_pct = Column(Float)
    fees = Column(Float)
    contract_id = Column(String(64))
    contract_decision = Column(String(20))
    contract_confidence = Column(Float)
    created_at = Column(DateTime)


class MLScorecard(MLBase):
    __tablename__ = "ml_scorecards"

    trade_id = Column(String(64), primary_key=True)
    symbol = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    direction_score = Column(Float)
    confidence_calibration_error = Column(Float)
    timing_score = Column(Float)
    exit_efficiency = Column(Float)
    overall_score = Column(Float)
    model_version = Column(String(64))
    strategy_name = Column(String(64))
    market_regime = Column(String(32))
    contract_id = Column(String(64))
    created_at = Column(DateTime)


class MLFeatureSnapshotExp(MLBase):
    __tablename__ = "ml_feature_snapshots_exp"

    trade_id = Column(String(64), primary_key=True)
    feature_name = Column(String(128), primary_key=True)
    feature_value = Column(Float)
