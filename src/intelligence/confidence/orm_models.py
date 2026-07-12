"""
SQLAlchemy ORM models for Contract Store tables.
"""
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, Index
from sqlalchemy.orm import DeclarativeBase


class ContractBase(DeclarativeBase):
    pass


class CTContract(ContractBase):
    __tablename__ = "ct_contracts"
    __table_args__ = (
        Index("idx_ct_contracts_symbol", "symbol"),
        Index("idx_ct_contracts_decision", "decision"),
        Index("idx_ct_contracts_created", "created_at"),
        Index("idx_ct_contracts_model", "model_version"),
    )

    contract_id = Column(String(64), primary_key=True)
    symbol = Column(String(32), nullable=False)
    direction = Column(String(10), nullable=False)
    decision = Column(String(20), nullable=False)
    final_confidence = Column(Float, nullable=False)
    recommendation = Column(Text)
    vetoed = Column(Boolean, default=False)
    vetoed_by = Column(String(32))
    veto_reason = Column(Text)
    recommended_position_pct = Column(Float, default=0.0)
    kelly_fraction = Column(Float, default=0.0)
    risk_grade = Column(String(10), default="")
    integrity_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False)
    valid_until = Column(DateTime, nullable=False)
    model_version = Column(String(64), default="")
    feature_set_version = Column(String(64), default="")
    dataset_version = Column(String(64), default="")
    walk_forward_passed = Column(Boolean)
    monte_carlo_passed = Column(Boolean)
    model_health_score = Column(Float)
    full_contract_json = Column(Text, nullable=False)
    execution_status = Column(String(20), default="pending")
    trade_id = Column(String(64))
    stored_at = Column(DateTime)


class CTContractStage(ContractBase):
    __tablename__ = "ct_contract_stages"

    contract_id = Column(String(64), primary_key=True)
    stage_name = Column(String(64), primary_key=True)
    score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    weight = Column(Float, nullable=False)
    severity = Column(String(20), default="none")
    veto = Column(Boolean, default=False)
    veto_reason = Column(Text, default="")
    evidence_json = Column(Text, default="{}")
    warnings_json = Column(Text, default="[]")
    blockers_json = Column(Text, default="[]")
    evaluation_ms = Column(Float, default=0.0)


class CTContractOutcome(ContractBase):
    __tablename__ = "ct_contract_outcomes"
    __table_args__ = (
        Index("idx_ct_outcomes_trade", "trade_id"),
    )

    contract_id = Column(String(64), primary_key=True)
    trade_id = Column(String(64), nullable=False)
    symbol = Column(String(32), nullable=False)
    side = Column(String(10), nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    actual_profitable = Column(Boolean)
    holding_minutes = Column(Integer)
    max_favorable_excursion = Column(Float)
    max_adverse_excursion = Column(Float)
    slippage_pct = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    exit_reason = Column(Text, default="")
    closed_at = Column(DateTime)
    recorded_at = Column(DateTime)
