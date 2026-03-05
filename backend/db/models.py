from sqlalchemy import (
    Column, String, Numeric, BigInteger, Integer,
    Boolean, DateTime, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    symbol    = Column(String(20),  nullable=False)
    timeframe = Column(String(5),   nullable=False)  # 1m,5m,15m,1h,1d,1w
    source    = Column(String(20),  nullable=False)  # stooq, alpaca
    ts        = Column(DateTime,    nullable=False)
    open      = Column(Numeric(12, 4), nullable=False)
    high      = Column(Numeric(12, 4), nullable=False)
    low       = Column(Numeric(12, 4), nullable=False)
    close     = Column(Numeric(12, 4), nullable=False)
    volume    = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "source", "ts", name="uq_candle"),
        Index("idx_candles_symbol_tf_ts", "symbol", "timeframe", "ts"),
        Index("idx_candles_symbol_tf_src_ts", "symbol", "timeframe", "source", "ts"),
        Index("idx_candles_ts", "ts"),
    )


class Zone(Base):
    __tablename__ = "zones"

    id             = Column(String(80), primary_key=True)
    symbol         = Column(String(20), nullable=False)
    timeframe      = Column(String(5),  nullable=False)
    zone_type      = Column(String(10), nullable=False)  # demand, supply
    price_low      = Column(Numeric(12, 4), nullable=False)
    price_high     = Column(Numeric(12, 4), nullable=False)
    strength_score = Column(Numeric(5, 4))
    touch_count    = Column(Integer, default=1)
    is_active      = Column(Boolean, default=True)
    first_formed_at = Column(DateTime)
    last_tested_at  = Column(DateTime)
    invalidated_at  = Column(DateTime)
    created_at      = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_zones_symbol_tf_active", "symbol", "timeframe", "is_active"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id                 = Column(String(80), primary_key=True)
    symbol             = Column(String(20), nullable=False)
    timeframe          = Column(String(5),  nullable=False)
    direction          = Column(String(5),  nullable=False)  # long, short
    setup_type         = Column(String(60), nullable=False)
    strategy           = Column(String(40))
    entry_price        = Column(Numeric(12, 4))
    stop_price         = Column(Numeric(12, 4))
    target_price       = Column(Numeric(12, 4))
    r_multiple         = Column(Numeric(6, 2))
    prob_success       = Column(Numeric(5, 4))
    status             = Column(String(20), default="active")
    zone_id            = Column(String(80))
    context_snapshot   = Column(Text)       # JSON string (SQLite-safe)
    created_at         = Column(DateTime, default=datetime.utcnow)
    horizon_expires_at = Column(DateTime)
    closed_at          = Column(DateTime)
    close_price        = Column(Numeric(12, 4))
    realized_r         = Column(Numeric(6, 2))

    __table_args__ = (
        Index("idx_signals_symbol_status", "symbol", "status"),
        Index("idx_signals_created_at", "created_at"),
    )


class SignalOutcome(Base):
    __tablename__ = "signal_outcomes"

    id                  = Column(String(120), primary_key=True)  # "{signal_id}_{horizon_bars}"
    signal_id           = Column(String(80),  nullable=False)
    symbol              = Column(String(20),  nullable=False)
    timeframe           = Column(String(5),   nullable=False)
    source              = Column(String(20))
    signal_ts           = Column(DateTime,    nullable=False)
    horizon_bars        = Column(Integer,     nullable=False)
    entry_price         = Column(Numeric(12, 4))
    exit_ts             = Column(DateTime)
    exit_price          = Column(Numeric(12, 4))
    return_pct          = Column(Numeric(10, 6))  # signed, direction-aware
    max_favorable_pct   = Column(Numeric(10, 6))  # MFE over horizon
    max_adverse_pct     = Column(Numeric(10, 6))  # MAE over horizon
    created_at          = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_signal_outcomes_signal_id", "signal_id"),
        Index("idx_signal_outcomes_symbol_tf_ts", "symbol", "timeframe", "signal_ts"),
    )


class Setup(Base):
    __tablename__ = "setups"

    id            = Column(String(160), primary_key=True)  # "{run_id}_{symbol}_{timeframe}_{source}_{ts_tag}"
    run_id        = Column(Integer,     nullable=False)
    symbol        = Column(String(20),  nullable=False)
    timeframe     = Column(String(5),   nullable=False)
    source        = Column(String(20))
    ts            = Column(DateTime,    nullable=False)
    close         = Column(Numeric(12, 4))
    ema_20        = Column(Numeric(12, 4))
    ema_50        = Column(Numeric(12, 4))
    rsi_14        = Column(Numeric(6, 2))
    atr_14        = Column(Numeric(12, 4))
    score_raw     = Column(Numeric(12, 4))
    score         = Column(Numeric(12, 6))
    distance_pct  = Column(Numeric(8, 4))
    current_state = Column(String(10))
    trigger_type  = Column(String(10))  # "golden" | "death" | None
    created_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_setups_run_id",        "run_id"),
        Index("idx_setups_symbol",        "symbol"),
        Index("idx_setups_ts",            "ts"),
        Index("idx_setups_score",         "score"),
    )


class ProbabilityHistory(Base):
    __tablename__ = "probability_history"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    setup_type        = Column(String(60), nullable=False)
    timeframe         = Column(String(5),  nullable=False)
    symbol            = Column(String(20))  # NULL = all symbols aggregated
    window_start      = Column(DateTime, nullable=False)
    window_end        = Column(DateTime, nullable=False)
    sample_count      = Column(Integer, nullable=False)
    win_count         = Column(Integer, nullable=False)
    win_rate          = Column(Numeric(5, 4))
    avg_r_won         = Column(Numeric(6, 2))
    avg_r_lost        = Column(Numeric(6, 2))
    expected_r        = Column(Numeric(6, 2))
    calibration_score = Column(Numeric(5, 4))
    computed_at       = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_prob_history_setup_tf", "setup_type", "timeframe", "computed_at"),
    )


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_type    = Column(String(30), nullable=False)  # stooq_daily, alpaca_intraday, etc.
    status      = Column(String(10), nullable=False)  # ok, error, partial
    symbols     = Column(Text)       # comma-separated string (SQLite-safe)
    rows_written = Column(Integer)
    error_msg   = Column(Text)
    started_at  = Column(DateTime, nullable=False)
    finished_at = Column(DateTime)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    started_at           = Column(DateTime, nullable=False)
    finished_at          = Column(DateTime)
    timeframe            = Column(String(5),  nullable=False)
    source               = Column(String(20))
    scanned              = Column(Integer)
    candidates_considered = Column(Integer)
    signals_created      = Column(Integer)
    status               = Column(String(10), nullable=False)  # ok, error
    error                = Column(Text)

    __table_args__ = (
        Index("idx_agent_runs_started_at", "started_at"),
    )