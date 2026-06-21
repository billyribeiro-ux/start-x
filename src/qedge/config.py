"""Central configuration for qedge — the single source of truth for every tunable.

The contract bans magic numbers: every threshold, window, and coefficient used
anywhere in the package is declared here and read from a :class:`QedgeConfig`
instance. Sections are nested pydantic models so each subsystem owns its own
namespace. Environment overrides (and the gitignored ``.env``) are loaded by the
root settings object; the live FMP key itself is read by the reused
``startx.settings`` layer from the same ``.env``.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Data-layer knobs: universe, history window, and dissemination latencies."""

    universe: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    benchmark: str = "SPY"
    # Indicator warm-up history; the analysis window is clipped downstream.
    history_start: str = "2010-01-01"
    cache_dir: str = "data/cache"
    # Realistic dissemination latency (trading days) applied to lagged feeds so
    # cross-asset / news features never use information before it was public.
    news_latency_days: int = 1
    cross_asset_latency_days: int = 1


class LabelingConfig(BaseModel):
    """Triple-barrier presets (reused from startx.labeling)."""

    short_horizon_days: int = 10
    short_pt_mult: float = 1.5
    short_sl_mult: float = 1.5
    short_vol_span: int = 21
    long_horizon_days: int = 63
    long_pt_mult: float = 2.5
    long_sl_mult: float = 2.5
    long_vol_span: int = 63


class ValidationConfig(BaseModel):
    """CPCV + embargo and the DSR/PBO survival gate."""

    cpcv_n_groups: int = 6
    cpcv_n_test_groups: int = 2
    embargo_pct: float = 0.01
    # Survival gate: an edge ships only if DSR >= dsr_min AND PBO <= pbo_max.
    dsr_min: float = 0.95
    pbo_max: float = 0.5
    walkforward_min_train: int = 504  # ~2 trading years before first OOS fold
    walkforward_test_span: int = 63


class FeatureConfig(BaseModel):
    """Feature engineering: fractional differentiation and clustering."""

    # Fractional-diff order search grid and the ADF stationarity threshold.
    fracdiff_d_grid: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )
    fracdiff_adf_pvalue: float = 0.05
    fracdiff_weight_threshold: float = 1e-4  # FFD fixed-window weight cutoff
    # Hierarchical feature clustering (correlation distance) before importance.
    cluster_distance_threshold: float = 0.5


class RegimeConfig(BaseModel):
    """First-class regime layer: HMM, change-point, vol/correlation clustering."""

    hmm_n_states: int = 3
    hmm_covariance_type: str = "diag"
    hmm_n_iter: int = 100
    changepoint_penalty: float = 10.0
    regime_lookback_days: int = 252


class OnlineConfig(BaseModel):
    """Online / incremental learning and concept-drift detection."""

    adwin_delta: float = 0.002
    page_hinkley_threshold: float = 50.0
    page_hinkley_delta: float = 0.005


class ExecutionConfig(BaseModel):
    """Execution realism applied BEFORE any Sharpe is reported."""

    commission_bps: float = 0.5
    slippage_bps: float = 1.0
    # Round-trip equity cost in basis points (per side already included above).
    spy_cost_bps: float = 2.0
    partial_fill_participation: float = 0.1  # max share of bar volume per fill
    intraday_latency_ms: float = 0.0  # stated; unused until an intraday strategy
    # Options: bid-ask is the dominant cost — modelled as half-spread per side.
    options_half_spread_bps: float = 50.0


class CapacityConfig(BaseModel):
    """Capacity / market-impact decay — the AUM at which the edge dies."""

    adv_participation_grid: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1)
    impact_coefficient: float = 0.1  # square-root-impact scaling
    aum_grid_usd: tuple[float, ...] = (
        1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9,
    )


class ScannerConfig(BaseModel):
    """Top-level scanner / reproducibility knobs."""

    seed: int = 7
    output_dir: str = "data/qedge_runs"
    min_trials_for_deflation: int = 1


class QedgeConfig(BaseSettings):
    """Root config. Nested sections are accessed as ``cfg.data``, ``cfg.validation`` …

    Environment variables use the ``QEDGE_`` prefix with ``__`` delimiting nested
    fields (e.g. ``QEDGE_VALIDATION__DSR_MIN=0.9``). The ``.env`` file is read but
    extra/unknown keys (such as ``FMP_API_KEY``) are ignored here — the FMP key is
    consumed by the reused ``startx.settings`` layer.
    """

    model_config = SettingsConfigDict(
        env_prefix="QEDGE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    data: DataConfig = Field(default_factory=DataConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)


_CONFIG: QedgeConfig | None = None


def get_config() -> QedgeConfig:
    """Return the process-wide config singleton (cached after first construction)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = QedgeConfig()
    return _CONFIG
