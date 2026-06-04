from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserStats:
    user_id: str
    total_credit: float
    total_quantity: float
    active_days: int
    poc_days: int
    daily_credit_mean: float
    daily_credit_std: float
    active_day_ratio: float
    segment: str = ""
    department: str = ""
    division: str = ""
    usage_type_breakdown: dict = field(default_factory=dict)


@dataclass
class SegmentStats:
    name: str          # "Heavy", "Medium", "Light"
    user_count: int
    credit_per_user_per_day_mean: float
    credit_per_user_per_day_p50: float
    credit_per_user_per_day_p75: float
    credit_per_user_per_day_p95: float
    active_day_ratio_mean: float
    monthly_credit_per_user_p50: float
    monthly_credit_per_user_p75: float
    monthly_credit_per_user_p95: float


@dataclass
class ScenarioResult:
    name: str               # "Conservative", "Moderate", "Aggressive"
    adoption_rate: float    # 0.2, 0.45, 0.7
    active_users: int
    bias_factor: float
    # Monthly cost estimates
    monthly_p50: float
    monthly_p75: float
    monthly_p95: float
    # 95% CI for P50 monthly
    ci_lower: float
    ci_upper: float
    # Annual estimates (with ramp-up)
    annual_p50: float
    annual_p75: float
    annual_p95: float
    # Annual steady-state (no ramp)
    annual_steady_p50: float
    annual_steady_p75: float
    annual_steady_p95: float


@dataclass
class ForecastReport:
    poc_period_days: int
    poc_start: str
    poc_end: str
    poc_users: int
    enterprise_users: int
    bias_factor: float
    total_poc_credit: float
    credit_to_usd: float
    usage_type_summary: dict
    usage_unit_summary: dict
    segment_stats: list
    scenarios: list
    warnings: list = field(default_factory=list)
