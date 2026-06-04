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
class DistFitResult:
    mu: float           # log-scale mean
    sigma: float        # log-scale std
    ks_stat: float      # KS 통계량
    ks_pvalue: float    # p-value: >0.05 → 로그정규 가설 기각 못함 (적합 양호)
    fit_quality: str    # "good" / "marginal" / "poor"
    p50_dist: float     # 분포 기반 P50 (경험적 값 대체에 사용)
    p75_dist: float
    p95_dist: float


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
    credit_share_pct: float = 0.0  # 전체 크레딧 중 이 세그먼트 비율


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
    # 95% CI for each percentile
    ci_p50_lower: float
    ci_p50_upper: float
    ci_p75_lower: float
    ci_p75_upper: float
    ci_p95_lower: float
    ci_p95_upper: float
    # Annual estimates (with ramp-up)
    annual_p50: float
    annual_p75: float
    annual_p95: float
    # Annual steady-state (no ramp)
    annual_steady_p50: float
    annual_steady_p75: float
    annual_steady_p95: float
    # Which percentile source was used
    used_dist_fit: bool = False
    # Monte Carlo CI (None when --monte-carlo not specified)
    mc_ci_p50_lower: Optional[float] = None
    mc_ci_p50_upper: Optional[float] = None
    mc_ci_p75_lower: Optional[float] = None
    mc_ci_p75_upper: Optional[float] = None
    mc_ci_p95_lower: Optional[float] = None
    mc_ci_p95_upper: Optional[float] = None


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
    working_day_ratio: float        # 실제 활동일 / 전체 기간 (근무일 보정용)
    monthly_per_user: list          # 사용자별 월 크레딧 (차트용 원본 데이터)
    dist_fit: DistFitResult         # 분포 적합 결과
    usage_type_summary: dict
    usage_unit_summary: dict
    segment_stats: list
    scenarios: list
    sensitivity: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
