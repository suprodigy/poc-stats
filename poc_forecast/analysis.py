import numpy as np
import pandas as pd
from .schemas import UserStats, SegmentStats
from .distribution import WORKING_DAYS_PER_MONTH

# POC 데이터의 활동일 비율로 근무일 보정 시 사용하는 이론적 근무일 비율 (5/7)
_IDEAL_WORK_RATIO = 5 / 7


def build_user_stats(df: pd.DataFrame) -> tuple[list[UserStats], int, float]:
    """
    사용자별 통계 산출. Returns (user_stats_list, poc_period_days, working_day_ratio).

    working_day_ratio: POC 기간 중 활동이 있었던 날짜 / 전체 기간
      - 이 비율로 캘린더일 기반 사용량을 근무일 기반으로 보정
      - 이상적 값: 5/7 ≈ 0.714 (평일만 사용 시)
    """
    poc_days = (df["date_partition"].max() - df["date_partition"].min()).days + 1

    # 전체 기간 중 실제 사용 기록이 있는 날짜 비율
    active_dates = df["date_partition"].nunique()
    working_day_ratio = max(active_dates / poc_days, 0.1)  # 최소 10% 방어

    user_daily = (
        df.groupby(["email", "date_partition"])
        .agg(daily_credit=("usage_credit", "sum"), daily_qty=("usage_quantity", "sum"))
        .reset_index()
    )

    user_meta = df.groupby("email").agg(
        division=("사업부", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
        department=("부서", lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
    ).reset_index()

    usage_type_totals = (
        df.groupby(["email", "usage_type"])["usage_credit"]
        .sum()
        .reset_index()
        .rename(columns={"usage_credit": "credit"})
    )

    stats_list = []
    for email, group in user_daily.groupby("email"):
        total_credit = group["daily_credit"].sum()
        active_days = len(group)
        active_ratio = active_days / poc_days if poc_days > 0 else 0
        daily_mean = group["daily_credit"].mean()
        daily_std = group["daily_credit"].std(ddof=1) if len(group) > 1 else 0.0

        meta_row = user_meta[user_meta["email"] == email]
        division = meta_row["division"].iloc[0] if not meta_row.empty else "unknown"
        department = meta_row["department"].iloc[0] if not meta_row.empty else "unknown"

        ut_rows = usage_type_totals[usage_type_totals["email"] == email]
        ut_breakdown = dict(zip(ut_rows["usage_type"], ut_rows["credit"]))

        stats_list.append(UserStats(
            user_id=email,
            total_credit=total_credit,
            total_quantity=group["daily_qty"].sum(),
            active_days=active_days,
            poc_days=poc_days,
            daily_credit_mean=daily_mean,
            daily_credit_std=daily_std,
            active_day_ratio=active_ratio,
            division=division,
            department=department,
            usage_type_breakdown=ut_breakdown,
        ))

    return stats_list, poc_days, working_day_ratio


def _monthly_scale(poc_days: int, working_day_ratio: float) -> float:
    """
    캘린더일 기준 POC 크레딧을 월간 근무일 기준으로 변환하는 계수.

    공식: WORKING_DAYS_PER_MONTH / (poc_days * working_day_ratio)
    의미: POC 기간의 실 근무일당 평균 크레딧 → 월 22 근무일 기준으로 확대
    """
    effective_working_days = poc_days * working_day_ratio
    if effective_working_days <= 0:
        return WORKING_DAYS_PER_MONTH / max(poc_days, 1)
    return WORKING_DAYS_PER_MONTH / effective_working_days


def assign_segments(user_stats: list[UserStats]) -> list[UserStats]:
    """Heavy/Medium/Light 세그먼트 할당 (총 크레딧 P60/P80 기준)."""
    totals = np.array([u.total_credit for u in user_stats])
    p80 = np.percentile(totals, 80)
    p60 = np.percentile(totals, 60)

    for u in user_stats:
        if u.total_credit >= p80:
            u.segment = "Heavy"
        elif u.total_credit >= p60:
            u.segment = "Medium"
        else:
            u.segment = "Light"

    return user_stats


def compute_segment_stats(
    user_stats: list[UserStats], poc_days: int, working_day_ratio: float
) -> list[SegmentStats]:
    total_credit_all = sum(u.total_credit for u in user_stats)
    scale = _monthly_scale(poc_days, working_day_ratio)
    segments = []

    for seg_name in ["Heavy", "Medium", "Light"]:
        users = [u for u in user_stats if u.segment == seg_name]
        if not users:
            continue

        daily_vals = np.array([u.daily_credit_mean for u in users])
        active_ratios = np.array([u.active_day_ratio for u in users])

        # 월간 크레딧 = 일평균 × 활성일비율 × 근무일 보정 스케일
        # (daily_mean은 활성일 기준이므로 active_ratio로 캘린더일당으로 변환 후 확대)
        monthly_vals = daily_vals * active_ratios * scale * poc_days

        seg_credit = sum(u.total_credit for u in users)
        credit_share = seg_credit / total_credit_all * 100 if total_credit_all > 0 else 0

        segments.append(SegmentStats(
            name=seg_name,
            user_count=len(users),
            credit_per_user_per_day_mean=float(np.mean(daily_vals)),
            credit_per_user_per_day_p50=float(np.percentile(daily_vals, 50)),
            credit_per_user_per_day_p75=float(np.percentile(daily_vals, 75)),
            credit_per_user_per_day_p95=float(np.percentile(daily_vals, 95)),
            active_day_ratio_mean=float(np.mean(active_ratios)),
            monthly_credit_per_user_p50=float(np.percentile(monthly_vals, 50)),
            monthly_credit_per_user_p75=float(np.percentile(monthly_vals, 75)),
            monthly_credit_per_user_p95=float(np.percentile(monthly_vals, 95)),
            credit_share_pct=float(credit_share),
        ))
    return segments


def compute_monthly_per_user(
    user_stats: list[UserStats], poc_days: int, working_day_ratio: float
) -> np.ndarray:
    """
    사용자별 월 크레딧 배열 반환 (분포 적합 및 백분위수 계산에 사용).

    monthly_i = total_credit_i * scale
    scale = WORKING_DAYS_PER_MONTH / (poc_days * working_day_ratio)
    """
    scale = _monthly_scale(poc_days, working_day_ratio)
    return np.array([u.total_credit * scale for u in user_stats])


def compute_population_percentiles(monthly_per_user: np.ndarray) -> dict:
    """사용자별 월 크레딧 배열로부터 집단 백분위수 산출."""
    return {
        "p50": float(np.percentile(monthly_per_user, 50)),
        "p75": float(np.percentile(monthly_per_user, 75)),
        "p95": float(np.percentile(monthly_per_user, 95)),
        "mean": float(np.mean(monthly_per_user)),
    }


def bootstrap_ci(
    monthly_per_user: np.ndarray,
    n_iterations: int = 2000,
    ci: float = 0.95,
) -> dict:
    """
    P50/P75/P95 각각에 대한 비모수 부트스트랩 신뢰구간.

    고정 seed 제거: 매 실행마다 다른 결과 (통계적 정직성).
    n=100 표본에서 heavy-tail 분포의 분위수 불확실성을 포착하기 위해 부트스트랩 사용.
    """
    rng = np.random.default_rng()  # 비결정적
    n = len(monthly_per_user)
    alpha = (1 - ci) / 2

    p50s, p75s, p95s = [], [], []
    for _ in range(n_iterations):
        sample = rng.choice(monthly_per_user, size=n, replace=True)
        p50s.append(np.percentile(sample, 50))
        p75s.append(np.percentile(sample, 75))
        p95s.append(np.percentile(sample, 95))

    def _ci(arr):
        return float(np.percentile(arr, alpha * 100)), float(np.percentile(arr, (1 - alpha) * 100))

    p50_lo, p50_hi = _ci(p50s)
    p75_lo, p75_hi = _ci(p75s)
    p95_lo, p95_hi = _ci(p95s)

    return {
        "p50_lower": p50_lo, "p50_upper": p50_hi,
        "p75_lower": p75_lo, "p75_upper": p75_hi,
        "p95_lower": p95_lo, "p95_upper": p95_hi,
    }
