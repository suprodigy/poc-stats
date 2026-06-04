import numpy as np
import pandas as pd
from .schemas import UserStats, SegmentStats


def build_user_stats(df: pd.DataFrame) -> tuple[list[UserStats], int]:
    """Compute per-user statistics. Returns (user_stats_list, poc_period_days)."""
    poc_days = (df["date_partition"].max() - df["date_partition"].min()).days + 1

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

    return stats_list, poc_days


def assign_segments(user_stats: list[UserStats]) -> list[UserStats]:
    """Assign Heavy/Medium/Light segments based on total credit percentiles."""
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
    user_stats: list[UserStats], poc_days: int
) -> list[SegmentStats]:
    segments = []
    for seg_name in ["Heavy", "Medium", "Light"]:
        users = [u for u in user_stats if u.segment == seg_name]
        if not users:
            continue

        # Daily credit on active days only
        daily_vals = np.array([u.daily_credit_mean for u in users])
        active_ratios = np.array([u.active_day_ratio for u in users])

        # Monthly credit per user = daily_mean * active_ratio * 30
        monthly_vals = daily_vals * active_ratios * 30

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
        ))
    return segments


def compute_population_percentiles(
    user_stats: list[UserStats], poc_days: int
) -> dict:
    """Compute P50/P75/P95 monthly credit per user across the full POC population."""
    # Monthly credit per user = total_credit / poc_days * 30
    monthly_per_user = np.array([
        u.total_credit / poc_days * 30 for u in user_stats
    ])
    return {
        "p50": float(np.percentile(monthly_per_user, 50)),
        "p75": float(np.percentile(monthly_per_user, 75)),
        "p95": float(np.percentile(monthly_per_user, 95)),
        "mean": float(np.mean(monthly_per_user)),
    }


def bootstrap_ci(
    user_stats: list[UserStats],
    poc_days: int,
    n_iterations: int = 2000,
    ci: float = 0.95,
) -> dict:
    """Bootstrap 95% CI for mean monthly credit per user."""
    rng = np.random.default_rng(42)
    monthly = np.array([u.total_credit / poc_days * 30 for u in user_stats])
    n = len(monthly)

    means = []
    p50s = []
    for _ in range(n_iterations):
        sample = rng.choice(monthly, size=n, replace=True)
        means.append(np.mean(sample))
        p50s.append(np.median(sample))

    alpha = (1 - ci) / 2
    return {
        "mean_lower": float(np.percentile(means, alpha * 100)),
        "mean_upper": float(np.percentile(means, (1 - alpha) * 100)),
        "p50_lower": float(np.percentile(p50s, alpha * 100)),
        "p50_upper": float(np.percentile(p50s, (1 - alpha) * 100)),
    }
