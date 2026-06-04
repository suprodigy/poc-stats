import numpy as np
from dataclasses import replace
from .schemas import ScenarioResult, DistFitResult

SCENARIOS = {
    "conservative": ("Conservative", 0.20),
    "moderate":     ("Moderate",     0.45),
    "aggressive":   ("Aggressive",   0.70),
}

# 램프업 곡선: 전사 도입 후 정착까지 6개월 소요 가정
# 근거: 엔터프라이즈 소프트웨어 도입 사례에서 일반적으로 관찰되는 S-curve 초기 구간
# 월별 스테디스테이트 대비 비율 (1-6월: 단계적 증가, 7-12월: 안정)
RAMP_MULTIPLIERS = [0.30, 0.30, 0.60, 0.60, 0.80, 0.80] + [1.0] * 6
RAMP_SUM = sum(RAMP_MULTIPLIERS)  # 6.60 → 연간 = 월간 × 6.60


def forecast_scenarios(
    population_percentiles: dict,
    bootstrap_ci: dict,
    dist_fit: DistFitResult,
    enterprise_users: int,
    bias_factor: float,
    scenarios: list[str],
    apply_ramp: bool = True,
) -> list[ScenarioResult]:
    """
    POC→전사 비용 예측 추론 사슬:

    1. 사용자당 월 크레딧 선택:
       - dist_fit.fit_quality == "good" → 로그정규 분포 기반 백분위수 사용
       - 그 외 → 경험적 백분위수 사용 (표본 직접 계산)
       근거: 로그정규 적합이 좋으면 표본 외 구간까지 더 신뢰할 수 있는 추정 가능

    2. 편향 보정 (bias_factor로 나눔):
       POC 참가자 = 자발적 참여 열성 사용자 → 일반 직원보다 bias_factor배 더 사용
       enterprise_user_credit = poc_user_credit / bias_factor

    3. 활성 사용자 수 = enterprise_users × adoption_rate

    4. 월 비용 = enterprise_user_credit_pXX × active_users

    5. 연간 비용 = 월 비용 × RAMP_SUM (램프업 포함 시)
    """
    use_dist = dist_fit.fit_quality == "good"

    # 백분위수 소스 선택
    if use_dist:
        p50_src = dist_fit.p50_dist
        p75_src = dist_fit.p75_dist
        p95_src = dist_fit.p95_dist
    else:
        p50_src = population_percentiles["p50"]
        p75_src = population_percentiles["p75"]
        p95_src = population_percentiles["p95"]

    results = []
    for key in scenarios:
        if key not in SCENARIOS:
            continue
        label, adoption_rate = SCENARIOS[key]
        active_users = int(enterprise_users * adoption_rate)

        # 편향 보정된 사용자당 월 크레딧
        p50_pu = p50_src / bias_factor
        p75_pu = p75_src / bias_factor
        p95_pu = p95_src / bias_factor

        monthly_p50 = p50_pu * active_users
        monthly_p75 = p75_pu * active_users
        monthly_p95 = p95_pu * active_users

        # 신뢰구간 (각 백분위수에 bias 보정 후 사용자 수 곱)
        ci_p50_lo = bootstrap_ci["p50_lower"] / bias_factor * active_users
        ci_p50_hi = bootstrap_ci["p50_upper"] / bias_factor * active_users
        ci_p75_lo = bootstrap_ci["p75_lower"] / bias_factor * active_users
        ci_p75_hi = bootstrap_ci["p75_upper"] / bias_factor * active_users
        ci_p95_lo = bootstrap_ci["p95_lower"] / bias_factor * active_users
        ci_p95_hi = bootstrap_ci["p95_upper"] / bias_factor * active_users

        annual_steady_p50 = monthly_p50 * 12
        annual_steady_p75 = monthly_p75 * 12
        annual_steady_p95 = monthly_p95 * 12

        if apply_ramp:
            annual_p50 = monthly_p50 * RAMP_SUM
            annual_p75 = monthly_p75 * RAMP_SUM
            annual_p95 = monthly_p95 * RAMP_SUM
        else:
            annual_p50 = annual_steady_p50
            annual_p75 = annual_steady_p75
            annual_p95 = annual_steady_p95

        results.append(ScenarioResult(
            name=label,
            adoption_rate=adoption_rate,
            active_users=active_users,
            bias_factor=bias_factor,
            monthly_p50=monthly_p50,
            monthly_p75=monthly_p75,
            monthly_p95=monthly_p95,
            ci_p50_lower=ci_p50_lo,
            ci_p50_upper=ci_p50_hi,
            ci_p75_lower=ci_p75_lo,
            ci_p75_upper=ci_p75_hi,
            ci_p95_lower=ci_p95_lo,
            ci_p95_upper=ci_p95_hi,
            annual_p50=annual_p50,
            annual_p75=annual_p75,
            annual_p95=annual_p95,
            annual_steady_p50=annual_steady_p50,
            annual_steady_p75=annual_steady_p75,
            annual_steady_p95=annual_steady_p95,
            used_dist_fit=use_dist,
        ))

    return results


_MC_ADOPTION_PARAMS = {
    "conservative": (0.20, 0.10),
    "moderate":     (0.45, 0.12),
    "aggressive":   (0.70, 0.10),
}


def _beta_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    """Beta 분포의 α, β를 mean과 std로부터 계산."""
    var = std ** 2
    alpha = mean * (mean * (1 - mean) / var - 1)
    beta  = (1 - mean) * (mean * (1 - mean) / var - 1)
    return max(alpha, 0.5), max(beta, 0.5)


def monte_carlo_ci(
    scenarios: list[ScenarioResult],
    population_percentiles: dict,
    enterprise_users: int,
    bias_factor: float,
    bias_sigma: float = 0.40,
    n_sim: int = 5_000,
    rng_seed: int | None = None,
) -> list[ScenarioResult]:
    """
    bias_factor 와 adoption_rate 의 불확실성을 Monte Carlo 로 반영해
    MC CI 필드를 채운 새 ScenarioResult 리스트를 반환한다.

    분포 설계:
      bias_factor  ~ LogNormal(log(bias_center), bias_sigma)
        → 양수, 중앙값 = bias_center, sigma=0.40 ≈ 90% 범위 1.4x~4.5x
      adoption_rate ~ Beta(α, β)  per scenario
        → 시나리오 중심값(Conservative 20% / Moderate 45% / Aggressive 70%) 기준
    """
    rng = np.random.default_rng(rng_seed)
    bias_sims = rng.lognormal(np.log(bias_factor), bias_sigma, n_sim)

    updated = []
    for s in scenarios:
        key = s.name.lower()
        adopt_mean, adopt_std = _MC_ADOPTION_PARAMS.get(key, (s.adoption_rate, 0.12))
        alpha, beta = _beta_from_mean_std(adopt_mean, adopt_std)
        adopt_sims = rng.beta(alpha, beta, n_sim)

        mc_fields = {}
        for pxx, attr in [("p50", "p50"), ("p75", "p75"), ("p95", "p95")]:
            user_pxx = population_percentiles[pxx]
            costs = user_pxx / bias_sims * enterprise_users * adopt_sims
            mc_fields[f"mc_ci_{attr}_lower"] = float(np.percentile(costs, 2.5))
            mc_fields[f"mc_ci_{attr}_upper"] = float(np.percentile(costs, 97.5))

        updated.append(replace(s, **mc_fields))

    return updated


def bias_sensitivity_table(
    population_percentiles: dict,
    dist_fit: DistFitResult,
    enterprise_users: int,
    adoption_rate: float = 0.45,
    bias_factors: list[float] = None,
) -> list[dict]:
    if bias_factors is None:
        bias_factors = [1.5, 2.0, 2.5, 3.0, 4.0]

    active_users = int(enterprise_users * adoption_rate)
    # 민감도 분석에는 분포 기반(양호) 또는 경험적 P75 사용
    p75_base = dist_fit.p75_dist if dist_fit.fit_quality == "good" else population_percentiles["p75"]
    p50_base = dist_fit.p50_dist if dist_fit.fit_quality == "good" else population_percentiles["p50"]

    rows = []
    for bf in bias_factors:
        rows.append({
            "bias_factor": bf,
            "credit_per_user_monthly_p50": p50_base / bf,
            "credit_per_user_monthly_p75": p75_base / bf,
            "total_monthly_p50": p50_base / bf * active_users,
            "total_monthly_p75": p75_base / bf * active_users,
        })
    return rows
