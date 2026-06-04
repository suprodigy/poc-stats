from .schemas import ScenarioResult

SCENARIOS = {
    "conservative": ("Conservative", 0.20),
    "moderate":     ("Moderate",     0.45),
    "aggressive":   ("Aggressive",   0.70),
}

# Ramp-up multiplier per month (index 0 = month 1)
RAMP_MULTIPLIERS = [0.30, 0.30, 0.60, 0.60, 0.80, 0.80] + [1.0] * 6  # 12 months


def forecast_scenarios(
    population_percentiles: dict,
    bootstrap_ci: dict,
    enterprise_users: int,
    bias_factor: float,
    scenarios: list[str],
    apply_ramp: bool = True,
) -> list[ScenarioResult]:
    results = []

    for key in scenarios:
        if key not in SCENARIOS:
            continue
        label, adoption_rate = SCENARIOS[key]
        active_users = int(enterprise_users * adoption_rate)

        # Per-user monthly credit after bias correction
        p50_per_user = population_percentiles["p50"] / bias_factor
        p75_per_user = population_percentiles["p75"] / bias_factor
        p95_per_user = population_percentiles["p95"] / bias_factor

        monthly_p50 = p50_per_user * active_users
        monthly_p75 = p75_per_user * active_users
        monthly_p95 = p95_per_user * active_users

        # Bootstrap CI (based on mean, applied to P50 scale)
        ci_lower = bootstrap_ci["p50_lower"] / bias_factor * active_users
        ci_upper = bootstrap_ci["p50_upper"] / bias_factor * active_users

        # Annual steady-state (12 * monthly)
        annual_steady_p50 = monthly_p50 * 12
        annual_steady_p75 = monthly_p75 * 12
        annual_steady_p95 = monthly_p95 * 12

        if apply_ramp:
            annual_p50 = sum(monthly_p50 * m for m in RAMP_MULTIPLIERS)
            annual_p75 = sum(monthly_p75 * m for m in RAMP_MULTIPLIERS)
            annual_p95 = sum(monthly_p95 * m for m in RAMP_MULTIPLIERS)
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
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            annual_p50=annual_p50,
            annual_p75=annual_p75,
            annual_p95=annual_p95,
            annual_steady_p50=annual_steady_p50,
            annual_steady_p75=annual_steady_p75,
            annual_steady_p95=annual_steady_p95,
        ))

    return results


def bias_sensitivity_table(
    population_percentiles: dict,
    enterprise_users: int,
    adoption_rate: float = 0.45,
    bias_factors: list[float] = None,
) -> list[dict]:
    if bias_factors is None:
        bias_factors = [1.5, 2.0, 2.5, 3.0, 4.0]

    active_users = int(enterprise_users * adoption_rate)
    rows = []
    for bf in bias_factors:
        p75_per_user = population_percentiles["p75"] / bf
        rows.append({
            "bias_factor": bf,
            "credit_per_user_monthly_p75": p75_per_user,
            "total_monthly_p75": p75_per_user * active_users,
        })
    return rows
