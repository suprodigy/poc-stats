from .scaling import RAMP_SUM, SCENARIOS


def _adjust_segment_ratios(segment_stats: list, bias_factor: float) -> dict:
    total = sum(s.user_count for s in segment_stats)
    if total == 0:
        return {"Heavy": 1 / 3, "Medium": 1 / 3, "Light": 1 / 3}

    raw = {s.name: s.user_count / total for s in segment_stats}
    adj = {
        "Heavy":  raw.get("Heavy",  0) / bias_factor,
        "Medium": raw.get("Medium", 0),
        "Light":  raw.get("Light",  0),
    }
    total_adj = sum(adj.values()) or 1.0
    return {k: v / total_adj for k, v in adj.items()}


def compute_role_scenarios(
    segment_stats: list,
    enterprise_users: int,
    bias_factor: float,
    scenario_keys: list,
    apply_ramp: bool = True,
) -> list:
    from .schemas import RoleScenarioResult

    adj_ratios = _adjust_segment_ratios(segment_stats, bias_factor)

    def get_seg(name):
        return next((s for s in segment_stats if s.name == name), None)

    h = get_seg("Heavy")
    m = get_seg("Medium")
    l = get_seg("Light")

    def cost(seg, pct):
        if seg is None:
            return 0.0
        return getattr(seg, f"monthly_credit_per_user_p{pct}", 0.0)

    results = []
    for key in scenario_keys:
        if key not in SCENARIOS:
            continue
        label, adoption_rate = SCENARIOS[key]
        active_users = int(enterprise_users * adoption_rate)

        heavy_n  = round(active_users * adj_ratios["Heavy"])
        medium_n = round(active_users * adj_ratios["Medium"])
        light_n  = round(active_users * adj_ratios["Light"])

        ramp = RAMP_SUM if apply_ramp else 12.0
        results.append(RoleScenarioResult(
            name=label,
            adoption_rate=adoption_rate,
            active_users=active_users,
            heavy_n=heavy_n,
            medium_n=medium_n,
            light_n=light_n,
            adj_heavy_ratio=adj_ratios["Heavy"],
            adj_medium_ratio=adj_ratios["Medium"],
            adj_light_ratio=adj_ratios["Light"],
            monthly_p50=heavy_n * cost(h, 50) + medium_n * cost(m, 50) + light_n * cost(l, 50),
            monthly_p75=heavy_n * cost(h, 75) + medium_n * cost(m, 75) + light_n * cost(l, 75),
            monthly_p95=heavy_n * cost(h, 95) + medium_n * cost(m, 95) + light_n * cost(l, 95),
            annual_p50=(heavy_n * cost(h, 50) + medium_n * cost(m, 50) + light_n * cost(l, 50)) * ramp,
            annual_p75=(heavy_n * cost(h, 75) + medium_n * cost(m, 75) + light_n * cost(l, 75)) * ramp,
            annual_p95=(heavy_n * cost(h, 95) + medium_n * cost(m, 95) + light_n * cost(l, 95)) * ramp,
        ))
    return results
