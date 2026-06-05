"""부서별 티어 분배 권장 모델.

POC 실측 월 사용량 → 편향 보정 → 평균 사용량을 커버하는 최소 티어 배정 →
부서별 티어 구성 비율(%) 집계.
"""
from .schemas import TierDef, DepartmentTierAllocation

DEFAULT_TIERS_USD  = [12.0, 72.0, 82.0, 492.0]
DEFAULT_TIER_NAMES = ["Tier1", "Tier2", "Tier3", "Tier4"]


def _assign_tier(corrected_usd: float, caps_usd: list) -> int:
    """평균 사용량을 커버하는 가장 작은 티어 인덱스. 모두 초과하면 최고 티어."""
    for i, cap in enumerate(caps_usd):
        if corrected_usd <= cap:
            return i
    return len(caps_usd) - 1


def build_tier_defs(caps_usd: list, names: list, credit_to_usd: float) -> list:
    rate = credit_to_usd if credit_to_usd else 1.0
    return [TierDef(name=n, monthly_usd=c, credit_cap=c / rate)
            for n, c in zip(names, caps_usd)]


def compute_tier_allocations(
    user_stats: list,
    monthly_per_user,
    caps_usd: list,
    names: list,
    credit_to_usd: float,
    bias_factor: float,
) -> tuple:
    """returns (dept_allocations: list[DepartmentTierAllocation], company_summary: dict)."""
    n_tiers = len(caps_usd)
    bias = bias_factor if bias_factor else 1.0

    # 사용자별 (부서, 사업부, 편향보정 월사용$, 티어 idx, OVER)
    rows = []
    for u, mcredit in zip(user_stats, monthly_per_user):
        corrected = mcredit * credit_to_usd / bias
        idx = _assign_tier(corrected, caps_usd)
        over = corrected > caps_usd[-1]
        rows.append((u.department or "unknown", u.division or "", corrected, idx, over))

    # 부서별 그룹화
    depts = {}
    order = []
    for dept, div, corrected, idx, over in rows:
        if dept not in depts:
            depts[dept] = {"div": div, "usages": [], "counts": [0] * n_tiers, "over": 0}
            order.append(dept)
        d = depts[dept]
        d["usages"].append(corrected)
        d["counts"][idx] += 1
        d["over"] += int(over)

    allocations = []
    for dept in order:
        d = depts[dept]
        n = len(d["usages"])
        counts = d["counts"]
        pcts = [c / n * 100 for c in counts]
        avg_usage = sum(d["usages"]) / n
        committed = sum(counts[i] * caps_usd[i] for i in range(n_tiers)) / n
        allocations.append(DepartmentTierAllocation(
            department=dept, division=d["div"], n_users=n,
            tier_counts=counts, tier_pcts=pcts,
            avg_monthly_usd=avg_usage, committed_per_user_usd=committed,
            over_count=d["over"],
        ))

    # 헤비 부서가 위로: 1인당 평균 사용량 내림차순
    allocations.sort(key=lambda a: a.avg_monthly_usd, reverse=True)

    # 전사 합산 요약
    total_n = len(rows)
    total_counts = [sum(a.tier_counts[i] for a in allocations) for i in range(n_tiers)]
    company = {
        "n_users": total_n,
        "tier_counts": total_counts,
        "tier_pcts": [c / total_n * 100 for c in total_counts] if total_n else [0] * n_tiers,
        "avg_expected_usd": sum(r[2] for r in rows) / total_n if total_n else 0.0,
        "avg_committed_usd": (sum(a.committed_per_user_usd * a.n_users for a in allocations) / total_n
                              if total_n else 0.0),
        "flat_top_usd": caps_usd[-1],
        "over_total": sum(a.over_count for a in allocations),
    }
    return allocations, company
