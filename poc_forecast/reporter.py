import csv
from tabulate import tabulate
from .schemas import ForecastReport
from .distribution import describe_fit
from .scaling import RAMP_SUM


def _fmt(value: float, credit_to_usd: float) -> str:
    if credit_to_usd != 1.0:
        return f"${value * credit_to_usd:,.0f}"
    return f"{value:,.1f}"


def _ci(lo: float, hi: float, usd: float) -> str:
    return f"[{_fmt(lo, usd)} ~ {_fmt(hi, usd)}]"


def print_report(report: ForecastReport, verbose: bool = False):
    usd = report.credit_to_usd
    unit = "USD" if usd != 1.0 else "크레딧"

    print()
    print("=" * 68)
    print("  POC AI 크레딧 사용량 분석 및 전사 도입 비용 예측")
    print("=" * 68)

    if report.warnings:
        print("\n[경고]")
        for w in report.warnings:
            print(f"  ⚠  {w}")

    # ── Section 1: POC 요약 ──────────────────────────────
    print("\n[1] POC 요약")
    print("-" * 48)
    summary = [
        ["분석 기간",      f"{report.poc_start} ~ {report.poc_end} ({report.poc_period_days}일)"],
        ["POC 사용자 수",  f"{report.poc_users}명"],
        ["전사 총 인원",   f"{report.enterprise_users:,}명"],
        ["POC 총 크레딧",  f"{report.total_poc_credit:,.1f} cr"],
        ["활동일 비율",    f"{report.working_day_ratio:.1%}  (근무일 기준: 22일/월 환산)"],
    ]
    print(tabulate(summary, tablefmt="plain"))

    if report.usage_type_summary:
        print("\n  서비스별 크레딧 비율:")
        total = sum(report.usage_type_summary.values())
        for ut, cr in sorted(report.usage_type_summary.items(), key=lambda x: -x[1])[:10]:
            pct = cr / total * 100 if total > 0 else 0
            print(f"    {ut:<30} {cr:>10,.1f} cr  ({pct:.1f}%)")

    # ── Section 2: 사용자 세분화 ────────────────────────
    print("\n[2] 사용자 세분화 (POC 기간 총 크레딧 기준)")
    print("-" * 68)
    seg_rows = [[
        s.name, s.user_count,
        f"{s.active_day_ratio_mean:.0%}",
        _fmt(s.credit_per_user_per_day_p50, usd),
        _fmt(s.monthly_credit_per_user_p50, usd),
        _fmt(s.monthly_credit_per_user_p75, usd),
        _fmt(s.monthly_credit_per_user_p95, usd),
        f"{s.credit_share_pct:.1f}%",
    ] for s in report.segment_stats]
    print(tabulate(
        seg_rows,
        headers=["세그먼트", "인원", "활성일%", "일평균P50", "월P50", "월P75", "월P95", "크레딧비율"],
        tablefmt="simple", disable_numparse=True,
    ))

    # ── Section 3: 예측 추론 사슬 ────────────────────────
    fit = report.dist_fit
    dist_source = "로그정규 분포 기반" if fit.fit_quality == "good" else "경험적 (표본 직접 계산)"
    print("\n[3] 예측 추론 사슬 및 가정")
    print("-" * 68)
    print(f"  Step 1. POC 관측 ({report.poc_users}명, {report.poc_period_days}일) → 월간 기준 변환")
    print(f"          근무일 22일/월, 활동일 비율 {report.working_day_ratio:.1%} 반영")
    print(f"  Step 2. 분포 적합: {describe_fit(fit)}")
    print(f"          백분위수 소스: {dist_source}")
    print(f"  Step 3. 편향 보정: POC 사용자 ÷ {report.bias_factor}x")
    print(f"          (POC 참가자는 일반 직원보다 약 {report.bias_factor}x 더 사용 가정)")
    print(f"          타당 범위: 1.5x(POC≈일반) ~ 4.0x(열성 얼리어답터)")
    print(f"  Step 4. 채택률: Conservative 20% / Moderate 45% / Aggressive 70%")
    print(f"  Step 5. 월 비용 = 보정된 사용자당 크레딧 × 활성 사용자 수")
    print(f"  Step 6. 연간 비용 = 월 비용 × {RAMP_SUM:.2f}")
    print(f"          (램프업: 1-2월 30%, 3-4월 60%, 5-6월 80%, 7-12월 100%)")

    # ── Section 4: 역할 기반 비용 예측 (권장) or 로그정규 fallback ──
    if report.role_scenarios:
        # ── [4] 역할 기반 (PRIMARY) ─────────────────────────
        print("\n[4] 역할 기반 비용 예측  ★ (Executive Summary 예산 기준)")
        print("-" * 68)
        rs = report.role_scenarios[0]
        print(f"  bias({report.bias_factor}x) 보정 후 세그먼트 비율:")
        print(f"    Heavy {rs.adj_heavy_ratio:.1%}  /  Medium {rs.adj_medium_ratio:.1%}  /  Light {rs.adj_light_ratio:.1%}")
        role_rows = [[
            r.name,
            f"{r.adoption_rate:.0%}",
            f"{r.heavy_n:,}",
            f"{r.medium_n:,}",
            f"{r.light_n:,}",
            _fmt(r.monthly_p50, usd),
            _fmt(r.monthly_p75, usd),
            _fmt(r.monthly_p95, usd),
            _fmt(r.annual_p50, usd),
            _fmt(r.annual_p75, usd),
        ] for r in report.role_scenarios]
        print(tabulate(
            role_rows,
            headers=["시나리오", "채택률", "Heavy", "Medium", "Light",
                     "월P50", "월P75★", "월P95", "연P50(램프)", "연P75(램프)"],
            tablefmt="simple", disable_numparse=True,
        ))
        print(f"  ★ 권장 예산 기준: Moderate 월P75 = {_fmt(next(r.monthly_p75 for r in report.role_scenarios if r.name=='Moderate'), usd)} {unit}")
        print("  * P50~P95: 같은 채택률 내 POC 세그먼트 분산 반영 (CI보다 좁음 — 채택률 불확실성 별도)")

        # ── [5] 로그정규 (검증용) ────────────────────────────
        if report.scenarios:
            print("\n[5] 로그정규 분포 예측 (검증용 — Bootstrap CI 포함)")
            print("-" * 68)
            print("  참고: CI 범위가 넓어 예산 편성보다 불확실성 파악에 활용하세요.")
            header = [""] + [f"{s.name}\n({s.adoption_rate:.0%}, {s.active_users:,}명)"
                             for s in report.scenarios]
            rows = []
            rows.append(["── 월간 ──"] + [""] * len(report.scenarios))
            has_mc = any(s.mc_ci_p50_lower is not None for s in report.scenarios)
            ci_label = "  95% CI (MC)" if has_mc else "  95% CI"

            def _pick_ci(s, mc_lo, mc_hi, boot_lo, boot_hi):
                if mc_lo is not None:
                    return _ci(mc_lo, mc_hi, usd)
                return _ci(boot_lo, boot_hi, usd)

            rows.append(["P50"] + [_fmt(s.monthly_p50, usd) for s in report.scenarios])
            rows.append([ci_label] + [_pick_ci(s, s.mc_ci_p50_lower, s.mc_ci_p50_upper, s.ci_p50_lower, s.ci_p50_upper) for s in report.scenarios])
            rows.append(["P75"] + [_fmt(s.monthly_p75, usd) for s in report.scenarios])
            rows.append([ci_label] + [_pick_ci(s, s.mc_ci_p75_lower, s.mc_ci_p75_upper, s.ci_p75_lower, s.ci_p75_upper) for s in report.scenarios])
            rows.append(["P95"] + [_fmt(s.monthly_p95, usd) for s in report.scenarios])
            rows.append([ci_label] + [_pick_ci(s, s.mc_ci_p95_lower, s.mc_ci_p95_upper, s.ci_p95_lower, s.ci_p95_upper) for s in report.scenarios])
            rows.append(["── 연간 (램프업 포함) ──"] + [""] * len(report.scenarios))
            rows.append(["P50"] + [_fmt(s.annual_p50, usd) for s in report.scenarios])
            rows.append(["P75"] + [_fmt(s.annual_p75, usd) for s in report.scenarios])
            rows.append(["P95"] + [_fmt(s.annual_p95, usd) for s in report.scenarios])
            print(tabulate(rows, headers=header, tablefmt="simple", disable_numparse=True))
            dist_label = "분포 기반" if report.scenarios[0].used_dist_fit else "경험적"
            ci_src = "Monte Carlo (bias·채택률 불확실성 포함)" if has_mc else "Bootstrap (표본 추정 오차)"
            print(f"  단위: {unit}  |  백분위수 소스: {dist_label}  |  CI 방법: {ci_src}")

    else:
        # role_scenarios 없을 때 로그정규를 [4]로
        print("\n[4] 전사 도입 비용 예측")
        print("-" * 68)
        if report.scenarios:
            header = [""] + [f"{s.name}\n({s.adoption_rate:.0%}, {s.active_users:,}명)" for s in report.scenarios]
            rows = []
            rows.append(["── 월간 ──"] + [""] * len(report.scenarios))
            has_mc = any(s.mc_ci_p50_lower is not None for s in report.scenarios)
            ci_label = "  95% CI (MC)" if has_mc else "  95% CI"

            def _pick_ci(s, mc_lo, mc_hi, boot_lo, boot_hi):
                if mc_lo is not None:
                    return _ci(mc_lo, mc_hi, usd)
                return _ci(boot_lo, boot_hi, usd)

            rows.append(["P50"] + [_fmt(s.monthly_p50, usd) for s in report.scenarios])
            rows.append([ci_label] + [_pick_ci(s, s.mc_ci_p50_lower, s.mc_ci_p50_upper, s.ci_p50_lower, s.ci_p50_upper) for s in report.scenarios])
            rows.append(["P75"] + [_fmt(s.monthly_p75, usd) for s in report.scenarios])
            rows.append([ci_label] + [_pick_ci(s, s.mc_ci_p75_lower, s.mc_ci_p75_upper, s.ci_p75_lower, s.ci_p75_upper) for s in report.scenarios])
            rows.append(["P95"] + [_fmt(s.monthly_p95, usd) for s in report.scenarios])
            rows.append([ci_label] + [_pick_ci(s, s.mc_ci_p95_lower, s.mc_ci_p95_upper, s.ci_p95_lower, s.ci_p95_upper) for s in report.scenarios])
            rows.append(["── 연간 (램프업 포함) ──"] + [""] * len(report.scenarios))
            rows.append(["P50"] + [_fmt(s.annual_p50, usd) for s in report.scenarios])
            rows.append(["P75"] + [_fmt(s.annual_p75, usd) for s in report.scenarios])
            rows.append(["P95"] + [_fmt(s.annual_p95, usd) for s in report.scenarios])
            print(tabulate(rows, headers=header, tablefmt="simple", disable_numparse=True))
            dist_label = "분포 기반" if report.scenarios[0].used_dist_fit else "경험적"
            ci_src = "Monte Carlo (bias·채택률 불확실성 포함)" if has_mc else "Bootstrap (표본 추정 오차)"
            print(f"  단위: {unit}  |  백분위수 소스: {dist_label}  |  CI 방법: {ci_src}")
            print("  * 용어: P50=중앙값(절반이 이 이하), P75=상위25%선(예산 기준), P95=상위5%선(최악 대비)")
            print("  * 95% CI=참값이 95% 확률로 들어가는 범위 (표본 한계로 인한 불확실성)")

    # ── Section 6: 민감도 ───────────────────────────────
    if report.sensitivity:
        print("\n[6] 편향 계수 민감도 — 로그정규 방법 기준 Moderate P75")
        print("-" * 55)
        sens_rows = [[
            f"{r['bias_factor']}x",
            _fmt(r["credit_per_user_monthly_p75"], usd),
            _fmt(r["total_monthly_p75"], usd),
        ] for r in report.sensitivity]
        print(tabulate(
            sens_rows,
            headers=["편향 계수", "사용자당 월P75", "전체 월P75 (Moderate)"],
            tablefmt="simple", disable_numparse=True,
        ))

    # ── Section 7: 부서별 티어 분배 권장 ─────────────────
    if report.tier_allocations and report.tier_company:
        comp = report.tier_company
        tdefs = report.tier_defs
        n_t = len(tdefs)
        print(f"\n[7] 부서별 티어 분배 권장 (편향보정 ÷{report.bias_factor}x · 평균 사용량 기준)")
        print("-" * 68)
        cap_str = " / ".join(f"{t.name} ${t.monthly_usd:,.0f}" for t in tdefs)
        cr_rate = report.credit_to_usd
        print(f"  티어 상한: {cap_str}  (1cr=${cr_rate:g})")
        mix_str = " / ".join(f"{tdefs[i].name} {comp['tier_pcts'][i]:.0f}%" for i in range(n_t))
        print(f"  ── 전사 합산 믹스: {mix_str}")
        util = comp["avg_expected_usd"] / comp["avg_committed_usd"] if comp["avg_committed_usd"] else 0
        print(f"     1인당 평균 실사용 ${comp['avg_expected_usd']:,.0f}/월 · "
              f"평균 배정 상한 ${comp['avg_committed_usd']:,.0f}/월 · 활용률 {util:.0%}")
        if comp["flat_top_usd"] > 0:
            save = 1 - comp["avg_committed_usd"] / comp["flat_top_usd"]
            print(f"     전원 최고티어(${comp['flat_top_usd']:,.0f}) 대비 천장 절감 {save:.0%}")

        tier_headers = [t.name + "%" for t in tdefs]
        rows = []
        for a in report.tier_allocations:
            mark = "★" if a.n_users < 5 else ""
            rows.append(
                [f"{mark}{a.department}", a.n_users]
                + [f"{a.tier_pcts[i]:.0f}" for i in range(n_t)]
                + [f"${a.avg_monthly_usd:,.0f}", f"${a.committed_per_user_usd:,.0f}"]
            )
        print(tabulate(
            rows,
            headers=["부서", "인원"] + tier_headers + ["평균사용$", "배정상한$"],
            tablefmt="simple", disable_numparse=True,
        ))
        if n_t >= 3 and abs(tdefs[2].monthly_usd - tdefs[1].monthly_usd) < tdefs[1].monthly_usd:
            print(f"  * {tdefs[1].name}(${tdefs[1].monthly_usd:,.0f})·{tdefs[2].name}(${tdefs[2].monthly_usd:,.0f})는 "
                  f"상한이 근접해 {tdefs[2].name} 배정이 적습니다.")
        print("  * ★ = 인원 5명 미만 부서 (표본이 작아 비율 변동이 큼)")
        if comp["over_total"]:
            print(f"  * 최고 티어 상한 초과 {comp['over_total']}명 — 상한 상향 또는 별도 관리 필요")

    print()


def export_csv(report: ForecastReport, path: str):
    usd = report.credit_to_usd
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "adoption_pct", "active_users", "bias_factor",
            "percentile", "monthly_credit", "annual_credit_with_ramp",
            "annual_credit_steady", "monthly_usd", "annual_usd_with_ramp",
            "ci_lower_credit", "ci_upper_credit",
            "poc_period_days", "poc_users", "poc_total_credit",
            "dist_fit_quality", "dist_mu", "dist_sigma",
        ])
        for s in report.scenarios:
            for pct, monthly, annual, annual_s, ci_lo, ci_hi in [
                ("P50", s.monthly_p50, s.annual_p50, s.annual_steady_p50, s.ci_p50_lower, s.ci_p50_upper),
                ("P75", s.monthly_p75, s.annual_p75, s.annual_steady_p75, s.ci_p75_lower, s.ci_p75_upper),
                ("P95", s.monthly_p95, s.annual_p95, s.annual_steady_p95, s.ci_p95_lower, s.ci_p95_upper),
            ]:
                writer.writerow([
                    s.name, f"{s.adoption_rate:.0%}", s.active_users, s.bias_factor,
                    pct,
                    round(monthly, 2), round(annual, 2), round(annual_s, 2),
                    round(monthly * usd, 2), round(annual * usd, 2),
                    round(ci_lo, 2), round(ci_hi, 2),
                    report.poc_period_days, report.poc_users,
                    round(report.total_poc_credit, 2),
                    report.dist_fit.fit_quality,
                    round(report.dist_fit.mu, 4),
                    round(report.dist_fit.sigma, 4),
                ])
    print(f"결과 CSV 저장: {path}")
