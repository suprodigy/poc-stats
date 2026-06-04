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

    # ── Section 4: 비용 예측 ────────────────────────────
    print("\n[4] 전사 도입 비용 예측")
    print("-" * 68)
    if report.scenarios:
        header = [""] + [f"{s.name}\n({s.adoption_rate:.0%}, {s.active_users:,}명)" for s in report.scenarios]
        rows = []
        rows.append(["── 월간 ──"] + [""] * len(report.scenarios))
        rows.append(["P50"] + [_fmt(s.monthly_p50, usd) for s in report.scenarios])
        rows.append(["  95% CI"] + [_ci(s.ci_p50_lower, s.ci_p50_upper, usd) for s in report.scenarios])
        rows.append(["P75"] + [_fmt(s.monthly_p75, usd) for s in report.scenarios])
        rows.append(["  95% CI"] + [_ci(s.ci_p75_lower, s.ci_p75_upper, usd) for s in report.scenarios])
        rows.append(["P95"] + [_fmt(s.monthly_p95, usd) for s in report.scenarios])
        rows.append(["  95% CI"] + [_ci(s.ci_p95_lower, s.ci_p95_upper, usd) for s in report.scenarios])
        rows.append(["── 연간 (램프업 포함) ──"] + [""] * len(report.scenarios))
        rows.append(["P50"] + [_fmt(s.annual_p50, usd) for s in report.scenarios])
        rows.append(["P75"] + [_fmt(s.annual_p75, usd) for s in report.scenarios])
        rows.append(["P95"] + [_fmt(s.annual_p95, usd) for s in report.scenarios])
        print(tabulate(rows, headers=header, tablefmt="simple", disable_numparse=True))
        dist_label = "분포 기반" if report.scenarios[0].used_dist_fit else "경험적"
        print(f"  단위: {unit}  |  백분위수 소스: {dist_label}")

    # ── Section 5: 민감도 ───────────────────────────────
    if report.sensitivity:
        print("\n[5] 편향 계수 민감도 (Moderate, P75 기준)")
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
