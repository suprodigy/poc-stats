import csv
import sys
from tabulate import tabulate
from .schemas import ForecastReport


def _c(value: float, credit_to_usd: float, prefix: str = "") -> str:
    """Format credit value, optionally converting to USD."""
    if credit_to_usd == 1.0:
        return f"{prefix}{value:,.1f} cr"
    return f"{prefix}${value * credit_to_usd:,.0f} (${value:,.1f} cr)"


def _fmt(value: float, credit_to_usd: float) -> str:
    if credit_to_usd != 1.0:
        return f"${value * credit_to_usd:,.0f}"
    return f"{value:,.1f}"


def print_report(report: ForecastReport, verbose: bool = False):
    usd = report.credit_to_usd
    unit = "USD" if usd == 1.0 else f"cr (×{usd} = USD)"

    print()
    print("=" * 65)
    print("  POC AI 크레딧 사용량 분석 및 전사 도입 비용 예측")
    print("=" * 65)

    # Warnings
    if report.warnings:
        print("\n[경고]")
        for w in report.warnings:
            print(f"  ⚠  {w}")

    # Section 1: POC Summary
    print("\n[1] POC 요약")
    print("-" * 45)
    summary = [
        ["분석 기간", f"{report.poc_start} ~ {report.poc_end} ({report.poc_period_days}일)"],
        ["POC 사용자 수", f"{report.poc_users}명"],
        ["전사 총 인원", f"{report.enterprise_users}명"],
        ["POC 총 크레딧", _c(report.total_poc_credit, usd)],
        ["POC 편향 보정 계수", f"{report.bias_factor}x"],
    ]
    print(tabulate(summary, tablefmt="plain"))

    # Usage type breakdown
    if report.usage_type_summary:
        print("\n  서비스별 크레딧 비율:")
        total = sum(report.usage_type_summary.values())
        rows = sorted(report.usage_type_summary.items(), key=lambda x: -x[1])
        for ut, cr in rows[:10]:
            pct = cr / total * 100 if total > 0 else 0
            print(f"    {ut:<30} {cr:>10,.1f} cr  ({pct:.1f}%)")

    # Section 2: User Segmentation
    print("\n[2] 사용자 세분화 (POC 기간 총 크레딧 기준)")
    print("-" * 65)
    seg_rows = []
    for s in report.segment_stats:
        seg_rows.append([
            s.name,
            s.user_count,
            f"{s.active_day_ratio_mean:.0%}",
            _fmt(s.credit_per_user_per_day_p50, usd),
            _fmt(s.monthly_credit_per_user_p50, usd),
            _fmt(s.monthly_credit_per_user_p75, usd),
            _fmt(s.monthly_credit_per_user_p95, usd),
        ])
    print(tabulate(
        seg_rows,
        headers=["세그먼트", "인원", "활성일%", "일평균P50", "월P50", "월P75", "월P95"],
        tablefmt="simple",
        disable_numparse=True,
    ))

    # Section 3: Scaling assumptions
    print("\n[3] 스케일링 가정")
    print("-" * 45)
    print(f"  POC 편향 보정: {report.bias_factor}x")
    print(f"  (POC 참가자는 일반 직원 대비 약 {report.bias_factor}x 더 많이 사용한다고 가정)")
    print(f"  램프업: 전사 도입 1~6개월 단계적 증가, 7개월부터 정상 운영")

    # Section 4: Cost Forecast Table
    print("\n[4] 전사 도입 비용 예측")
    print("-" * 65)
    header = ["", "Conservative\n(채택률20%)", "Moderate\n(채택률45%)", "Aggressive\n(채택률70%)"]
    rows = []

    def _row(label, attr):
        return [label] + [_fmt(getattr(s, attr), usd) for s in report.scenarios]

    rows.append(["활성 사용자"] + [f"{s.active_users:,}명" for s in report.scenarios])
    rows.append(["─" * 14] + ["─" * 18] * len(report.scenarios))
    rows.append(["월간 P50"] + [_fmt(s.monthly_p50, usd) for s in report.scenarios])
    rows.append(["월간 P75"] + [_fmt(s.monthly_p75, usd) for s in report.scenarios])
    rows.append(["월간 P95"] + [_fmt(s.monthly_p95, usd) for s in report.scenarios])
    rows.append(["─" * 14] + ["─" * 18] * len(report.scenarios))
    rows.append(["연간 P50 (램프포함)"] + [_fmt(s.annual_p50, usd) for s in report.scenarios])
    rows.append(["연간 P75 (램프포함)"] + [_fmt(s.annual_p75, usd) for s in report.scenarios])
    rows.append(["연간 P95 (램프포함)"] + [_fmt(s.annual_p95, usd) for s in report.scenarios])
    rows.append(["─" * 14] + ["─" * 18] * len(report.scenarios))
    rows.append(["95% CI (월P50)"] + [
        f"[{_fmt(s.ci_lower, usd)} ~ {_fmt(s.ci_upper, usd)}]"
        for s in report.scenarios
    ])

    print(tabulate(rows, headers=header, tablefmt="simple"))
    print(f"  단위: {unit}")

    # Section 5: Bias Sensitivity
    if hasattr(report, "sensitivity") and report.sensitivity:
        print("\n[5] 편향 계수 민감도 분석 (Moderate 시나리오, P75 기준)")
        print("-" * 55)
        sens_rows = [[
            f"{r['bias_factor']}x",
            _fmt(r["credit_per_user_monthly_p75"], usd),
            _fmt(r["total_monthly_p75"], usd),
        ] for r in report.sensitivity]
        print(tabulate(
            sens_rows,
            headers=["편향 계수", "사용자당 월P75", "전체 월P75"],
            tablefmt="simple",
            disable_numparse=True,
        ))

    print()


def export_csv(report: ForecastReport, path: str):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "adoption_pct", "active_users", "bias_factor",
            "percentile", "monthly_credit", "annual_credit_with_ramp",
            "annual_credit_steady", "monthly_usd", "annual_usd_with_ramp",
            "ci_lower_credit", "ci_upper_credit",
            "poc_period_days", "poc_users", "poc_total_credit",
        ])
        usd = report.credit_to_usd
        for s in report.scenarios:
            for pct, monthly, annual, annual_s in [
                ("P50", s.monthly_p50, s.annual_p50, s.annual_steady_p50),
                ("P75", s.monthly_p75, s.annual_p75, s.annual_steady_p75),
                ("P95", s.monthly_p95, s.annual_p95, s.annual_steady_p95),
            ]:
                writer.writerow([
                    s.name,
                    f"{s.adoption_rate:.0%}",
                    s.active_users,
                    s.bias_factor,
                    pct,
                    round(monthly, 2),
                    round(annual, 2),
                    round(annual_s, 2),
                    round(monthly * usd, 2),
                    round(annual * usd, 2),
                    round(s.ci_lower, 2) if pct == "P50" else "",
                    round(s.ci_upper, 2) if pct == "P50" else "",
                    report.poc_period_days,
                    report.poc_users,
                    round(report.total_poc_credit, 2),
                ])
    print(f"결과 CSV 저장: {path}")
