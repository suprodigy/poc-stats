import argparse
import sys
import numpy as np

from . import ingestion, analysis, scaling, reporter
from .distribution import fit_lognormal
from .schemas import ForecastReport


def main():
    parser = argparse.ArgumentParser(
        description="POC AI 크레딧 사용량 기반 전사 도입 비용 예측 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m poc_forecast poc_data.csv
  python -m poc_forecast poc_data.csv --bias-factor 3.0 --output-html report.html
  python -m poc_forecast poc_data.csv --credit-to-usd 0.01 --no-ramp
        """,
    )
    parser.add_argument("input_file", help="POC 사용량 CSV 또는 Excel 파일 경로")
    parser.add_argument("--enterprise-users", type=int, default=3000, metavar="N")
    parser.add_argument("--poc-users", type=int, default=None, metavar="N",
                        help="POC 실제 참여 인원 (미지정 시 CSV 고유 이메일 수 사용)")
    parser.add_argument("--bias-factor", type=float, default=2.5, metavar="X",
                        help="POC 편향 보정 계수 기본값 2.5 (POC 사용자가 일반 직원보다 X배 더 사용)")
    parser.add_argument("--scenarios", default="conservative,moderate,aggressive")
    parser.add_argument("--credit-to-usd", type=float, default=1.0, metavar="RATE",
                        help="크레딧→USD 환산 비율 (기본: 1.0, 즉 크레딧=USD)")
    parser.add_argument("--output-csv", metavar="PATH")
    parser.add_argument("--output-html", metavar="PATH",
                        help="HTML 리포트 저장 경로 (차트 6개 포함 단일 파일)")
    parser.add_argument("--no-ramp", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000, metavar="N")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="bias_factor·채택률 불확실성을 Monte Carlo로 추가 반영 (CI 확대)")
    parser.add_argument("--mc-bias-sigma", type=float, default=0.40, metavar="S",
                        help="bias_factor 불확실성 로그 스케일 sigma (기본 0.40 ≈ 90%% 범위 1.4x~4.5x)")
    parser.add_argument("--mc-iterations", type=int, default=5000, metavar="N",
                        help="Monte Carlo 반복 횟수 (기본 5000)")
    parser.add_argument("--tiers", default="12,72,82,492", metavar="USD",
                        help="부서별 티어 월 USD 상한 (쉼표구분, 기본 12,72,82,492)")
    parser.add_argument("--tier-names", default="Tier1,Tier2,Tier3,Tier4", metavar="NAMES")
    parser.add_argument("--no-tiers", action="store_true",
                        help="부서별 티어 분배 분석 생략")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ── 데이터 로드 ───────────────────────────────────────
    print(f"파일 로딩 중: {args.input_file}")
    try:
        df, warnings = ingestion.load(args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 사용자 통계 ───────────────────────────────────────
    user_stats, poc_days, working_day_ratio = analysis.build_user_stats(df)
    user_stats = analysis.assign_segments(user_stats)
    poc_users = args.poc_users or len(user_stats)

    if poc_users < 10:
        warnings.append(f"POC 사용자 수 {poc_users}명 — 표본이 너무 작아 예측 신뢰도가 낮습니다.")

    # ── 월간 크레딧 배열 산출 ──────────────────────────────
    monthly_per_user = analysis.compute_monthly_per_user(user_stats, poc_days, working_day_ratio)

    # ── 분포 적합 ─────────────────────────────────────────
    dist_fit = fit_lognormal(monthly_per_user)
    if dist_fit.fit_quality == "poor":
        warnings.append(
            f"로그정규 분포 적합 불량 (KS p={dist_fit.ks_pvalue:.3f}). "
            "경험적 백분위수를 사용합니다. 사용 패턴이 매우 이질적일 수 있습니다."
        )
    elif dist_fit.fit_quality == "marginal":
        warnings.append(
            f"로그정규 분포 적합 보통 (KS p={dist_fit.ks_pvalue:.3f}). "
            "경험적 백분위수를 사용합니다."
        )

    # ── 세그먼트 통계 ─────────────────────────────────────
    segment_stats = analysis.compute_segment_stats(user_stats, poc_days, working_day_ratio)

    # ── 집단 백분위수 & 부트스트랩 CI ──────────────────────
    pop_pct = analysis.compute_population_percentiles(monthly_per_user)
    ci = analysis.bootstrap_ci(monthly_per_user, n_iterations=args.bootstrap_iterations)

    ci_width_ratio = (ci["p50_upper"] - ci["p50_lower"]) / pop_pct["p50"] if pop_pct["p50"] > 0 else 0
    if ci_width_ratio > 0.8:
        warnings.append(
            f"P50 부트스트랩 CI 폭이 P50의 {ci_width_ratio:.0%}. "
            "표본 분산이 커서 예측 불확실성이 높습니다."
        )

    if not args.no_tiers and args.credit_to_usd == 1.0:
        warnings.append(
            "티어 분석: --credit-to-usd 미설정(1.0). "
            "실제 크레딧 단가(예: 0.04)를 지정해야 티어 상한이 올바르게 적용됩니다."
        )

    # ── 시나리오 예측 ─────────────────────────────────────
    scenario_keys = [s.strip().lower() for s in args.scenarios.split(",")]
    scenarios = scaling.forecast_scenarios(
        population_percentiles=pop_pct,
        bootstrap_ci=ci,
        dist_fit=dist_fit,
        enterprise_users=args.enterprise_users,
        bias_factor=args.bias_factor,
        scenarios=scenario_keys,
        apply_ramp=not args.no_ramp,
    )

    if args.monte_carlo:
        scenarios = scaling.monte_carlo_ci(
            scenarios,
            population_percentiles=pop_pct,
            enterprise_users=args.enterprise_users,
            bias_factor=args.bias_factor,
            bias_sigma=args.mc_bias_sigma,
            n_sim=args.mc_iterations,
        )

    sensitivity = scaling.bias_sensitivity_table(
        population_percentiles=pop_pct,
        dist_fit=dist_fit,
        enterprise_users=args.enterprise_users,
        adoption_rate=0.45,
    )

    # ── Sanity check ──────────────────────────────────────
    total_credit = df["usage_credit"].sum()
    simple_monthly = (
        total_credit / poc_days * analysis.WORKING_DAYS_PER_MONTH
        / working_day_ratio
        / poc_days  # already in working days... simplify:
    )
    # 단순 비례: (total_credit / poc_working_days) * 22 * (enterprise / poc) / bias
    poc_working_days = poc_days * working_day_ratio
    simple_monthly = (total_credit / poc_working_days * analysis.WORKING_DAYS_PER_MONTH
                      * args.enterprise_users / poc_users / args.bias_factor)
    if scenarios:
        mod = next((s for s in scenarios if "Moderate" in s.name), scenarios[0])
        ratio = mod.monthly_p50 / simple_monthly if simple_monthly > 0 else 0
        if ratio > 3 or ratio < 0.1:
            warnings.append(
                f"평균 기반 추정 ({simple_monthly:,.0f} cr/월) vs P50 예측 ({mod.monthly_p50:,.0f} cr/월): "
                f"{1/ratio if ratio < 1 else ratio:.1f}배 차이. "
                "사용 분포가 heavy-tail이거나 편향 계수를 재검토하세요."
            )

    # ── 리포트 조립 ───────────────────────────────────────
    report = ForecastReport(
        poc_period_days=poc_days,
        poc_start=str(df["date_partition"].min().date()),
        poc_end=str(df["date_partition"].max().date()),
        poc_users=poc_users,
        enterprise_users=args.enterprise_users,
        bias_factor=args.bias_factor,
        total_poc_credit=total_credit,
        credit_to_usd=args.credit_to_usd,
        working_day_ratio=working_day_ratio,
        monthly_per_user=monthly_per_user.tolist(),
        dist_fit=dist_fit,
        usage_type_summary=df.groupby("usage_type")["usage_credit"].sum().to_dict(),
        usage_unit_summary=df.groupby("usage_unit")["usage_credit"].sum().to_dict(),
        segment_stats=segment_stats,
        department_usage=analysis.compute_department_usage(user_stats, monthly_per_user, args.credit_to_usd),
        scenarios=scenarios,
        sensitivity=sensitivity,
        warnings=warnings,
    )

    # ── 역할 기반 자동 추론 예측 ──────────────────────────
    from . import role_model
    report.role_scenarios = role_model.compute_role_scenarios(
        segment_stats=segment_stats,
        enterprise_users=args.enterprise_users,
        bias_factor=args.bias_factor,
        scenario_keys=scenario_keys,
        apply_ramp=not args.no_ramp,
    )

    # ── 부서별 티어 분배 권장 ─────────────────────────────
    if not args.no_tiers:
        from . import tier_model
        caps = [float(x) for x in args.tiers.split(",")]
        names = [x.strip() for x in args.tier_names.split(",")][:len(caps)]
        if len(names) < len(caps):
            names += [f"Tier{i+1}" for i in range(len(names), len(caps))]
        report.tier_defs = tier_model.build_tier_defs(caps, names, args.credit_to_usd)
        report.tier_allocations, report.tier_company = tier_model.compute_tier_allocations(
            user_stats, monthly_per_user, caps, names,
            args.credit_to_usd, args.bias_factor,
        )

    # ── 출력 ──────────────────────────────────────────────
    reporter.print_report(report, verbose=args.verbose)

    if args.output_csv:
        reporter.export_csv(report, args.output_csv)

    if args.output_html:
        from . import html_reporter
        html_reporter.generate(report, args.output_html)


if __name__ == "__main__":
    main()
