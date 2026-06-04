import argparse
import sys

from . import ingestion, analysis, scaling, reporter
from .schemas import ForecastReport


def main():
    parser = argparse.ArgumentParser(
        description="POC AI 크레딧 사용량 기반 전사 도입 비용 예측 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m poc_forecast poc_data.csv
  python -m poc_forecast poc_data.csv --bias-factor 3.0 --output-csv result.csv
  python -m poc_forecast poc_data.csv --credit-to-usd 0.01 --no-ramp --verbose
        """,
    )
    parser.add_argument("input_file", help="POC 사용량 CSV 또는 Excel 파일 경로")
    parser.add_argument("--enterprise-users", type=int, default=3000, metavar="N",
                        help="전사 총 인원 (기본: 3000)")
    parser.add_argument("--poc-users", type=int, default=None, metavar="N",
                        help="POC 실제 참여 인원 (미지정 시 CSV 고유 이메일 수 사용)")
    parser.add_argument("--bias-factor", type=float, default=2.5, metavar="X",
                        help="POC 편향 보정 계수 (기본: 2.5). POC 사용자가 일반 직원보다 X배 더 사용한다고 가정")
    parser.add_argument("--scenarios", default="conservative,moderate,aggressive",
                        help="실행할 시나리오 (기본: conservative,moderate,aggressive)")
    parser.add_argument("--credit-to-usd", type=float, default=1.0, metavar="RATE",
                        help="크레딧→USD 환산 비율 (기본: 1.0)")
    parser.add_argument("--output-csv", metavar="PATH",
                        help="결과를 CSV 파일로 저장")
    parser.add_argument("--no-ramp", action="store_true",
                        help="램프업 곡선 비적용 (연간 = 월간 × 12)")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000, metavar="N",
                        help="부트스트랩 반복 횟수 (기본: 2000)")
    parser.add_argument("--verbose", action="store_true",
                        help="세그먼트별 상세 출력")

    args = parser.parse_args()

    # --- Load data ---
    print(f"파일 로딩 중: {args.input_file}")
    try:
        df, warnings = ingestion.load(args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Build user stats ---
    user_stats, poc_days = analysis.build_user_stats(df)
    user_stats = analysis.assign_segments(user_stats)
    poc_users = args.poc_users or len(user_stats)

    # --- Sanity checks ---
    if poc_users < 10:
        warnings.append(f"POC 사용자 수가 {poc_users}명으로 너무 적어 예측 신뢰도가 낮습니다.")

    total_credit = df["usage_credit"].sum()
    avg_daily_credit_per_user = total_credit / poc_days / poc_users if poc_days * poc_users > 0 else 0
    if avg_daily_credit_per_user > 50:
        warnings.append(
            f"사용자당 일평균 크레딧이 {avg_daily_credit_per_user:.1f}로 높습니다. "
            "--credit-to-usd 설정을 확인하세요."
        )

    # --- Analysis ---
    segment_stats = analysis.compute_segment_stats(user_stats, poc_days)
    pop_pct = analysis.compute_population_percentiles(user_stats, poc_days)
    ci = analysis.bootstrap_ci(user_stats, poc_days, n_iterations=args.bootstrap_iterations)

    # CI width check
    if pop_pct["p50"] > 0:
        ci_width_ratio = (ci["p50_upper"] - ci["p50_lower"]) / pop_pct["p50"]
        if ci_width_ratio > 0.8:
            warnings.append(
                f"부트스트랩 CI 폭이 P50의 {ci_width_ratio:.0%}입니다. "
                "표본이 적거나 분산이 커서 예측 불확실성이 높습니다."
            )

    # --- Scaling ---
    scenario_keys = [s.strip().lower() for s in args.scenarios.split(",")]
    scenarios = scaling.forecast_scenarios(
        population_percentiles=pop_pct,
        bootstrap_ci=ci,
        enterprise_users=args.enterprise_users,
        bias_factor=args.bias_factor,
        scenarios=scenario_keys,
        apply_ramp=not args.no_ramp,
    )

    sensitivity = scaling.bias_sensitivity_table(
        population_percentiles=pop_pct,
        enterprise_users=args.enterprise_users,
        adoption_rate=0.45,
    )

    # Sanity check: simple proportional vs P50
    simple_monthly = (total_credit / poc_days * 30) * (args.enterprise_users / poc_users) / args.bias_factor
    if scenarios:
        mod = next((s for s in scenarios if "Moderate" in s.name), scenarios[0])
        ratio = mod.monthly_p50 / simple_monthly if simple_monthly > 0 else 0
        if ratio > 3 or ratio < 0.2:
            warnings.append(
                f"평균 기반 추정({simple_monthly:,.0f} cr/월)과 중앙값(P50) 예측({mod.monthly_p50:,.0f} cr/월) 간 "
                f"{1/ratio:.1f}배 차이 발생. 소수 헤비유저가 평균을 끌어올리는 heavy-tail 분포이거나, "
                "편향 계수를 재검토해야 할 수 있습니다."
            )

    # --- Build report ---
    usage_type_summary = (
        df.groupby("usage_type")["usage_credit"].sum().to_dict()
    )
    usage_unit_summary = (
        df.groupby("usage_unit")["usage_credit"].sum().to_dict()
    )

    report = ForecastReport(
        poc_period_days=poc_days,
        poc_start=str(df["date_partition"].min().date()),
        poc_end=str(df["date_partition"].max().date()),
        poc_users=poc_users,
        enterprise_users=args.enterprise_users,
        bias_factor=args.bias_factor,
        total_poc_credit=total_credit,
        credit_to_usd=args.credit_to_usd,
        usage_type_summary=usage_type_summary,
        usage_unit_summary=usage_unit_summary,
        segment_stats=segment_stats,
        scenarios=scenarios,
        warnings=warnings,
    )
    # Attach sensitivity for reporter
    report.sensitivity = sensitivity  # type: ignore[attr-defined]

    # --- Output ---
    reporter.print_report(report, verbose=args.verbose)

    if args.output_csv:
        reporter.export_csv(report, args.output_csv)


if __name__ == "__main__":
    main()
