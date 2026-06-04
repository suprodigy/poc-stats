"""HTML 리포트 생성 모듈 — 차트 6개 + 표 + 추론 사슬을 단일 HTML로 조합."""
from datetime import date as _date
from .schemas import ForecastReport
from .distribution import describe_fit
from .scaling import RAMP_MULTIPLIERS, RAMP_SUM
from . import visualizer as viz


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
       background: #f5f6fa; color: #2c3e50; font-size: 14px; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; color: #1a252f; margin-bottom: 6px; }
.meta { color: #7f8c8d; font-size: 12px; margin-bottom: 24px; }
h2 { font-size: 16px; color: #2c3e50; margin: 32px 0 12px;
     padding-bottom: 6px; border-bottom: 2px solid #3498db; }
h3 { font-size: 13px; color: #555; margin: 16px 0 8px; }
.cards { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.card { background: #fff; border-radius: 8px; padding: 18px 22px;
        flex: 1; min-width: 200px;
        box-shadow: 0 2px 8px rgba(0,0,0,.07); }
.card .label { font-size: 11px; color: #7f8c8d; text-transform: uppercase;
               letter-spacing: .5px; }
.card .value { font-size: 24px; font-weight: 700; color: #2c3e50; margin-top: 4px; }
.card .sub   { font-size: 11px; color: #95a5a6; margin-top: 2px; }
.chart-wrap { background: #fff; border-radius: 8px;
              padding: 16px; margin: 12px 0;
              box-shadow: 0 2px 8px rgba(0,0,0,.07); text-align: center; }
.chart-wrap img { max-width: 100%; }
.chart-note { font-size: 12px; color: #7f8c8d; margin-top: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 12px;
        background: #fff; border-radius: 8px; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,.07); }
th { background: #2c3e50; color: #fff; padding: 10px 14px; text-align: center; }
td { padding: 8px 14px; text-align: right; border-bottom: 1px solid #ecf0f1; }
td:first-child { text-align: left; font-weight: 500; }
tr:hover td { background: #f8f9fa; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
         font-size: 11px; font-weight: 600; }
.badge-good     { background: #d5f5e3; color: #1e8449; }
.badge-marginal { background: #fef9e7; color: #9a7d0a; }
.badge-poor     { background: #fde8e8; color: #922b21; }
.warn { background: #fff8e1; border-left: 4px solid #f39c12;
        padding: 10px 14px; margin: 8px 0; border-radius: 0 6px 6px 0;
        font-size: 12px; }
.chain { background: #fff; border-radius: 8px; padding: 20px 24px;
         box-shadow: 0 2px 8px rgba(0,0,0,.07); }
.chain-step { display: flex; gap: 14px; margin-bottom: 14px; }
.step-num { background: #3498db; color: #fff; border-radius: 50%;
            width: 24px; height: 24px; display: flex; align-items: center;
            justify-content: center; font-size: 11px; font-weight: 700;
            flex-shrink: 0; }
.step-body .title { font-weight: 600; font-size: 13px; }
.step-body .desc  { font-size: 12px; color: #555; margin-top: 3px; }
.ci { font-size: 10px; color: #7f8c8d; }
section { margin-bottom: 8px; }
"""


def _fmt(v: float, usd: float) -> str:
    if usd != 1.0:
        return f"${v * usd:,.0f}"
    return f"{v:,.1f}"


def _ci_span(lo: float, hi: float, usd: float) -> str:
    return f'<span class="ci">[{_fmt(lo, usd)} ~ {_fmt(hi, usd)}]</span>'


def generate(report: ForecastReport, path: str):
    usd = report.credit_to_usd
    unit = "USD" if usd != 1.0 else "크레딧"
    today = _date.today().isoformat()
    fit = report.dist_fit

    # --- 차트 생성 ---
    monthly_arr = __import__("numpy").array(report.monthly_per_user)
    c1 = viz.chart_user_distribution(monthly_arr, fit)
    c2 = viz.chart_segment_breakdown(report.segment_stats)
    c3 = viz.chart_service_breakdown(report.usage_type_summary)
    c4 = viz.chart_scenario_comparison(report.scenarios, usd)
    c5 = viz.chart_rampup_curve(report.scenarios, usd)
    c6 = viz.chart_bias_sensitivity(
        {"p50": report.monthly_per_user[len(report.monthly_per_user)//2],
         "p75": __import__("numpy").percentile(report.monthly_per_user, 75)},
        fit,
        report.enterprise_users,
        report.bias_factor,
        usd,
    )

    # --- Moderate 시나리오 (요약 카드용) ---
    mod = next((s for s in report.scenarios if "Moderate" in s.name), report.scenarios[0]) if report.scenarios else None

    # --- 경고 블록 ---
    warn_html = ""
    for w in report.warnings:
        warn_html += f'<div class="warn">⚠ {w}</div>\n'

    # --- 요약 카드 ---
    poc_cr = f"{report.total_poc_credit:,.0f} 크레딧"
    mod_monthly = _fmt(mod.monthly_p50, usd) if mod else "-"
    mod_annual  = _fmt(mod.annual_p75,  usd) if mod else "-"

    cards_html = f"""
<div class="cards">
  <div class="card">
    <div class="label">POC 총 크레딧</div>
    <div class="value">{poc_cr}</div>
    <div class="sub">{report.poc_users}명 · {report.poc_period_days}일간</div>
  </div>
  <div class="card">
    <div class="label">전사 예측 월간 (Moderate P50)</div>
    <div class="value">{mod_monthly}</div>
    <div class="sub">채택률 45% · {unit} 기준</div>
  </div>
  <div class="card">
    <div class="label">전사 예측 연간 (Moderate P75)</div>
    <div class="value">{mod_annual}</div>
    <div class="sub">램프업 포함 · {unit} 기준</div>
  </div>
  <div class="card">
    <div class="label">분포 적합 품질</div>
    <div class="value" style="font-size:18px">
      <span class="badge badge-{fit.fit_quality}">
        {"양호" if fit.fit_quality=="good" else "보통" if fit.fit_quality=="marginal" else "불량"}
      </span>
    </div>
    <div class="sub">KS p={fit.ks_pvalue:.3f} · {'로그정규 사용' if fit.fit_quality=='good' else '경험적 사용'}</div>
  </div>
</div>"""

    # --- 세그먼트 표 ---
    seg_rows = ""
    for s in report.segment_stats:
        seg_rows += f"""
<tr>
  <td>{s.name}</td>
  <td>{s.user_count}</td>
  <td>{s.active_day_ratio_mean:.0%}</td>
  <td>{_fmt(s.credit_per_user_per_day_p50, usd)}</td>
  <td>{_fmt(s.monthly_credit_per_user_p50, usd)}</td>
  <td>{_fmt(s.monthly_credit_per_user_p75, usd)}</td>
  <td>{_fmt(s.monthly_credit_per_user_p95, usd)}</td>
  <td>{s.credit_share_pct:.1f}%</td>
</tr>"""

    seg_table = f"""
<table>
<tr><th>세그먼트</th><th>인원</th><th>활성일%</th>
    <th>일평균P50</th><th>월P50</th><th>월P75</th><th>월P95</th><th>크레딧비율</th></tr>
{seg_rows}
</table>"""

    # --- 비용 예측 표 ---
    def _sc_col(s, pct_attr, ci_lo, ci_hi):
        return f"{_fmt(getattr(s, pct_attr), usd)}<br>{_ci_span(ci_lo * usd if usd == 1.0 else ci_lo, ci_hi * usd if usd == 1.0 else ci_hi, usd)}"

    forecast_rows = ""
    for s in report.scenarios:
        forecast_rows += f"""
<tr>
  <td>{s.name} ({s.adoption_rate:.0%})<br><small>{s.active_users:,}명 활성</small></td>
  <td>{_fmt(s.monthly_p50, usd)}<br>{_ci_span(s.ci_p50_lower, s.ci_p50_upper, usd)}</td>
  <td>{_fmt(s.monthly_p75, usd)}<br>{_ci_span(s.ci_p75_lower, s.ci_p75_upper, usd)}</td>
  <td>{_fmt(s.monthly_p95, usd)}<br>{_ci_span(s.ci_p95_lower, s.ci_p95_upper, usd)}</td>
  <td>{_fmt(s.annual_p50, usd)}</td>
  <td>{_fmt(s.annual_p75, usd)}</td>
</tr>"""

    forecast_table = f"""
<table>
<tr><th>시나리오</th>
    <th>월간 P50<br><small>(95% CI)</small></th>
    <th>월간 P75<br><small>(95% CI)</small></th>
    <th>월간 P95<br><small>(95% CI)</small></th>
    <th>연간 P50<br><small>(램프업포함)</small></th>
    <th>연간 P75<br><small>(램프업포함)</small></th></tr>
{forecast_rows}
</table>
<p style="font-size:11px;color:#7f8c8d;margin-top:6px">단위: {unit} · CI = 95% 부트스트랩 신뢰구간</p>"""

    # --- 민감도 표 ---
    sens_rows = ""
    if report.sensitivity:
        for r in report.sensitivity:
            sens_rows += f"""
<tr>
  <td>{r['bias_factor']}x</td>
  <td>{_fmt(r['credit_per_user_monthly_p50'], usd)}</td>
  <td>{_fmt(r['credit_per_user_monthly_p75'], usd)}</td>
  <td>{_fmt(r['total_monthly_p50'], usd)}</td>
  <td>{_fmt(r['total_monthly_p75'], usd)}</td>
</tr>"""

    sens_table = f"""
<table>
<tr><th>편향 계수</th>
    <th>사용자당 P50</th><th>사용자당 P75</th>
    <th>전체 P50 (Moderate)</th><th>전체 P75 (Moderate)</th></tr>
{sens_rows}
</table>""" if sens_rows else ""

    # --- 추론 사슬 ---
    dist_source = "로그정규 분포 기반 백분위수" if fit.fit_quality == "good" else "경험적 백분위수 (표본 직접 계산)"
    chain_html = f"""
<div class="chain">
  <div class="chain-step">
    <div class="step-num">1</div>
    <div class="step-body">
      <div class="title">POC 관측 데이터 수집</div>
      <div class="desc">{report.poc_users}명 · {report.poc_period_days}일간 크레딧 사용량 집계.
        월간 환산 기준: <strong>근무일 22일/월</strong>, 실제 활동일 비율 {report.working_day_ratio:.1%} 반영.</div>
    </div>
  </div>
  <div class="chain-step">
    <div class="step-num">2</div>
    <div class="step-body">
      <div class="title">분포 적합 검정</div>
      <div class="desc">사용자별 월 크레딧에 로그정규 분포 적합.
        <strong>{describe_fit(fit)}</strong><br>
        결과: <strong>{dist_source}</strong> 사용.
        로그정규 선택 근거: 엔터프라이즈 소프트웨어 사용량은 소수 파워유저가 대부분을 차지하는
        heavy-tail 특성으로, 로그정규 분포가 경험적으로 잘 맞는다.</div>
    </div>
  </div>
  <div class="chain-step">
    <div class="step-num">3</div>
    <div class="step-body">
      <div class="title">POC 선발 편향 보정</div>
      <div class="desc">POC 참가자는 자발적·열성적 사용자로 일반 직원보다 더 많이 사용.
        현재 설정: <strong>편향 계수 {report.bias_factor}x</strong> (POC 사용량을 bias_factor로 나눔).
        타당 범위: 1.5x (POC ≈ 일반 대표) ~ 4.0x (열성 얼리어답터).
        이 계수가 가장 큰 불확실성 요인이므로 민감도 분석 필수.</div>
    </div>
  </div>
  <div class="chain-step">
    <div class="step-num">4</div>
    <div class="step-body">
      <div class="title">채택률 시나리오 적용</div>
      <div class="desc">전체 {report.enterprise_users:,}명 중 실제 AI를 사용할 비율:
        Conservative 20% / Moderate 45% / Aggressive 70%.
        활성 사용자 수 = 전사 인원 × 채택률.</div>
    </div>
  </div>
  <div class="chain-step">
    <div class="step-num">5</div>
    <div class="step-body">
      <div class="title">월간 비용 산출</div>
      <div class="desc">월간 비용 = (보정된 사용자당 월 크레딧 Pxx) × 활성 사용자 수.
        P50 = 중앙값 (대표 시나리오), P75 = 예산 계획용, P95 = 최악 시나리오.
        각 백분위수에 95% 부트스트랩 신뢰구간 제공.</div>
    </div>
  </div>
  <div class="chain-step">
    <div class="step-num">6</div>
    <div class="step-body">
      <div class="title">연간 비용 및 램프업</div>
      <div class="desc">전사 도입 후 즉시 정상 운영되지 않음.
        1~2월 30%, 3~4월 60%, 5~6월 80%, 7~12월 100%.
        연간 비용 = 월간 비용 × {RAMP_SUM:.2f} (= 12개월 램프업 가중합).
        스테디스테이트 기준 연간 = 월간 × 12.</div>
    </div>
  </div>
</div>"""

    # --- 전체 HTML 조합 ---
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>POC AI 크레딧 비용 예측 리포트</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>POC AI 크레딧 사용량 분석 및 전사 도입 비용 예측</h1>
  <div class="meta">생성일: {today} &nbsp;|&nbsp;
    POC: {report.poc_users}명, {report.poc_start} ~ {report.poc_end} ({report.poc_period_days}일) &nbsp;|&nbsp;
    전사: {report.enterprise_users:,}명 &nbsp;|&nbsp;
    편향 계수: {report.bias_factor}x &nbsp;|&nbsp;
    단위: {unit}
  </div>

  {warn_html}
  {cards_html}

  <section id="distribution">
    <h2>1. POC 사용자 크레딧 분포</h2>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c1}" alt="사용자 크레딧 분포">
      <div class="chart-note">
        로그정규 분포 적합 결과: {describe_fit(fit)}<br>
        수직 점선은 각 백분위수(P50/P75/P95) 위치.
        곡선이 히스토그램에 잘 맞을수록 분포 기반 예측의 신뢰도가 높아진다.
      </div>
    </div>
  </section>

  <section id="segmentation">
    <h2>2. 사용자 세분화</h2>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c2}" alt="사용자 세분화">
      <div class="chart-note">
        POC 기간 총 크레딧 기준 상위 20% = Heavy, 중간 20% = Medium, 하위 60% = Light.
        크레딧 점유율은 Heavy 사용자가 대부분을 차지하는 것이 일반적이다.
      </div>
    </div>
    {seg_table}
  </section>

  <section id="services">
    <h2>3. 서비스별 크레딧 사용</h2>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c3}" alt="서비스별 사용량">
      <div class="chart-note">usage_type별 POC 기간 총 크레딧 소비량 (상위 10개 표시).</div>
    </div>
  </section>

  <section id="forecast">
    <h2>4. 전사 도입 비용 예측</h2>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c4}" alt="시나리오별 비용">
      <div class="chart-note">
        P50 = 중앙값(대표 케이스), P75 = 예산 계획용, P95 = 최악 케이스.
        막대 위 수치는 월간 비용.
      </div>
    </div>
    {forecast_table}
  </section>

  <section id="rampup">
    <h2>5. 도입 첫 12개월 비용 추이</h2>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c5}" alt="램프업 곡선">
      <div class="chart-note">
        점선은 각 시나리오의 스테디스테이트(정상 운영) 수준.
        회색 배경: 램프업 구간(1~6월), 파란 배경: 정상 운영 구간(7~12월).
      </div>
    </div>
  </section>

  <section id="sensitivity">
    <h2>6. 편향 계수 민감도 분석</h2>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c6}" alt="민감도 분석">
      <div class="chart-note">
        편향 계수는 가장 큰 불확실성 요인. 녹색 구간(1.5x~4.0x)이 일반적으로 합리적 범위.
        수직 점선 = 현재 설정({report.bias_factor}x).
      </div>
    </div>
    {sens_table}
  </section>

  <section id="assumptions">
    <h2>7. 예측 추론 사슬 및 가정</h2>
    {chain_html}
  </section>

</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 리포트 저장: {path}")
