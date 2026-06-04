"""HTML 리포트 생성 모듈 — 차트 6개 + 표 + 추론 사슬을 단일 HTML로 조합."""
from datetime import date as _date
import numpy as np
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

/* 도입부 안내 */
.intro { background: #fff; border-radius: 8px; padding: 20px 24px;
         box-shadow: 0 2px 8px rgba(0,0,0,.07); margin-bottom: 24px; }
.intro h3 { margin-top: 0; color: #2c3e50; font-size: 14px; }
.intro ol { margin: 8px 0 16px 20px; font-size: 13px; color: #444; }
.intro ol li { margin-bottom: 4px; }
.flow { display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
        margin-top: 12px; }
.flow-step { background: #eaf2fb; color: #2471a3; border-radius: 6px;
             padding: 7px 12px; font-size: 12px; font-weight: 600; }
.flow-arrow { color: #aab7c4; font-weight: 700; font-size: 14px; }

/* 쉽게 말하면 콜아웃 */
.plain { background: #eef7fd; border-left: 4px solid #3498db;
         border-radius: 0 6px 6px 0; padding: 12px 16px; margin: 10px 0;
         font-size: 13px; line-height: 1.6; color: #34495e; }
.plain .tag { font-weight: 700; color: #2471a3; margin-right: 6px; }

/* 더 알아보기 토글 */
details.toggle { background: #fafbfc; border: 1px solid #e4e8eb;
                 border-radius: 6px; margin: 8px 0; padding: 0; }
details.toggle summary { cursor: pointer; padding: 10px 14px;
                 font-size: 12px; font-weight: 600; color: #5a6c7d;
                 list-style: none; user-select: none; }
details.toggle summary::-webkit-details-marker { display: none; }
details.toggle summary::before { content: "▶ "; color: #95a5a6; font-size: 10px; }
details.toggle[open] summary::before { content: "▼ "; }
details.toggle summary:hover { color: #2c3e50; background: #f0f3f5; }
details.toggle .body { padding: 4px 16px 14px; font-size: 12.5px;
                 line-height: 1.7; color: #555; border-top: 1px solid #eef1f3; }
details.toggle .body p { margin: 8px 0; }
details.toggle .body strong { color: #2c3e50; }

/* 실제 데이터 예시 박스 */
.example { background: #fffdf5; border: 1px dashed #d4b483;
           border-radius: 6px; padding: 12px 16px; margin: 10px 0;
           font-size: 13px; line-height: 1.6; color: #6b5638; }
.example .tag { display: inline-block; background: #d4b483; color: #fff;
                font-size: 10px; font-weight: 700; padding: 1px 8px;
                border-radius: 10px; margin-right: 8px; }
.example b { color: #b8860b; }

/* 결론 — 예산 범위 바 */
.concl-range { position: relative; height: 56px; margin: 24px 0 12px;
               background: linear-gradient(90deg,#d5f5e3 0%,#fef9e7 45%,#fdeecf 75%,#fde8e8 100%);
               border-radius: 28px; }
.concl-range .marker { position: absolute; top: -6px; transform: translateX(-50%);
               text-align: center; }
.concl-range .marker .dot { width: 14px; height: 14px; border-radius: 50%;
               background: #2c3e50; margin: 0 auto 2px;
               border: 3px solid #fff; box-shadow: 0 1px 3px rgba(0,0,0,.3); }
.concl-range .marker .lbl { font-size: 10px; color: #2c3e50; font-weight: 600;
               white-space: nowrap; }
.concl-range .band { position: absolute; top: 18px; height: 20px;
               background: rgba(243,156,18,.35); border: 2px solid #f39c12;
               border-radius: 10px; }
.concl-cards { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.concl-card { flex: 1; min-width: 180px; border-radius: 8px; padding: 16px 18px;
              box-shadow: 0 2px 8px rgba(0,0,0,.07); }
.concl-card.low  { background: #f0faf4; border-top: 3px solid #27ae60; }
.concl-card.rec  { background: #fffaf0; border-top: 3px solid #f39c12; }
.concl-card.high { background: #fdf2f2; border-top: 3px solid #e74c3c; }
.concl-card .ttl { font-size: 12px; font-weight: 700; color: #555; }
.concl-card .num { font-size: 20px; font-weight: 800; color: #2c3e50; margin: 6px 0 2px; }
.concl-card .num small { font-size: 12px; font-weight: 600; color: #7f8c8d; }
.concl-card .dsc { font-size: 11px; color: #7f8c8d; line-height: 1.5; }
.rec-box { background: #2c3e50; color: #fff; border-radius: 8px;
           padding: 16px 20px; font-size: 14px; line-height: 1.7; margin: 16px 0; }
.rec-box strong { color: #f6d97a; }
.caution { background: #fff; border-radius: 8px; padding: 16px 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,.07); }
.caution ul { margin: 4px 0 0 18px; font-size: 12.5px; line-height: 1.8; color: #555; }
.footnote { font-size: 11px; color: #95a5a6; margin-top: 10px; line-height: 1.6; }
"""


def _fmt(v: float, usd: float) -> str:
    if usd != 1.0:
        return f"${v * usd:,.0f}"
    return f"{v:,.1f}"


def _ci_span(lo: float, hi: float, usd: float) -> str:
    return f'<span class="ci">[{_fmt(lo, usd)} ~ {_fmt(hi, usd)}]</span>'


def _plain(text: str) -> str:
    """💡 쉽게 말하면 콜아웃 박스."""
    return f'<div class="plain"><span class="tag">💡 쉽게 말하면</span>{text}</div>'


def _toggle(summary: str, body_html: str) -> str:
    """▶ 더 알아보기 접이식 토글 (HTML 네이티브 <details>, JS 불필요)."""
    return (f'<details class="toggle"><summary>{summary}</summary>'
            f'<div class="body">{body_html}</div></details>')


def _example(html: str) -> str:
    """실제 데이터 예시 박스."""
    return f'<div class="example"><span class="tag">이 회사 예시</span>{html}</div>'


def _build_conclusion(cons, modr, aggr, usd: float, unit: str) -> str:
    """범위형 결론/권장 예산 섹션 HTML 생성."""
    if not (cons and modr and aggr):
        return ""

    # 월간 기준 범위 구성
    low_m   = cons.monthly_p50          # 현실적 하한
    rec_lo  = modr.monthly_p50          # 권장 범위 하단
    rec_hi  = modr.monthly_p75          # 권장 범위 상단
    high_m  = aggr.monthly_p95          # 상방 리스크
    # 연간 (램프업 포함)
    low_a   = cons.annual_p50
    rec_lo_a = modr.annual_p50
    rec_hi_a = modr.annual_p75
    high_a  = aggr.annual_p95

    # 범위 바 위 마커 위치 (0~100%) — 로그 스케일로 배치해 가독성 확보
    import math
    vals = [low_m, rec_lo, rec_hi, high_m]
    lo_log, hi_log = math.log(max(low_m, 1)), math.log(max(high_m, 1))
    span = (hi_log - lo_log) or 1.0
    def _pos(v):
        return (math.log(max(v, 1)) - lo_log) / span * 90 + 5  # 5~95% 안쪽 배치

    band_left = _pos(rec_lo)
    band_right = _pos(rec_hi)

    markers = (
        f'<div class="marker" style="left:{_pos(low_m):.1f}%">'
        f'<div class="dot"></div><div class="lbl">최소<br>{_fmt(low_m, usd)}</div></div>'
        f'<div class="marker" style="left:{_pos(high_m):.1f}%">'
        f'<div class="dot"></div><div class="lbl">최대<br>{_fmt(high_m, usd)}</div></div>'
    )
    band = f'<div class="band" style="left:{band_left:.1f}%;width:{max(band_right-band_left,3):.1f}%"></div>'

    return f"""
<div class="concl-cards">
  <div class="concl-card low">
    <div class="ttl">🟢 최소 예상 (낙관)</div>
    <div class="num">{_fmt(low_m, usd)}<small> /월</small></div>
    <div class="dsc">연 {_fmt(low_a, usd)} · Conservative 채택률 20%, 대표 사용량(P50) 기준.
      도입이 더디고 사용량이 평이할 때.</div>
  </div>
  <div class="concl-card rec">
    <div class="ttl">🟡 권장 예산 범위</div>
    <div class="num">{_fmt(rec_lo, usd)} ~ {_fmt(rec_hi, usd)}<small> /월</small></div>
    <div class="dsc">연 {_fmt(rec_lo_a, usd)} ~ {_fmt(rec_hi_a, usd)} · Moderate 채택률 45%,
      대표값(P50)~예산버퍼(P75). <b>실제 편성 권장 구간.</b></div>
  </div>
  <div class="concl-card high">
    <div class="ttl">🔴 최대 리스크 (대비)</div>
    <div class="num">{_fmt(high_m, usd)}<small> /월</small></div>
    <div class="dsc">연 {_fmt(high_a, usd)} · Aggressive 채택률 70%, 최악(P95) 기준.
      전사 빠른 확산 + 헤비유저 폭증 시.</div>
  </div>
</div>

<div class="concl-range">
  {band}
  {markers}
</div>
<div style="text-align:center;font-size:11px;color:#95a5a6;margin-bottom:18px">
  ↑ 월간 비용 스펙트럼 (가로축: 로그 스케일). 주황 구간 = 권장 예산 범위.
</div>

<div class="rec-box">
  📌 <strong>권장</strong>: 월 예산은 <strong>{_fmt(rec_lo, usd)} ~ {_fmt(rec_hi, usd)} {unit}</strong>
  (Moderate P50~P75) 범위로 편성하고,
  급격한 확산에 대비해 <strong>{_fmt(high_m, usd)} {unit}</strong>(Aggressive P95)까지
  감당할 여력을 확보하는 것을 권장합니다.
</div>

<div class="caution">
  <div style="font-weight:700;font-size:13px;color:#2c3e50;margin-bottom:4px">⚠ 예측 해석 시 주의사항</div>
  <ul>
    <li><strong>편향 계수가 가장 큰 변수입니다.</strong> 이 값을 1.5x↔4.0x로 바꾸면 예측이 2~3배 출렁입니다 (섹션 6 참고).
        POC 참가자가 일반 직원을 얼마나 대표하는지 조직 특성에 맞게 검증하세요.</li>
    <li><strong>POC 기간이 짧으면 신뢰구간이 넓어집니다.</strong> 표본/기간이 작을수록 불확실성이 커지므로,
        가능하면 더 긴 데이터로 재예측하세요.</li>
    <li><strong>도입 후 실제 채택률·사용량을 모니터링하며 분기별로 재예측</strong>하는 것을 권장합니다.
        이 리포트는 의사결정의 출발점이지 확정 수치가 아닙니다.</li>
  </ul>
</div>"""


def generate(report: ForecastReport, path: str):
    usd = report.credit_to_usd
    unit = "USD" if usd != 1.0 else "크레딧"
    today = _date.today().isoformat()
    fit = report.dist_fit

    # --- 예시용 실제 통계값 (사용자별 월 크레딧 기준) ---
    monthly_arr = np.array(report.monthly_per_user)
    ex_p50 = float(np.percentile(monthly_arr, 50))
    ex_p75 = float(np.percentile(monthly_arr, 75))
    ex_p95 = float(np.percentile(monthly_arr, 95))
    ex_mean = float(np.mean(monthly_arr))
    ex_max = float(np.max(monthly_arr))
    mean_vs_p50 = ex_mean / ex_p50 if ex_p50 > 0 else 1.0

    # --- 차트 생성 ---
    c1 = viz.chart_user_distribution(monthly_arr, fit)
    c2 = viz.chart_segment_breakdown(report.segment_stats)
    c3 = viz.chart_service_breakdown(report.usage_type_summary)
    c4 = viz.chart_scenario_comparison(report.scenarios, usd)
    c5 = viz.chart_rampup_curve(report.scenarios, usd)
    c6 = viz.chart_bias_sensitivity(
        {"p50": ex_p50, "p75": ex_p75},
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

    # --- 도입부: 이 리포트 읽는 법 ---
    intro_html = """
<div class="intro">
  <h3>📋 이 리포트는 다음 3가지 질문에 답합니다</h3>
  <ol>
    <li><strong>전사에 도입하면 매달/매년 비용이 얼마나 들까?</strong> — POC 실측 데이터를 전사 규모로 확대 추정</li>
    <li><strong>그 숫자는 얼마나 믿을 수 있나?</strong> — 신뢰구간과 분포 적합 검정으로 불확실성을 함께 제시</li>
    <li><strong>그래서 예산은 얼마로 잡아야 하나?</strong> — 안전마진을 포함한 권장 예산 <em>범위</em> 제시</li>
  </ol>
  <div style="font-size:12px;color:#7f8c8d;margin-bottom:6px">분석은 아래 순서로 진행됩니다 (각 단계는 해당 번호 섹션에서 자세히 설명):</div>
  <div class="flow">
    <span class="flow-step">① POC 실측</span><span class="flow-arrow">→</span>
    <span class="flow-step">② 사용자 분포 분석</span><span class="flow-arrow">→</span>
    <span class="flow-step">③ POC 편향 보정</span><span class="flow-arrow">→</span>
    <span class="flow-step">④ 시나리오별 확대</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑤ 비용 범위 결론</span>
  </div>
  <div style="font-size:11.5px;color:#95a5a6;margin-top:10px">
    💡 파란 박스(쉽게 말하면)는 핵심을 쉬운 말로, <b>▶ 더 알아보기</b>를 누르면 통계 개념을 자세히 볼 수 있습니다.
  </div>
</div>"""

    # --- 결론 섹션 데이터 ---
    sc_by = {s.name: s for s in report.scenarios}
    cons = sc_by.get("Conservative")
    modr = sc_by.get("Moderate")
    aggr = sc_by.get("Aggressive")
    conclusion_html = _build_conclusion(cons, modr, aggr, usd, unit)

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
  {intro_html}
  {cards_html}

  <section id="distribution">
    <h2>1. POC 사용자 크레딧 분포</h2>
    {_plain("100명이 한 달에 얼마나 쓰는지를 한 장에 모은 그림입니다. 막대(히스토그램)는 <b>비슷한 사용량을 가진 사람이 몇 명인지</b>를 나타냅니다. 대부분은 적게 쓰고 소수가 아주 많이 쓰는, 한쪽으로 긴 꼬리 모양이 전형적입니다.")}
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c1}" alt="사용자 크레딧 분포">
      <div class="chart-note">
        빨간 곡선 = 데이터에 맞춘 로그정규 분포. 수직 점선 = P50/P75/P95 위치.
        곡선이 막대에 잘 겹칠수록 분포 기반 예측의 신뢰도가 높습니다.
      </div>
    </div>
    {_toggle("로그정규분포가 뭔가요? μ·σ·KS검정은?",
        "<p><strong>로그정규분포</strong>는 '값이 0보다 작아질 수 없고, 한쪽(오른쪽)으로 길게 늘어지는' 데이터에 잘 맞는 분포입니다. "
        "사용량, 소득, 파일 크기처럼 '대부분은 작고 소수가 매우 큰' 데이터가 여기에 해당합니다.</p>"
        "<p><strong>μ(뮤)</strong>는 분포의 중심 위치, <strong>σ(시그마)</strong>는 퍼진 정도를 나타냅니다. σ가 클수록 사용자 간 편차가 큽니다.</p>"
        "<p><strong>KS 검정</strong>은 '실제 데이터가 이 분포 모양과 비슷한가'를 0~1 사이 p값으로 평가합니다. "
        "<strong>p > 0.05면 '이 분포로 봐도 무리 없다'</strong>는 뜻이고, 이때 표본 100명을 넘어서는 일반화가 더 안전해집니다.</p>")}
    {_example(f"이 회사 POC에서 사용자별 월 사용량은 <b>절반(P50)이 {ex_p50:,.0f} 크레딧 이하</b>, "
        f"<b>상위 5%(P95)는 {ex_p95:,.0f} 크레딧 이상</b>을 썼습니다. "
        f"가장 많이 쓴 사람은 약 {ex_max:,.0f} 크레딧으로, 전형적인 '롱테일' 모양입니다.")}
  </section>

  <section id="segmentation">
    <h2>2. 사용자 세분화</h2>
    {_plain("사용자를 사용량 순으로 줄 세운 뒤 <b>상위 20%=Heavy, 다음 20%=Medium, 하위 60%=Light</b> 세 그룹으로 나눴습니다. 그룹마다 비용 기여도가 크게 다르기 때문입니다.")}
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c2}" alt="사용자 세분화">
      <div class="chart-note">
        왼쪽 = 그룹별 인원 비율, 오른쪽 = 그룹별 크레딧 점유율.
        소수의 Heavy 그룹이 전체 비용의 대부분을 차지하는 경우가 많습니다.
      </div>
    </div>
    {seg_table}
    {_toggle("왜 소수가 대부분을 쓰나요? (파레토 법칙)",
        "<p>대부분의 소프트웨어에서 <strong>소수 핵심 사용자가 전체 사용량의 대부분</strong>을 차지합니다 "
        "(이른바 80/20 파레토 법칙). AI 도구도 마찬가지로, 일부가 자동화·반복 작업에 집중적으로 활용합니다.</p>"
        f"<p>이 회사도 Heavy 그룹(상위 20%)이 전체 크레딧의 "
        f"<strong>{report.segment_stats[0].credit_share_pct:.0f}%</strong>를 사용했습니다. "
        "따라서 '평균 사용자'로 비용을 추정하면 이 쏠림을 놓치게 되어, 백분위수 기반 접근이 필요합니다.</p>")}
  </section>

  <section id="services">
    <h2>3. 서비스별 크레딧 사용</h2>
    {_plain("어떤 AI 모델·서비스(GPT-4o, 이미지 생성 등)에 크레딧이 얼마나 쓰였는지 보여줍니다. 비싼 모델에 사용이 몰리면 같은 사용량이라도 비용이 커집니다.")}
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c3}" alt="서비스별 사용량">
      <div class="chart-note">usage_type별 POC 기간 총 크레딧 소비량 (상위 10개).</div>
    </div>
  </section>

  <section id="forecast">
    <h2>4. 전사 도입 비용 예측</h2>
    {_plain("여기서 <b>P50·P75·P95</b>가 핵심입니다. <b>백분위수</b>란 '100명을 사용량 순으로 한 줄로 세웠을 때 몇 번째 사람인가'를 뜻합니다.<br>"
        "&nbsp;&nbsp;• <b>P50</b> = 50번째 사람 = <b>중앙값</b>(딱 중간). 대표적인 보통 사용자.<br>"
        "&nbsp;&nbsp;• <b>P75</b> = 75번째 사람 = 상위 25% 진입선. <b>예산을 넉넉히 잡을 때</b> 기준.<br>"
        "&nbsp;&nbsp;• <b>P95</b> = 95번째 사람 = 거의 최상위. <b>최악의 경우</b>를 대비할 때 기준.")}
    {_example(f"이 회사 POC에서 사용자를 월 사용량 순으로 세우면 — "
        f"<b>50번째 사람 ≈ {ex_p50:,.0f}</b>, <b>75번째 ≈ {ex_p75:,.0f}</b>, "
        f"<b>95번째 ≈ {ex_p95:,.0f} 크레딧</b>이었습니다. "
        f"전사 비용은 이 '사용자 1명당 값'에 편향 보정과 사용자 수를 곱해 계산합니다.")}
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c4}" alt="시나리오별 비용">
      <div class="chart-note">
        막대 = 시나리오별 월간 비용. 같은 시나리오 안에서도 P50→P95로 갈수록
        '더 보수적인(비싼) 가정'이라 비용이 올라갑니다.
      </div>
    </div>
    {forecast_table}
    {_toggle("왜 평균이 아니라 중앙값(P50)을 쓰나요?",
        "<p><strong>평균은 소수의 헤비유저 때문에 부풀려집니다.</strong> 예를 들어 9명이 1을 쓰고 1명이 100을 쓰면 "
        "평균은 10.9지만, 실제 '보통 사람'은 1을 씁니다. 이때 중앙값(P50)이 현실을 더 잘 대표합니다.</p>"
        f"<p>이 회사도 <strong>평균 {ex_mean:,.0f}</strong> vs <strong>중앙값(P50) {ex_p50:,.0f}</strong>으로, "
        f"평균이 중앙값의 약 <strong>{mean_vs_p50:.1f}배</strong>입니다. "
        "평균으로 곱하면 전사 비용을 과대평가할 위험이 있어, 본 리포트는 백분위수를 기준으로 삼습니다.</p>")}
    {_toggle("신뢰구간(95% CI)과 부트스트랩이란?",
        "<p>POC 표본은 100명뿐이라 '진짜 전사 값'을 정확히는 알 수 없습니다. "
        "<strong>95% 신뢰구간(CI)</strong>은 '같은 조사를 여러 번 반복하면 약 95%는 이 범위 안에 들어온다'는 뜻으로, "
        "<strong>숫자 하나가 아니라 '범위'로 불확실성을 정직하게 보여주는</strong> 장치입니다.</p>"
        "<p>이 범위는 <strong>부트스트랩</strong>이라는 방법으로 구합니다. 가진 데이터에서 무작위로 100명을 "
        "여러 번(2000회) 다시 뽑아 매번 값을 계산하고, 그 결과들이 퍼진 정도로 불확실성을 추정합니다. "
        "표(위)의 각 P값 아래 <span class='ci'>[ ~ ]</span>가 바로 이 95% 신뢰구간입니다. "
        "구간이 넓을수록 표본이 작거나 편차가 크다는 신호입니다.</p>")}
  </section>

  <section id="rampup">
    <h2>5. 도입 첫 12개월 비용 추이</h2>
    {_plain("전사 도입 첫날부터 3천 명이 한꺼번에 쓰진 않습니다. 교육·온보딩을 거쳐 <b>약 6개월에 걸쳐 점진적으로 정착</b>하는 현실을 반영했습니다. 그래서 1년차 비용은 '월 비용 × 12'보다 작습니다.")}
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c5}" alt="램프업 곡선">
      <div class="chart-note">
        점선 = 각 시나리오의 정상 운영(스테디스테이트) 수준.
        회색 배경 = 램프업 구간(1~6월), 파란 배경 = 정상 운영(7~12월).
      </div>
    </div>
    {_toggle("램프업 가중치는 어떻게 정했나요?",
        "<p>일반적인 엔터프라이즈 소프트웨어 도입은 S자 곡선을 그립니다. 본 모델은 "
        "<strong>1~2개월 30% → 3~4개월 60% → 5~6개월 80% → 7개월 이후 100%</strong>로 단계적 정착을 가정했습니다.</p>"
        f"<p>12개월 가중치 합 = <strong>{RAMP_SUM:.2f}</strong>이므로, 1년차 연간 비용 = 월 비용 × {RAMP_SUM:.2f}입니다 "
        "(정착 후 기준 연간은 월 비용 × 12). 도입 속도가 더 빠르다면 이 가중치를 올려 재계산할 수 있습니다.</p>")}
  </section>

  <section id="sensitivity">
    <h2>6. 편향 계수 민감도 분석</h2>
    {_plain("<b>편향 계수</b>는 'POC 참가자가 일반 직원보다 몇 배 더 쓰는가'에 대한 추정값이라 본질적으로 불확실합니다. 이 값을 바꾸면 비용이 얼마나 출렁이는지 보여줍니다 — <b>가장 조심해서 검증해야 할 가정</b>입니다.")}
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c6}" alt="민감도 분석">
      <div class="chart-note">
        녹색 구간(1.5x~4.0x) = 일반적으로 합리적인 범위.
        수직 점선 = 현재 설정({report.bias_factor}x). 곡선이 가파를수록 이 가정에 비용이 민감합니다.
      </div>
    </div>
    {sens_table}
  </section>

  <section id="conclusion">
    <h2>7. 결론 및 권장 예산</h2>
    {_plain("앞의 분석을 종합한 <b>예산 권장안</b>입니다. 단일 숫자가 아니라 <b>안전마진을 포함한 범위</b>로 제시합니다 — 예측에는 불확실성이 있으므로 범위로 대비하는 것이 안전합니다.")}
    {conclusion_html}
  </section>

  <section id="assumptions">
    <h2>8. 예측 추론 사슬 및 가정</h2>
    {_plain("위 결론이 어떤 논리로 도출됐는지 6단계로 정리했습니다. 각 단계의 가정을 바꾸면 결과도 달라지므로, 조직 상황에 맞게 검토하세요.")}
    {chain_html}
  </section>

</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 리포트 저장: {path}")
