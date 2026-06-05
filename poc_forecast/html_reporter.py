"""HTML 리포트 생성 모듈 — 주장/논거 구조 임원 설득형 보고서."""
from datetime import date as _date
import math
import numpy as np
from .schemas import ForecastReport
from .scaling import RAMP_MULTIPLIERS, RAMP_SUM
from .distribution import WORKING_DAYS_PER_MONTH
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

/* Executive Summary */
.exec-summ { background: #1a252f; color: #ecf0f1; border-radius: 10px;
             padding: 24px 28px; margin-bottom: 28px; }
.exec-summ h2 { color: #f6d97a; border: none; margin: 0 0 14px;
                font-size: 15px; padding-bottom: 0; }
.exec-summ .find { display: flex; gap: 10px; margin: 8px 0;
                   font-size: 13.5px; line-height: 1.6; }
.exec-summ .find .icon { flex-shrink: 0; }
.exec-summ .cred { font-size: 11.5px; color: #aab7c4; margin-top: 10px;
                   padding-top: 10px; border-top: 1px solid rgba(255,255,255,.1); }
.exec-summ .rec-line { background: rgba(246,217,122,.15); border: 1px solid #f6d97a;
             border-radius: 6px; padding: 12px 16px; margin-top: 14px;
             font-size: 14px; font-weight: 600; color: #f6d97a; line-height: 1.7; }

/* 도입부 안내 */
.intro { background: #fff; border-radius: 8px; padding: 20px 24px;
         box-shadow: 0 2px 8px rgba(0,0,0,.07); margin-bottom: 24px; }
.intro h3 { margin-top: 0; color: #2c3e50; font-size: 14px; }
.flow { display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
        margin-top: 12px; }
.flow-step { background: #eaf2fb; color: #2471a3; border-radius: 6px;
             padding: 7px 12px; font-size: 12px; font-weight: 600; }
.flow-arrow { color: #aab7c4; font-weight: 700; font-size: 14px; }

/* 주장(Claim) 박스 */
.claim { background: #f0faf4; border-left: 4px solid #27ae60;
         border-radius: 0 6px 6px 0; padding: 12px 16px; margin: 10px 0;
         font-size: 13px; line-height: 1.6; color: #1e5631; }
.claim .tag { font-weight: 700; color: #1e8449; margin-right: 6px; }

/* 쉽게 말하면 콜아웃 */
.plain { background: #eef7fd; border-left: 4px solid #3498db;
         border-radius: 0 6px 6px 0; padding: 12px 16px; margin: 10px 0;
         font-size: 13px; line-height: 1.6; color: #34495e; }
.plain .tag { font-weight: 700; color: #2471a3; margin-right: 6px; }

/* 시사점(Implication) 박스 */
.impl { background: #fdfefe; border: 1px solid #d5dbdb;
        border-radius: 6px; padding: 10px 14px; margin: 10px 0;
        font-size: 12.5px; color: #555; line-height: 1.6; }
.impl::before { content: "💬 시사점: "; font-weight: 700; color: #2c3e50; }

/* 섹션 간 이동 링크 */
.section-link { font-size: 12px; color: #95a5a6; margin: 14px 0 4px;
                text-align: right; }
.section-link::before { content: "→ "; }

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

/* 결론 방어 근거 박스 */
.defense { background: #fff; border-radius: 8px; padding: 18px 22px;
           box-shadow: 0 2px 8px rgba(0,0,0,.08);
           border-top: 3px solid #27ae60; margin: 16px 0; }
.defense h4 { font-size: 13px; color: #1e8449; margin-bottom: 10px; }
.defense ol { margin-left: 18px; font-size: 13px; line-height: 1.9; color: #444; }
.defense ol li { margin-bottom: 2px; }

.footnote { font-size: 11px; color: #95a5a6; margin-top: 10px; line-height: 1.6; }
.appendix-note { font-size: 12px; color: #7f8c8d; background: #f8f9fa;
                 border-radius: 6px; padding: 8px 14px; margin: 8px 0; }

/* Step-by-Step 계산 흐름 */
.sbs-step { display: flex; gap: 14px; background: #fff; border-radius: 8px;
            padding: 16px 20px; margin: 0; box-shadow: 0 2px 8px rgba(0,0,0,.07);
            border-left: 4px solid #3498db; }
.sbs-num { background: #3498db; color: #fff; border-radius: 50%; width: 30px; height: 30px;
           flex-shrink: 0; display: flex; align-items: center; justify-content: center;
           font-weight: 700; font-size: 13px; }
.sbs-body { flex: 1; }
.sbs-body .t { font-weight: 700; font-size: 14px; color: #2c3e50; }
.sbs-body .d { font-size: 12.5px; color: #555; line-height: 1.6; margin-top: 4px; }
.sbs-calc { background: #f4f8fb; border-radius: 6px; padding: 8px 12px; margin-top: 8px;
            font-size: 12.5px; font-family: 'Consolas','Monaco',monospace; color: #34495e;
            line-height: 1.7; word-break: break-word; }
.sbs-result { display: inline-block; background: #eafaf1; border: 1px solid #27ae60;
              color: #1e8449; font-weight: 700; border-radius: 6px;
              padding: 3px 10px; margin-top: 8px; font-size: 13px; }
.sbs-arrow { text-align: center; color: #bcc6cf; font-size: 18px; margin: 2px 0; line-height: 1; }
.sbs-why { font-size: 11.5px; color: #7f8c8d; margin-top: 6px; font-style: italic; }
"""


def _fmt(v: float, usd: float) -> str:
    if usd != 1.0:
        return f"${v * usd:,.0f}"
    return f"{v:,.1f}"


def _ci_span(lo: float, hi: float, usd: float) -> str:
    return f'<span class="ci">[{_fmt(lo, usd)} ~ {_fmt(hi, usd)}]</span>'


def _plain(text: str) -> str:
    return f'<div class="plain"><span class="tag">💡 쉽게 말하면</span>{text}</div>'


def _claim(text: str) -> str:
    return f'<div class="claim"><span class="tag">✅ 주장</span>{text}</div>'


def _impl(text: str) -> str:
    return f'<div class="impl">{text}</div>'


def _toggle(summary: str, body_html: str) -> str:
    return (f'<details class="toggle"><summary>{summary}</summary>'
            f'<div class="body">{body_html}</div></details>')


def _example(html: str) -> str:
    return f'<div class="example"><span class="tag">이 회사 예시</span>{html}</div>'


def _section_link(text: str) -> str:
    return f'<p class="section-link">{text}</p>'


_SBS_ARROW = '<div class="sbs-arrow">▼</div>'


def _step(n, title: str, desc: str, calc: str = None,
          result: str = None, why: str = None) -> str:
    """Step-by-Step 계산 흐름의 단일 단계 카드."""
    calc_html   = f'<div class="sbs-calc">{calc}</div>' if calc else ""
    result_html = f'<div class="sbs-result">→ {result}</div>' if result else ""
    why_html    = f'<div class="sbs-why">{why}</div>' if why else ""
    return (f'<div class="sbs-step"><div class="sbs-num">{n}</div>'
            f'<div class="sbs-body"><div class="t">{title}</div>'
            f'<div class="d">{desc}</div>{calc_html}{result_html}{why_html}</div></div>')


def _build_steps(
    report: ForecastReport, fit, usd: float, unit: str,
    stats: dict, modr, aggr, has_mc: bool,
    c1: str, c2: str, c3: str, c4: str, c5: str, c6: str,
    seg_table: str, forecast_table: str, sens_table: str,
    role_table: str = "",
) -> str:
    """원자료 → 기초 통계 → 분포 → 보정 → 전사 확대 → 결론까지의 단계별 유도 과정.
    각 단계 카드 아래에 관련 차트·표·설명을 직접 삽입한다."""
    f = lambda v: _fmt(v, usd)
    quality_kr = {"good": "양호", "marginal": "보통", "poor": "불량"}[fit.fit_quality]
    use_dist = fit.fit_quality == "good"
    src_p50 = fit.p50_dist if use_dist else stats["ex_p50"]
    src_label = "분포 기반" if use_dist else "경험적"
    bf = report.bias_factor
    corrected_p50 = src_p50 / bf
    active = modr.active_users if modr else 0
    heavy_share = report.segment_stats[0].credit_share_pct if report.segment_stats else 0

    def _chart(img_b64: str, note: str) -> str:
        return (f'<div class="chart-wrap">'
                f'<img src="data:image/png;base64,{img_b64}" alt="">'
                f'<div class="chart-note">{note}</div></div>')

    blocks = []

    # ── Step 1: 원자료 수집 ────────────────────────────────────────────
    b = _step(
        1, "원자료 수집",
        f"POC 참가자 {report.poc_users}명이 {report.poc_period_days}일간 남긴 "
        "<b>usage_credit</b> 사용 로그를 서비스(usage_type)별로 집계합니다.",
        calc=f"Σ(전체 사용 로그의 usage_credit) = <b>{report.total_poc_credit:,.0f}</b> 크레딧",
        result=f"POC 총 크레딧 {report.total_poc_credit:,.0f} cr",
        why="모든 추정의 출발점은 가정이 아닌 실측 데이터입니다.",
    )
    b += _chart(c3,
        "usage_type별 POC 기간 총 크레딧 소비량 (상위 10개). "
        "비싼 모델에 사용이 몰릴수록 같은 사용량이라도 비용이 커집니다.")
    b += _plain("GPT-4o·이미지 생성 등 서비스별로 크레딧이 얼마나 소비됐는지 보여줍니다. "
                "어떤 서비스가 비용을 주도하는지 파악하면 향후 요금제 협상이나 사용 가이드 설계에 활용할 수 있습니다.")
    blocks.append(b)

    # ── Step 2: 사용자별 집계 ──────────────────────────────────────────
    b = _step(
        2, "사용자별 집계",
        f"로그를 이메일(사용자) 단위로 묶어 {report.poc_users}명 각각의 총 크레딧을 구합니다. "
        "분석 단위는 '개별 로그'가 아니라 '사용자 1명'입니다.",
        calc=f"사용자당 평균(단순) = {report.total_poc_credit:,.0f} ÷ {report.poc_users} "
             f"= <b>{stats['raw_avg_per_user']:,.1f}</b> cr/인 (POC 기간 전체)",
        result=f"사용자별 총 크레딧 {report.poc_users}개 값",
        why="비용은 사용자 수에 비례하므로 사용자 단위로 봐야 전사 확대가 가능합니다.",
    )
    b += _plain("이 단계에서 분석 단위가 '로그 행'에서 '사람'으로 바뀝니다. "
                "이후 모든 계산은 사용자 1명이 한 달에 평균 얼마를 쓰는지를 기준으로 진행됩니다.")
    blocks.append(b)

    # ── Step 3: 기초 기술통계 ─────────────────────────────────────────
    b = _step(
        3, "기초 기술통계",
        "사용자별 값의 분포를 평균·중앙값·표준편차·범위로 요약합니다. "
        "(아래 숫자는 5단계 월간 환산 기준)",
        calc=f"평균 {f(stats['ex_mean'])} · 중앙값(P50) {f(stats['ex_p50'])} · "
             f"표준편차 {f(stats['ex_std'])}<br>"
             f"최소 {f(stats['ex_min'])} · 최대 {f(stats['ex_max'])} (단위: {unit}/월)",
        result=f"평균 {f(stats['ex_mean'])} vs 중앙값 {f(stats['ex_p50'])}",
        why="평균과 중앙값이 크게 다르면 분포가 한쪽으로 치우쳤다는 신호입니다.",
    )
    b += _chart(c1,
        "빨간 곡선 = 데이터에 맞춘 로그정규 분포. 수직 점선 = P50/P75/P95 위치. "
        "곡선이 막대에 잘 겹칠수록 분포 기반 예측의 신뢰도가 높습니다.")
    b += _plain("히스토그램은 비슷한 사용량을 가진 사람이 몇 명인지 막대로 보여줍니다. "
                "대부분은 적게 쓰고 소수가 아주 많이 쓰는, 오른쪽으로 긴 꼬리 모양이 전형적입니다.")
    b += _chart(c2,
        "왼쪽 = 그룹별 인원 비율, 오른쪽 = 그룹별 크레딧 점유율. "
        "소수의 Heavy 그룹이 전체 비용의 대부분을 차지합니다.")
    b += seg_table
    b += _toggle("왜 소수가 대부분을 쓰나요? (파레토 법칙)",
        "<p>대부분의 소프트웨어에서 <strong>소수 핵심 사용자가 전체 사용량의 대부분</strong>을 차지합니다 "
        "(80/20 파레토 법칙). AI 도구도 마찬가지로, 일부가 자동화·반복 작업에 집중적으로 활용합니다.</p>"
        f"<p>이 회사도 Heavy 그룹(상위 20%)이 전체 크레딧의 "
        f"<strong>{heavy_share:.0f}%</strong>를 사용했습니다. "
        "따라서 '평균 사용자'로 비용을 추정하면 이 쏠림을 놓치게 됩니다.</p>")
    blocks.append(b)

    # ── Step 4: 평균의 함정 ───────────────────────────────────────────
    b = _step(
        4, "평균의 함정 발견 — 왜 백분위수인가",
        "평균이 중앙값보다 크게 높고 표준편차가 평균에 육박하면, 소수 헤비유저가 "
        "평균을 끌어올린 <b>우편향(heavy-tail)</b> 분포입니다.",
        calc=f"평균 ÷ 중앙값 = {stats['ex_mean']:,.0f} ÷ {stats['ex_p50']:,.0f} "
             f"= <b>{stats['mean_vs_p50']:.1f}배</b> &nbsp;|&nbsp; "
             f"변동계수 CV = 표준편차 ÷ 평균 = <b>{stats['ex_cv']:.2f}</b>",
        result="평균은 과대평가 → 백분위수(P50/P75/P95) 채택",
        why="평균으로 전사 비용을 곱하면 실제보다 부풀려집니다. 이것이 백분위수를 쓰는 핵심 근거입니다.",
    )
    b += _toggle("왜 평균이 아니라 중앙값(P50)을 쓰나요?",
        "<p><strong>평균은 소수의 헤비유저 때문에 부풀려집니다.</strong> 9명이 1을 쓰고 1명이 100을 쓰면 "
        "평균은 10.9지만, 실제 보통 사람은 1을 씁니다. 중앙값(P50)이 현실을 더 잘 대표합니다.</p>"
        f"<p>이 회사도 <strong>평균 {stats['ex_mean']:,.0f}</strong> vs <strong>중앙값 {stats['ex_p50']:,.0f}</strong>으로, "
        f"평균이 중앙값의 약 <strong>{stats['mean_vs_p50']:.1f}배</strong>입니다. "
        "평균으로 곱하면 전사 비용을 과대평가할 위험이 있어, 본 리포트는 백분위수를 기준으로 삼습니다.</p>")
    blocks.append(b)

    # ── Step 5: 월간 정규화 ───────────────────────────────────────────
    b = _step(
        5, "월간 정규화",
        f"POC는 {report.poc_period_days}일(캘린더 기준)이지만 비용은 '월' 단위로 관리합니다. "
        "실 활동일을 반영해 월 근무일 22일 기준으로 환산합니다.",
        calc=f"scale = 22 ÷ ({report.poc_period_days}일 × 활동일비율 {report.working_day_ratio:.0%}) "
             f"= <b>{stats['monthly_scale']:.3f}</b><br>"
             f"사용자별 월 크레딧 = 사용자별 총 크레딧 × {stats['monthly_scale']:.3f}",
        result=f"사용자별 '월 크레딧' 배열 {report.poc_users}개",
        why="기간이 다른 POC들을 공정하게 비교하고 월 예산에 직접 대응시키기 위함입니다.",
    )
    b += _plain(f"POC가 {report.poc_period_days}일이라도, 실제 사용이 있던 날은 그 중 "
                f"{report.working_day_ratio:.0%}뿐입니다. 이 비율을 반영해 '월 22 근무일' 기준으로 통일합니다. "
                "이후 모든 백분위수와 예측은 이 월간 환산값을 기준으로 합니다.")
    blocks.append(b)

    # ── Step 6: 백분위수 산출 ─────────────────────────────────────────
    b = _step(
        6, "백분위수 산출",
        "월 크레딧을 작은 값부터 정렬해 P50/P75/P95 위치의 값을 읽습니다. "
        "각각 대표값·예산값·최악값에 해당합니다.",
        calc=f"P50(중앙값) {f(stats['ex_p50'])} · P75(예산선) {f(stats['ex_p75'])} · "
             f"P95(최악) {f(stats['ex_p95'])} &nbsp;<i>(경험적, {unit}/월·사용자당)</i>",
        result=f"사용자 1명당 P50 {f(stats['ex_p50'])}",
        why="단일 평균 대신 3개 지점으로 '보통~최악'의 폭을 함께 봅니다.",
    )
    b += _example(
        f"이 회사 POC에서 월 사용량 순위를 매기면 — "
        f"<b>50번째 사람 ≈ {stats['ex_p50']:,.0f}</b>, <b>75번째 ≈ {stats['ex_p75']:,.0f}</b>, "
        f"<b>95번째 ≈ {stats['ex_p95']:,.0f} 크레딧</b>이었습니다. "
        "전사 비용은 이 '사용자 1명당 값'에 편향 보정과 사용자 수를 곱해 계산합니다.")
    b += _plain("<b>P50·P75·P95</b>는 100명을 사용량 순으로 한 줄로 세웠을 때 몇 번째 사람인지를 뜻합니다. "
                "P50 = 50번째(중앙값, 보통 사람). P75 = 75번째(예산을 넉넉히 잡을 때). P95 = 95번째(최악 대비).")
    blocks.append(b)

    # ── Step 7: 분포 적합 ────────────────────────────────────────────
    b = _step(
        7, "분포 적합 (로그정규)",
        "표본 100명을 넘어선 일반화를 위해 사용량에 로그정규 분포를 적합하고, "
        "KS 검정으로 적합 품질을 객관적으로 평가합니다.",
        calc=f"적합 모수 μ={fit.mu:.2f}, σ={fit.sigma:.2f} · KS p={fit.ks_pvalue:.3f} "
             f"[<b>{quality_kr}</b>]<br>"
             f"분포 기반 백분위수 P_q = exp(μ + z_q·σ) → "
             f"P50 {f(fit.p50_dist)} · P75 {f(fit.p75_dist)} · P95 {f(fit.p95_dist)}",
        result=(f"<b>{src_label}</b> 백분위수 채택 (P50 {f(src_p50)})"),
        why=("KS p>0.05라 분포 적합이 양호 → 분포 기반 백분위수 사용 (표본 외 극단값까지 안정적 추정)"
             if use_dist else
             "KS p<0.05라 적합이 불충분 → 경험적 백분위수 사용 (표본 직접 계산)"),
    )
    b += _toggle("로그정규분포가 뭔가요? μ·σ·KS검정은?",
        "<p><strong>로그정규분포</strong>는 '값이 0보다 작아질 수 없고, 한쪽(오른쪽)으로 길게 늘어지는' 데이터에 잘 맞는 분포입니다. "
        "사용량, 소득, 파일 크기처럼 '대부분은 작고 소수가 매우 큰' 데이터가 여기에 해당합니다.</p>"
        "<p><strong>μ(뮤)</strong>는 분포의 중심 위치, <strong>σ(시그마)</strong>는 퍼진 정도를 나타냅니다. σ가 클수록 사용자 간 편차가 큽니다.</p>"
        "<p><strong>KS 검정</strong>은 '실제 데이터가 이 분포 모양과 비슷한가'를 0~1 사이 p값으로 평가합니다. "
        "<strong>p > 0.05면 이 분포로 봐도 무리 없다</strong>는 뜻이고, 이때 표본을 넘어선 일반화가 더 안전해집니다.</p>")
    blocks.append(b)

    # ── Step 8: 부트스트랩 ────────────────────────────────────────────
    bci = (modr.ci_p50_lower, modr.ci_p50_upper) if modr else (0, 0)
    b = _step(
        8, "표본 불확실성 추정 (부트스트랩)",
        f"표본이 {report.poc_users}명뿐이라 백분위수 자체에 추정 오차가 있습니다. "
        "데이터를 2000회 재추출해 그 오차 범위를 정량화합니다.",
        calc=f"부트스트랩 2000회 → Moderate 월간 P50의 95% 신뢰구간 = "
             f"[{f(bci[0])} ~ {f(bci[1])}]",
        result=f"P50 95% CI [{f(bci[0])} ~ {f(bci[1])}]",
        why="'점 추정 하나'가 아니라 '범위'로 표본 한계를 정직하게 드러냅니다.",
    )
    b += _toggle("신뢰구간(95% CI)과 부트스트랩이란?",
        "<p>POC 표본은 100명뿐이라 '진짜 전사 값'을 정확히는 알 수 없습니다. "
        "<strong>95% 신뢰구간(CI)</strong>은 '같은 조사를 여러 번 반복하면 약 95%는 이 범위 안에 들어온다'는 뜻으로, "
        "숫자 하나가 아닌 '범위'로 불확실성을 정직하게 보여주는 장치입니다.</p>"
        "<p><strong>부트스트랩</strong>은 가진 데이터에서 무작위로 100명을 2000회 다시 뽑아 매번 값을 계산하고, "
        "그 결과들이 퍼진 정도로 불확실성을 추정합니다. "
        "표의 각 P값 아래 <span class='ci'>[ ~ ]</span>가 바로 이 95% 신뢰구간입니다.</p>")
    blocks.append(b)

    # ── Step 9: 편향 보정 ────────────────────────────────────────────
    b = _step(
        9, "POC 선발 편향 보정",
        "POC 참가자는 자발적·열성적 사용자라 일반 직원보다 많이 씁니다. "
        f"편향 계수 {bf}x로 나눠 일반 직원 수준으로 낮춥니다.",
        calc=f"보정된 사용자당 P50 = {f(src_p50)} ÷ {bf} = <b>{f(corrected_p50)}</b> {unit}/월",
        result=f"일반 직원 1명당 P50 {f(corrected_p50)}",
        why="이 계수가 예측의 가장 큰 변수입니다 — 아래 민감도 분석으로 범위를 검증합니다.",
    )
    b += _plain(f"<b>편향 계수</b>는 'POC 참가자가 일반 직원보다 몇 배 더 쓰는가'에 대한 추정값으로, "
                "불확실성이 가장 높은 가정입니다. 아래 차트는 이 값을 바꾸면 비용이 어떻게 달라지는지 보여줍니다.")
    b += _chart(c6,
        f"녹색 구간(1.5x~4.0x) = 합리적인 범위. 수직 점선 = 현재 설정({bf}x). "
        "곡선이 가파를수록 이 가정에 비용이 민감합니다.")
    b += sens_table
    if has_mc:
        b += _toggle("Bootstrap CI vs Monte Carlo CI — 무엇이 다른가요?",
            "<p><strong>Bootstrap CI</strong>는 '표본 100명에서 P50/P75/P95를 추정할 때 생기는 오차'만 반영합니다. "
            "즉, bias_factor=2.5가 정확하다고 가정하고 사용자 샘플링 오차만 측정합니다.</p>"
            "<p><strong>Monte Carlo CI</strong>는 여기에 두 가지 불확실성을 추가합니다:<br>"
            "① <b>bias_factor</b>: 2.5x가 맞는지 확실하지 않으므로 LogNormal 분포로 모델링 (중앙값 2.5, 90% 범위 ≈ 1.4x~4.5x)<br>"
            "② <b>채택률</b>: 45%가 맞는지 확실하지 않으므로 Beta 분포로 모델링 (중앙 45%, 범위 10~80%)</p>"
            "<p>결과적으로 Monte Carlo CI는 Bootstrap CI보다 훨씬 넓습니다. "
            "이것이 '틀린' 것이 아니라 <strong>더 정직한 불확실성 표현</strong>입니다.</p>")
    blocks.append(b)

    # ── Step 10: 전사 확대 + 시나리오 ────────────────────────────────
    b = _step(
        10, "전사 확대 + 채택률 시나리오",
        f"보정된 1명당 값에 전사 {report.enterprise_users:,}명과 채택률을 곱합니다. "
        "채택률은 Conservative 20% / Moderate 45% / Aggressive 70% 3가지.",
        calc=f"Moderate: {f(corrected_p50)} × ({report.enterprise_users:,}명 × 45%={active:,}명) "
             f"= <b>{f(modr.monthly_p50) if modr else '-'}</b> {unit}/월",
        result=f"Moderate 월간 P50 {f(modr.monthly_p50) if modr else '-'}",
        why="POC 1명의 사용량을 전사 규모 비용으로 확대하는 핵심 단계입니다.",
    )
    b += _chart(c4,
        "막대 = 시나리오별 월간 비용. 같은 시나리오 안에서도 P50→P95로 갈수록 비용이 올라갑니다.")
    b += forecast_table
    if role_table:
        b += role_table
        b += _toggle("역할 기반 vs 기존 방법 — 무엇이 다른가요?",
            "<b>기존(로그정규)</b>: 전체 사용자 분포를 로그정규로 fitting → 백분위수 → bias 보정. "
            "Bootstrap CI가 채택률 불확실성과 함께 넓은 범위를 만듭니다.<br>"
            "<b>역할 기반</b>: Heavy/Medium/Light 세그먼트별 실측 사용량 중앙값 × 세그먼트 인원 수. "
            "같은 시나리오 안에서 P50~P95 범위가 훨씬 좁고 계산이 직관적입니다.")
    blocks.append(b)

    # ── Step 11: 몬테카를로 (조건부) ─────────────────────────────────
    has_mc_result = has_mc and modr and modr.mc_ci_p50_lower is not None
    if has_mc_result:
        b = _step(
            11, "파라미터 불확실성 (몬테카를로)",
            "편향 계수와 채택률 '자체'도 불확실합니다. 두 값을 분포로 두고 5000회 "
            "시뮬레이션해 더 솔직한 범위를 구합니다.",
            calc="편향계수 ~ LogNormal, 채택률 ~ Beta · 5000회<br>"
                 f"Moderate 월간 P50의 95% CI = "
                 f"[{f(modr.mc_ci_p50_lower)} ~ {f(modr.mc_ci_p50_upper)}]",
            result=f"MC P50 95% CI [{f(modr.mc_ci_p50_lower)} ~ {f(modr.mc_ci_p50_upper)}]",
            why="8단계(표본 오차)보다 넓어집니다 — 가정의 불확실성까지 더했기 때문입니다.",
        )
        b += _plain("Monte Carlo는 '우리가 설정한 값(편향 계수 2.5, 채택률 45%) 자체가 틀릴 수 있다'는 가능성까지 반영합니다. "
                    "부트스트랩 CI보다 훨씬 넓은 범위가 나오는데, 이것이 더 정직한 불확실성 표현입니다.")
        blocks.append(b)

    # ── Step 11/12: 연간화 ───────────────────────────────────────────
    n = 12 if has_mc_result else 11
    b = _step(
        n, "연간화 (램프업 반영)",
        "전사 도입 첫해는 6개월에 걸쳐 점진적으로 정착합니다. 월 비용에 12개월 "
        "램프업 가중합을 곱해 1년차 비용을 구합니다.",
        calc=f"연간 = 월간 × {RAMP_SUM:.2f} (1~2월 30%·3~4월 60%·5~6월 80%·7~12월 100%)<br>"
             f"Moderate 연간 P50 = {f(modr.monthly_p50) if modr else '-'} × {RAMP_SUM:.2f} "
             f"= <b>{f(modr.annual_p50) if modr else '-'}</b>",
        result=f"Moderate 연간 P50 {f(modr.annual_p50) if modr else '-'}",
        why="첫해는 '월×12'보다 작습니다. 도입 현실을 반영한 보정입니다.",
    )
    b += _chart(c5,
        "점선 = 정상 운영(스테디스테이트) 수준. "
        "회색 배경 = 램프업 구간(1~6월), 파란 배경 = 정상 운영(7~12월).")
    b += _plain("전사 도입 첫날부터 모든 직원이 한꺼번에 사용하진 않습니다. "
                "교육·온보딩을 거쳐 <b>약 6개월에 걸쳐 점진적으로 정착</b>하는 현실을 반영했습니다. "
                "그래서 1년차 비용은 '월 비용 × 12'보다 작습니다.")
    b += _toggle("램프업 가중치는 어떻게 정했나요?",
        "<p>일반적인 엔터프라이즈 소프트웨어 도입은 S자 곡선을 그립니다. 본 모델은 "
        "<strong>1~2개월 30% → 3~4개월 60% → 5~6개월 80% → 7개월 이후 100%</strong>로 단계적 정착을 가정했습니다.</p>"
        f"<p>12개월 가중치 합 = <strong>{RAMP_SUM:.2f}</strong>이므로, 1년차 연간 비용 = 월 비용 × {RAMP_SUM:.2f}입니다 "
        "(정착 후 기준 연간은 월 비용 × 12). 도입 속도가 더 빠르다면 가중치를 올려 재계산할 수 있습니다.</p>")
    blocks.append(b)

    # ── Step N+1: 결론 도출 ───────────────────────────────────────────
    b = _step(
        n + 1, "결론 — 예산 범위 도출",
        "위 과정을 종합해 단일 숫자가 아닌 <b>방어 가능한 예산 범위</b>로 정리합니다.",
        calc=f"권장 월 예산 = Moderate P50~P75 = "
             f"<b>{f(modr.monthly_p50) if modr else '-'} ~ {f(modr.monthly_p75) if modr else '-'}</b><br>"
             f"최대 대비 = Aggressive P95 = {f(aggr.monthly_p95) if aggr else '-'} {unit}/월",
        result="아래 '결론' 섹션의 권장 예산 범위",
        why="각 단계가 근거를 한 층씩 쌓아 이 결론을 뒷받침합니다.",
    )
    b += _section_link("이 과정을 거쳐 도출된 최종 예산 권장안을 아래 결론 섹션에서 확인하세요.")
    blocks.append(b)

    return _SBS_ARROW.join(blocks)


def _build_exec_summary(cons, modr, aggr, usd: float, unit: str,
                        poc_users: int, poc_days: int, fit) -> str:
    if not (modr and aggr):
        return ""
    rec_lo = _fmt(modr.monthly_p50, usd)
    rec_hi = _fmt(modr.monthly_p75, usd)
    high_m = _fmt(aggr.monthly_p95, usd)
    rec_lo_a = _fmt(modr.annual_p50, usd)
    rec_hi_a = _fmt(modr.annual_p75, usd)
    fit_label = "로그정규 분포 적합 양호" if fit.fit_quality == "good" else \
                "로그정규 분포 적합 보통 (경험적 백분위 사용)" if fit.fit_quality == "marginal" else \
                "분포 적합 불량 (경험적 백분위 사용)"

    return f"""
<div class="exec-summ">
  <h2>📊 핵심 결론 — 이것만 기억하세요</h2>
  <div class="find">
    <span class="icon">①</span>
    <span>POC 참가자 <strong>{poc_users}명</strong>의 <strong>{poc_days}일</strong> 실측 데이터를 분석한 결과,
    사용량은 소수 헤비유저가 대부분을 차지하는 <strong>롱테일 분포</strong>를 따릅니다.
    단순 평균이 아닌 <strong>통계적 분포 모델</strong>로 전사 비용을 추정했습니다.</span>
  </div>
  <div class="find">
    <span class="icon">②</span>
    <span>전사 도입 시 <strong>월간 권장 예산은 {rec_lo} ~ {rec_hi} {unit}</strong> (채택률 45%, 대표값~예산버퍼 기준).
    연간으로는 도입 초기 램프업을 포함해 <strong>{rec_lo_a} ~ {rec_hi_a} {unit}</strong> 수준입니다.</span>
  </div>
  <div class="find">
    <span class="icon">③</span>
    <span>급격한 전사 확산 시나리오(채택률 70%, P95 기준) 최대 <strong>{high_m} {unit}/월</strong>까지 감당할
    여력을 별도로 확보하는 것을 권장합니다.</span>
  </div>
  <div class="rec-line">
    📌 권장: 월 <strong>{rec_lo} ~ {rec_hi} {unit}</strong>으로 예산을 편성하고,
    최대 <strong>{high_m} {unit}</strong>까지 대비하십시오.
  </div>
  <div class="cred">
    신뢰도 근거: {poc_users}명 {poc_days}일 실측 · {fit_label} · 95% 부트스트랩 신뢰구간 포함
  </div>
</div>"""


def _build_conclusion(cons, modr, aggr, usd: float, unit: str,
                      bias_factor: float, poc_users: int, poc_days: int, fit) -> str:
    if not (cons and modr and aggr):
        return ""

    low_m    = cons.monthly_p50
    rec_lo   = modr.monthly_p50
    rec_hi   = modr.monthly_p75
    high_m   = aggr.monthly_p95
    low_a    = cons.annual_p50
    rec_lo_a = modr.annual_p50
    rec_hi_a = modr.annual_p75
    high_a   = aggr.annual_p95

    lo_log = math.log(max(low_m, 1))
    hi_log = math.log(max(high_m, 1))
    span = (hi_log - lo_log) or 1.0

    def _pos(v):
        return (math.log(max(v, 1)) - lo_log) / span * 90 + 5

    band_left  = _pos(rec_lo)
    band_right = _pos(rec_hi)

    markers = (
        f'<div class="marker" style="left:{_pos(low_m):.1f}%">'
        f'<div class="dot"></div><div class="lbl">최소<br>{_fmt(low_m, usd)}</div></div>'
        f'<div class="marker" style="left:{_pos(high_m):.1f}%">'
        f'<div class="dot"></div><div class="lbl">최대<br>{_fmt(high_m, usd)}</div></div>'
    )
    band = (f'<div class="band" style="left:{band_left:.1f}%;'
            f'width:{max(band_right - band_left, 3):.1f}%"></div>')

    fit_label = "양호 (로그정규 분포 채택)" if fit.fit_quality == "good" else \
                "보통 (경험적 백분위 사용)" if fit.fit_quality == "marginal" else "불량"

    return f"""
<div class="concl-cards">
  <div class="concl-card low">
    <div class="ttl">🟢 최소 예상 (낙관)</div>
    <div class="num">{_fmt(low_m, usd)}<small> /월</small></div>
    <div class="dsc">연 {_fmt(low_a, usd)} · Conservative 채택률 20%, P50 기준.
      도입이 더디고 사용량이 평이할 때.</div>
  </div>
  <div class="concl-card rec">
    <div class="ttl">🟡 권장 예산 범위</div>
    <div class="num">{_fmt(rec_lo, usd)} ~ {_fmt(rec_hi, usd)}<small> /월</small></div>
    <div class="dsc">연 {_fmt(rec_lo_a, usd)} ~ {_fmt(rec_hi_a, usd)} · Moderate 채택률 45%,
      P50~P75. <b>실제 편성 권장 구간.</b></div>
  </div>
  <div class="concl-card high">
    <div class="ttl">🔴 최대 리스크 (대비)</div>
    <div class="num">{_fmt(high_m, usd)}<small> /월</small></div>
    <div class="dsc">연 {_fmt(high_a, usd)} · Aggressive 채택률 70%, P95 기준.
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

<div class="defense">
  <h4>✅ 왜 이 예산 권장안이 합리적인가 — 5가지 근거</h4>
  <ol>
    <li><strong>실측 데이터 기반</strong>: 추정이나 벤치마크가 아닌, 자사 POC {poc_users}명 {poc_days}일의 실제 크레딧 사용 기록에서 출발합니다.</li>
    <li><strong>보수적 편향 보정</strong>: POC 참가자는 열성적 얼리어답터라 일반 직원보다 더 사용합니다. 편향 계수 {bias_factor}x로 나눠 전사 비용을 상향 방지했습니다.</li>
    <li><strong>통계적 분포 모델링</strong>: 분포 적합 품질 <strong>{fit_label}</strong> — 이론적으로 검증된 모델로 전사 백분위수를 추정해 신뢰성을 확보했습니다.</li>
    <li><strong>예산 버퍼 포함</strong>: P50(중앙값)이 아닌 P75(상위 25% 선)까지 권장 범위에 포함해 실제 편성 시 안전마진을 내재화했습니다.</li>
    <li><strong>모니터링 권장</strong>: 이 예측은 의사결정의 출발점입니다. 도입 후 실제 채택률과 사용량을 분기별로 측정해 재예측하면 정확도가 계속 개선됩니다.</li>
  </ol>
</div>

<div class="caution">
  <div style="font-weight:700;font-size:13px;color:#2c3e50;margin-bottom:4px">⚠ 예측 해석 시 주의사항</div>
  <ul>
    <li><strong>편향 계수가 가장 큰 변수입니다.</strong> 1.5x↔4.0x로 바꾸면 예측이 2~3배 출렁입니다 (섹션 3 민감도 분석 참고).
        POC 참가자가 일반 직원을 얼마나 대표하는지 조직 특성에 맞게 검증하세요.</li>
    <li><strong>POC 기간이 짧으면 신뢰구간이 넓어집니다.</strong> 표본·기간이 작을수록 불확실성이 커지므로
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

    # --- 예시용 실제 통계값 ---
    monthly_arr = np.array(report.monthly_per_user)
    ex_p50  = float(np.percentile(monthly_arr, 50))
    ex_p75  = float(np.percentile(monthly_arr, 75))
    ex_p95  = float(np.percentile(monthly_arr, 95))
    ex_mean = float(np.mean(monthly_arr))
    ex_max  = float(np.max(monthly_arr))
    ex_min  = float(np.min(monthly_arr))
    ex_std  = float(np.std(monthly_arr, ddof=1)) if len(monthly_arr) > 1 else 0.0
    ex_cv   = ex_std / ex_mean if ex_mean > 0 else 0.0
    mean_vs_p50 = ex_mean / ex_p50 if ex_p50 > 0 else 1.0
    raw_avg_per_user = report.total_poc_credit / report.poc_users if report.poc_users else 0.0
    eff_working = report.poc_period_days * report.working_day_ratio
    monthly_scale = WORKING_DAYS_PER_MONTH / eff_working if eff_working > 0 else 0.0
    step_stats = {
        "ex_p50": ex_p50, "ex_p75": ex_p75, "ex_p95": ex_p95,
        "ex_mean": ex_mean, "ex_max": ex_max, "ex_min": ex_min,
        "ex_std": ex_std, "ex_cv": ex_cv, "mean_vs_p50": mean_vs_p50,
        "raw_avg_per_user": raw_avg_per_user, "monthly_scale": monthly_scale,
    }

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

    # --- 시나리오 조회 ---
    mod  = next((s for s in report.scenarios if "Moderate"    in s.name), report.scenarios[0]) if report.scenarios else None
    sc_by = {s.name: s for s in report.scenarios}
    cons = sc_by.get("Conservative")
    modr = sc_by.get("Moderate")
    aggr = sc_by.get("Aggressive")

    # --- 경고 블록 ---
    warn_html = "".join(f'<div class="warn">⚠ {w}</div>\n' for w in report.warnings)

    # --- 요약 카드 ---
    poc_cr      = f"{report.total_poc_credit:,.0f} 크레딧"
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
    seg_rows = "".join(f"""
<tr>
  <td>{s.name}</td><td>{s.user_count}</td><td>{s.active_day_ratio_mean:.0%}</td>
  <td>{_fmt(s.credit_per_user_per_day_p50, usd)}</td>
  <td>{_fmt(s.monthly_credit_per_user_p50, usd)}</td>
  <td>{_fmt(s.monthly_credit_per_user_p75, usd)}</td>
  <td>{_fmt(s.monthly_credit_per_user_p95, usd)}</td>
  <td>{s.credit_share_pct:.1f}%</td>
</tr>""" for s in report.segment_stats)

    seg_table = f"""
<table>
<tr><th>세그먼트</th><th>인원</th><th>활성일%</th>
    <th>일평균P50</th><th>월P50</th><th>월P75</th><th>월P95</th><th>크레딧비율</th></tr>
{seg_rows}
</table>"""

    # --- 비용 예측 표 ---
    has_mc = any(s.mc_ci_p50_lower is not None for s in report.scenarios)

    def _ci_cell_html(pval, mc_lo, mc_hi, boot_lo, boot_hi):
        if mc_lo is not None:
            return (f"{_fmt(pval, usd)}<br>{_ci_span(mc_lo, mc_hi, usd)}"
                    f"<br><span style='font-size:10px;color:#aaa'>Monte Carlo</span>")
        return (f"{_fmt(pval, usd)}<br>{_ci_span(boot_lo, boot_hi, usd)}"
                f"<br><span style='font-size:10px;color:#aaa'>Bootstrap</span>")

    forecast_rows = "".join(f"""
<tr>
  <td>{s.name} ({s.adoption_rate:.0%})<br><small>{s.active_users:,}명 활성</small></td>
  <td>{_ci_cell_html(s.monthly_p50, s.mc_ci_p50_lower, s.mc_ci_p50_upper, s.ci_p50_lower, s.ci_p50_upper)}</td>
  <td>{_ci_cell_html(s.monthly_p75, s.mc_ci_p75_lower, s.mc_ci_p75_upper, s.ci_p75_lower, s.ci_p75_upper)}</td>
  <td>{_ci_cell_html(s.monthly_p95, s.mc_ci_p95_lower, s.mc_ci_p95_upper, s.ci_p95_lower, s.ci_p95_upper)}</td>
  <td>{_fmt(s.annual_p50, usd)}</td>
  <td>{_fmt(s.annual_p75, usd)}</td>
</tr>""" for s in report.scenarios)

    ci_label = "Monte Carlo 95% CI (bias·채택률 불확실성 포함)" if has_mc else "Bootstrap 95% CI (표본 추정 오차)"
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
<p style="font-size:11px;color:#7f8c8d;margin-top:6px">단위: {unit} · CI = {ci_label}</p>"""

    # --- 민감도 표 ---
    sens_rows = "".join(f"""
<tr>
  <td>{r['bias_factor']}x</td>
  <td>{_fmt(r['credit_per_user_monthly_p50'], usd)}</td>
  <td>{_fmt(r['credit_per_user_monthly_p75'], usd)}</td>
  <td>{_fmt(r['total_monthly_p50'], usd)}</td>
  <td>{_fmt(r['total_monthly_p75'], usd)}</td>
</tr>""" for r in report.sensitivity) if report.sensitivity else ""

    sens_table = f"""
<table>
<tr><th>편향 계수</th>
    <th>사용자당 P50</th><th>사용자당 P75</th>
    <th>전체 P50 (Moderate)</th><th>전체 P75 (Moderate)</th></tr>
{sens_rows}
</table>""" if sens_rows else ""

    # --- 역할 기반 자동 추론 표 ---
    role_table = ""
    if report.role_scenarios:
        rs0 = report.role_scenarios[0]
        role_rows_html = "".join(f"""
<tr>
  <td>{r.name} ({r.adoption_rate:.0%})<br><small>{r.active_users:,}명 활성</small></td>
  <td style="text-align:center">{r.heavy_n:,}</td>
  <td style="text-align:center">{r.medium_n:,}</td>
  <td style="text-align:center">{r.light_n:,}</td>
  <td>{_fmt(r.monthly_p50, usd)}</td>
  <td>{_fmt(r.monthly_p75, usd)}</td>
  <td>{_fmt(r.monthly_p95, usd)}</td>
  <td>{_fmt(r.annual_p50, usd)}</td>
</tr>""" for r in report.role_scenarios)
        role_table = f"""
<h4 style="margin:18px 0 8px">역할 기반 자동 추론 예측</h4>
<p style="font-size:12px;color:#7f8c8d;margin:0 0 8px">
  POC 세그먼트 비율(Heavy {rs0.adj_heavy_ratio:.1%} / Medium {rs0.adj_medium_ratio:.1%} / Light {rs0.adj_light_ratio:.1%})을
  bias 보정({report.bias_factor}x) 후 각 시나리오 채택 인원에 자동 적용.
</p>
<table>
<tr><th>시나리오</th>
    <th>Heavy 인원</th><th>Medium 인원</th><th>Light 인원</th>
    <th>월간 P50</th><th>월간 P75</th><th>월간 P95</th>
    <th>연간 P50<br><small>(램프업포함)</small></th></tr>
{role_rows_html}
</table>
<p style="font-size:11px;color:#7f8c8d;margin-top:6px">
  P50~P95 범위 = 같은 채택률 내 POC 분산만 반영 (채택률 불확실성 없음 → 기존 CI보다 좁음)
</p>"""

    # --- Executive Summary ---
    exec_summary_html = _build_exec_summary(
        cons, modr, aggr, usd, unit,
        report.poc_users, report.poc_period_days, fit,
    )

    # --- 도입부: 보고서 읽는 법 ---
    intro_html = """
<div class="intro">
  <h3>📋 이 보고서는 원자료에서 출발해 단계별로 예산 결론을 도출합니다</h3>
  <div style="font-size:12px;color:#7f8c8d;margin-bottom:6px">
    각 단계의 <b>녹색 결과값이 다음 단계의 입력</b>으로 이어지며, 단계 아래에 관련 차트와 설명이 함께 제공됩니다:
  </div>
  <div class="flow">
    <span class="flow-step">① 원자료 수집</span><span class="flow-arrow">→</span>
    <span class="flow-step">② 사용자별 집계</span><span class="flow-arrow">→</span>
    <span class="flow-step">③ 기초 통계</span><span class="flow-arrow">→</span>
    <span class="flow-step">④ 평균의 함정</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑤ 월간 정규화</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑥ 백분위수</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑦ 분포 적합</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑧ 부트스트랩</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑨ 편향 보정</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑩ 전사 확대</span><span class="flow-arrow">→</span>
    <span class="flow-step">⑪ 연간화</span><span class="flow-arrow">→</span>
    <span class="flow-step">결론</span>
  </div>
  <div style="font-size:11.5px;color:#95a5a6;margin-top:10px">
    💡 파란 박스(쉽게 말하면)는 핵심을 쉬운 말로, <b>▶ 더 알아보기</b>는 통계 심화 개념을 설명합니다.
  </div>
</div>"""

    # --- 결론 섹션 ---
    conclusion_html = _build_conclusion(
        cons, modr, aggr, usd, unit,
        report.bias_factor, report.poc_users, report.poc_period_days, fit,
    )

    # --- Step-by-Step 유도 과정 ---
    steps_html = _build_steps(
        report, fit, usd, unit, step_stats, modr, aggr, has_mc,
        c1, c2, c3, c4, c5, c6,
        seg_table, forecast_table, sens_table, role_table,
    )

    # --- 파라미터 요약표 (부록 B) ---
    fit_q_kr = {"good": "양호", "marginal": "보통", "poor": "불량"}[fit.fit_quality]
    params_table = f"""
<table>
<tr><th>파라미터</th><th>값</th><th>의미 / 근거</th></tr>
<tr><td>POC 표본</td><td>{report.poc_users}명 · {report.poc_period_days}일</td>
    <td>분석 기반 실측 데이터</td></tr>
<tr><td>활동일 비율</td><td>{report.working_day_ratio:.0%}</td>
    <td>월간 환산 scale = {step_stats['monthly_scale']:.3f} 산출에 사용</td></tr>
<tr><td>분포 적합</td><td>μ={fit.mu:.2f}, σ={fit.sigma:.2f} (KS p={fit.ks_pvalue:.3f})</td>
    <td>로그정규 · 적합 품질 <b>{fit_q_kr}</b></td></tr>
<tr><td>편향 계수</td><td>{report.bias_factor}x</td>
    <td>POC 열성 사용자 → 일반 직원 보정 (타당 1.5x~4.0x)</td></tr>
<tr><td>채택률 시나리오</td><td>20% / 45% / 70%</td>
    <td>Conservative / Moderate / Aggressive</td></tr>
<tr><td>램프업 가중합</td><td>{RAMP_SUM:.2f}</td>
    <td>1년차 연간 = 월간 × {RAMP_SUM:.2f} (6개월 점진 정착)</td></tr>
<tr><td>전사 인원</td><td>{report.enterprise_users:,}명</td>
    <td>확대 대상 모집단</td></tr>
</table>"""

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
  {exec_summary_html}
  {intro_html}
  {cards_html}

  <!-- ══ 분석 과정 Step-by-Step ════════════════════════════════════ -->
  <section id="analysis">
    <h2>분석 과정 — 원자료에서 예산 결론까지 (Step by Step)</h2>
    {steps_html}
  </section>

  <!-- ══ 결론: 권장 예산 ════════════════════════════════════════════ -->
  <section id="conclusion">
    <h2>결론 — 권장 예산 및 방어 근거</h2>
    {_plain("위 단계별 분석을 종합한 <b>예산 권장안</b>입니다. 단일 숫자가 아닌 <b>범위</b>로 제시하며, 각 범위의 근거와 신뢰 수준을 함께 명시합니다.")}
    {conclusion_html}
  </section>

  <!-- ══ 부록 A: 서비스별 사용 ══════════════════════════════════════ -->
  <section id="services">
    <h2>부록 A — 서비스별 크레딧 사용 (참고)</h2>
    <div class="appendix-note">📎 어떤 AI 모델·서비스(GPT-4o, 이미지 생성 등)에 크레딧이 쓰였는지 참고용으로 제공합니다. 비싼 모델에 사용이 몰리면 같은 사용량이라도 비용이 커집니다.</div>
    <div class="chart-wrap">
      <img src="data:image/png;base64,{c3}" alt="서비스별 사용량">
      <div class="chart-note">usage_type별 POC 기간 총 크레딧 소비량 (상위 10개).</div>
    </div>
  </section>

  <!-- ══ 부록 B: 핵심 가정·파라미터 요약 ════════════════════════════ -->
  <section id="assumptions">
    <h2>부록 B — 핵심 가정 및 파라미터 요약 (검증용)</h2>
    <div class="appendix-note">📎 예측에 사용된 모든 파라미터를 한 표로 정리했습니다. 단계별 상세 유도 과정은 위 <b>'분석 과정 전체 (Step by Step)'</b> 섹션을 참고하세요.</div>
    {params_table}
  </section>

</div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 리포트 저장: {path}")
