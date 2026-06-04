"""
matplotlib 차트 생성 모듈.
각 함수는 base64 인코딩된 PNG 이미지 문자열을 반환한다.
한국어 폰트 불가 환경에서는 영어 레이블로 자동 폴백.
"""
import base64
import io
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from scipy import stats

from .distribution import WORKING_DAYS_PER_MONTH
from .schemas import DistFitResult
from .scaling import RAMP_MULTIPLIERS

# --- 폰트 설정 ---
_KO_AVAILABLE = False

def _setup_font():
    global _KO_AVAILABLE
    import matplotlib.font_manager as fm
    candidates = [
        "NanumGothic", "Malgun Gothic", "AppleGothic",
        "Noto Sans CJK KR", "Noto Sans KR", "UnDotum",
    ]
    default_fp = fm.findfont(fm.FontProperties())  # default font path
    for name in candidates:
        try:
            fp = fm.findfont(fm.FontProperties(family=name))
            # findfont returns default font when not found — check if it actually found our font
            if fp != default_fp and fp:
                matplotlib.rcParams["font.family"] = name
                _KO_AVAILABLE = True
                return
        except Exception:
            continue
    _KO_AVAILABLE = False

_setup_font()
matplotlib.rcParams["axes.unicode_minus"] = False

# 컬러 팔레트
_COLORS = {
    "heavy":        "#E74C3C",
    "medium":       "#F39C12",
    "light":        "#27AE60",
    "p50":          "#2980B9",
    "p75":          "#8E44AD",
    "p95":          "#C0392B",
    "conservative": "#27AE60",
    "moderate":     "#F39C12",
    "aggressive":   "#E74C3C",
    "lognorm":      "#E74C3C",
    "hist":         "#85C1E9",
}


def _label(ko: str, en: str) -> str:
    return ko if _KO_AVAILABLE else en


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _credit_fmt(usd_rate: float):
    if usd_rate == 1.0:
        return FuncFormatter(lambda x, _: f"{x:,.0f}")
    return FuncFormatter(lambda x, _: f"${x * usd_rate:,.0f}")


# ─────────────────────────────────────────────
# 차트 ①: 사용자별 월 크레딧 분포
# ─────────────────────────────────────────────
def chart_user_distribution(monthly_credits: np.ndarray, dist_fit: DistFitResult) -> str:
    data = monthly_credits[monthly_credits > 0]
    fig, ax = plt.subplots(figsize=(8, 4))

    # 히스토그램 (로그 x축)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.hist(data, bins=30, color=_COLORS["hist"], alpha=0.7, density=True,
                label=_label("관측 분포", "Observed"))

    ax.set_xscale("log")

    # 로그정규 곡선 오버레이
    x = np.logspace(np.log10(max(data.min(), 0.01)), np.log10(data.max()), 300)
    pdf = stats.lognorm.pdf(x, dist_fit.sigma, 0, np.exp(dist_fit.mu))
    ax.plot(x, pdf, color=_COLORS["lognorm"], lw=2,
            label=f"LogNormal(μ={dist_fit.mu:.2f}, σ={dist_fit.sigma:.2f})")

    # 백분위수 수직선
    for pct, val, col, lbl in [
        (50, dist_fit.p50_dist, _COLORS["p50"],  "P50"),
        (75, dist_fit.p75_dist, _COLORS["p75"],  "P75"),
        (95, dist_fit.p95_dist, _COLORS["p95"],  "P95"),
    ]:
        ax.axvline(val, color=col, linestyle="--", lw=1.2, label=f"{lbl}: {val:,.0f}")

    ax.set_xlabel(_label("월 크레딧 (로그 스케일)", "Monthly Credits (log scale)"))
    ax.set_ylabel(_label("밀도", "Density"))
    ax.set_title(_label("POC 사용자별 월 크레딧 분포", "Monthly Credit Distribution per User"))
    quality_map = {"good": "✓ 적합 양호", "marginal": "△ 적합 보통", "poor": "✗ 적합 불량"}
    q_en = {"good": "Good fit", "marginal": "Marginal fit", "poor": "Poor fit"}
    q_lbl = quality_map.get(dist_fit.fit_quality) if _KO_AVAILABLE else q_en.get(dist_fit.fit_quality, "")
    ax.set_title(ax.get_title() + f"  [{q_lbl}, KS p={dist_fit.ks_pvalue:.3f}]" if not np.isnan(dist_fit.ks_pvalue) else ax.get_title())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _to_base64(fig)


# ─────────────────────────────────────────────
# 차트 ②: 세그먼트 분해 (파이 + 크레딧 막대)
# ─────────────────────────────────────────────
def chart_segment_breakdown(segment_stats: list) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    names = [s.name for s in segment_stats]
    counts = [s.user_count for s in segment_stats]
    shares = [s.credit_share_pct for s in segment_stats]
    colors = [_COLORS.get(n.lower(), "#999") for n in names]

    # 파이차트: 사용자 수
    wedges, texts, autotexts = ax1.pie(
        counts, labels=names, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax1.set_title(_label("사용자 수 비율", "User Count Share"))

    # 가로 막대: 크레딧 점유율
    y = np.arange(len(names))
    bars = ax2.barh(y, shares, color=colors, height=0.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel(_label("크레딧 점유율 (%)", "Credit Share (%)"))
    ax2.set_title(_label("세그먼트별 크레딧 점유", "Credit Share by Segment"))
    ax2.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.set_xlim(0, max(shares) * 1.2)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(_label("사용자 세분화", "User Segmentation"), fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _to_base64(fig)


# ─────────────────────────────────────────────
# 차트 ③: 서비스별 크레딧 사용
# ─────────────────────────────────────────────
def chart_service_breakdown(usage_type_summary: dict) -> str:
    items = sorted(usage_type_summary.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, values = zip(*items) if items else ([], [])
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5 + 1)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(labels)))[::-1]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(_label("총 크레딧", "Total Credits"))
    ax.set_title(_label("서비스별 크레딧 사용량 (상위 10개)", "Credit by Service (Top 10)"))

    for bar, val in zip(bars, values):
        pct = val / total * 100 if total > 0 else 0
        ax.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}  ({pct:.1f}%)", va="center", fontsize=8)

    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _to_base64(fig)


# ─────────────────────────────────────────────
# 차트 ④: 시나리오별 월간 비용 비교
# ─────────────────────────────────────────────
def chart_scenario_comparison(scenarios: list, credit_to_usd: float) -> str:
    scenario_names = [s.name for s in scenarios]
    n = len(scenario_names)
    pcts = ["P50", "P75", "P95"]
    bar_w = 0.22
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = [-bar_w, 0, bar_w]
    pct_colors = [_COLORS["p50"], _COLORS["p75"], _COLORS["p95"]]

    for i, (pct, off, col) in enumerate(zip(pcts, offsets, pct_colors)):
        vals = [getattr(s, f"monthly_{pct.lower()}") * credit_to_usd for s in scenarios]
        bars = ax.bar(x + off, vals, width=bar_w, color=col, alpha=0.85, label=pct)
        labels_txt = [f"{v:,.0f}" for v in vals]
        ax.bar_label(bars, labels=labels_txt, fontsize=7, padding=2)

    ax.set_xticks(x)
    sc_labels = [
        f"{s.name}\n({s.adoption_rate:.0%}, {s.active_users:,}" + _label("명", "u") + ")"
        for s in scenarios
    ]
    ax.set_xticklabels(sc_labels, fontsize=9)
    ax.yaxis.set_major_formatter(_credit_fmt(credit_to_usd))
    unit = "USD" if credit_to_usd != 1.0 else _label("크레딧", "Credits")
    ax.set_ylabel(unit)
    ax.set_title(_label("시나리오별 월간 비용 예측 (P50 / P75 / P95)", "Monthly Cost Forecast by Scenario"))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _to_base64(fig)


# ─────────────────────────────────────────────
# 차트 ⑤: 12개월 램프업 곡선
# ─────────────────────────────────────────────
def chart_rampup_curve(scenarios: list, credit_to_usd: float) -> str:
    months = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(9, 5))

    sc_colors = [_COLORS["conservative"], _COLORS["moderate"], _COLORS["aggressive"]]
    for s, col in zip(scenarios, sc_colors):
        monthly_costs = [s.monthly_p50 * m * credit_to_usd for m in RAMP_MULTIPLIERS]
        ax.plot(months, monthly_costs, marker="o", markersize=4, color=col,
                label=f"{s.name} P50 ({s.adoption_rate:.0%})")
        # 스테디스테이트 점선
        ax.axhline(s.monthly_p50 * credit_to_usd, color=col, linestyle=":", alpha=0.5)

    ax.set_xticks(months)
    ax.set_xlabel(_label("도입 후 경과 월", "Month After Rollout"))
    ax.yaxis.set_major_formatter(_credit_fmt(credit_to_usd))
    unit = "USD" if credit_to_usd != 1.0 else _label("크레딧", "Credits")
    ax.set_ylabel(unit)
    ax.set_title(_label("전사 도입 후 월별 비용 추이 (P50, 램프업 포함)", "Monthly Cost Over 12 Months (P50, with Ramp-Up)"))

    # 램프업 구간 음영
    ax.axvspan(1, 6.5, alpha=0.06, color="gray",
               label=_label("램프업 구간 (1~6개월)", "Ramp-up Period (M1-6)"))
    ax.axvspan(6.5, 12, alpha=0.03, color="blue",
               label=_label("정상 운영 구간", "Steady State"))

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _to_base64(fig)


# ─────────────────────────────────────────────
# 차트 ⑥: 편향 계수 민감도
# ─────────────────────────────────────────────
def chart_bias_sensitivity(
    pop_percentiles: dict,
    dist_fit: DistFitResult,
    enterprise_users: int,
    current_bias: float,
    credit_to_usd: float,
    adoption_rate: float = 0.45,
) -> str:
    active_users = int(enterprise_users * adoption_rate)
    bias_range = np.linspace(1.0, 5.0, 100)

    p75_base = dist_fit.p75_dist if dist_fit.fit_quality == "good" else pop_percentiles["p75"]
    p50_base = dist_fit.p50_dist if dist_fit.fit_quality == "good" else pop_percentiles["p50"]

    y_p75 = [p75_base / bf * active_users * credit_to_usd for bf in bias_range]
    y_p50 = [p50_base / bf * active_users * credit_to_usd for bf in bias_range]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(bias_range, y_p75, color=_COLORS["p75"], lw=2,
            label=_label("Moderate P75 월간", "Moderate P75 Monthly"))
    ax.plot(bias_range, y_p50, color=_COLORS["p50"], lw=2, linestyle="--",
            label=_label("Moderate P50 월간", "Moderate P50 Monthly"))

    # 현재 편향 계수 표시
    ax.axvline(current_bias, color="black", linestyle="--", lw=1.5,
               label=_label(f"현재 설정 ({current_bias}x)", f"Current ({current_bias}x)"))
    current_p75 = p75_base / current_bias * active_users * credit_to_usd
    ax.scatter([current_bias], [current_p75], color="black", zorder=5, s=50)

    # 합리적 범위 음영 (1.5~4.0)
    ax.axvspan(1.5, 4.0, alpha=0.08, color="green",
               label=_label("권장 범위 (1.5x~4.0x)", "Reasonable Range (1.5x-4.0x)"))

    ax.set_xlabel(_label("편향 보정 계수", "Bias Factor"))
    ax.yaxis.set_major_formatter(_credit_fmt(credit_to_usd))
    unit = "USD" if credit_to_usd != 1.0 else _label("크레딧", "Credits")
    ax.set_ylabel(unit)
    ax.set_title(_label(
        f"편향 계수별 월간 비용 민감도 (Moderate, 채택률 {adoption_rate:.0%})",
        f"Bias Factor Sensitivity (Moderate, {adoption_rate:.0%} adoption)"
    ))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _to_base64(fig)
