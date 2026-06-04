"""
로그정규 분포 적합 및 검정 모듈.

엔터프라이즈 소프트웨어 사용량은 경험적으로 로그정규 분포를 따른다.
소수의 파워유저가 대부분의 사용량을 차지하는 heavy-tail 특성이 log-normal로 잘 포착된다.

분포를 적합시키면:
  - 경험적 백분위수(표본 의존) 대신 모수적 백분위수(분포 기반) 사용 가능
  - POC 표본 외 극단값 예측 가능 (P99 등)
  - KS 검정으로 적합 품질을 객관적으로 평가
"""
import numpy as np
from scipy import stats
from .schemas import DistFitResult

WORKING_DAYS_PER_MONTH = 22


def fit_lognormal(monthly_credits: np.ndarray) -> DistFitResult:
    """
    사용자별 월 크레딧 데이터에 로그정규 분포를 적합한다.

    scipy.stats.lognorm.fit(data, floc=0) 반환값:
      s    → sigma (shape, log-scale std)
      loc  → location (0으로 고정)
      scale → exp(mu)
    따라서 mu = log(scale).
    """
    # 0이나 음수 제거 (log 변환 불가)
    data = monthly_credits[monthly_credits > 0]
    if len(data) < 5:
        # 데이터 부족 시 경험적 값으로 대체
        return _fallback_fit(monthly_credits)

    try:
        s, loc, scale = stats.lognorm.fit(data, floc=0)
        mu = float(np.log(scale))
        sigma = float(s)
    except Exception:
        return _fallback_fit(monthly_credits)

    ks_stat, ks_pvalue = _ks_test(data, mu, sigma)
    fit_quality = _quality_label(ks_pvalue)
    pcts = _lognormal_percentiles(mu, sigma)

    return DistFitResult(
        mu=mu,
        sigma=sigma,
        ks_stat=float(ks_stat),
        ks_pvalue=float(ks_pvalue),
        fit_quality=fit_quality,
        p50_dist=pcts[50],
        p75_dist=pcts[75],
        p95_dist=pcts[95],
    )


def _ks_test(data: np.ndarray, mu: float, sigma: float) -> tuple[float, float]:
    """KS 검정: 데이터가 LogNormal(mu, sigma)에서 왔는지 검정."""
    stat, pval = stats.kstest(data, "lognorm", args=(sigma, 0, np.exp(mu)))
    return stat, pval


def _lognormal_percentiles(mu: float, sigma: float, pcts=(50, 75, 95)) -> dict:
    """
    로그정규 분포의 q번째 백분위수 = exp(mu + z_q * sigma)
    z_q = 표준정규분포 q번째 분위수 (norm.ppf(q/100))
    """
    result = {}
    for q in pcts:
        z = stats.norm.ppf(q / 100)
        result[q] = float(np.exp(mu + z * sigma))
    return result


def _quality_label(pvalue: float) -> str:
    if pvalue >= 0.05:
        return "good"
    elif pvalue >= 0.01:
        return "marginal"
    else:
        return "poor"


def _fallback_fit(data: np.ndarray) -> DistFitResult:
    """데이터 부족 또는 적합 실패 시 경험적 값 기반 더미 결과."""
    safe = data[data > 0] if len(data[data > 0]) > 0 else np.array([1.0])
    log_data = np.log(safe)
    mu = float(np.mean(log_data))
    sigma = float(np.std(log_data, ddof=1)) if len(log_data) > 1 else 1.0
    pcts = _lognormal_percentiles(mu, sigma)
    return DistFitResult(
        mu=mu, sigma=sigma,
        ks_stat=float("nan"), ks_pvalue=0.0,
        fit_quality="poor",
        p50_dist=pcts[50], p75_dist=pcts[75], p95_dist=pcts[95],
    )


def describe_fit(fit: DistFitResult) -> str:
    """콘솔/HTML 출력용 분포 적합 요약 문자열."""
    quality_kr = {"good": "양호", "marginal": "보통", "poor": "불량"}
    q = quality_kr.get(fit.fit_quality, fit.fit_quality)
    if np.isnan(fit.ks_stat):
        return f"로그정규(μ={fit.mu:.2f}, σ={fit.sigma:.2f}) — 데이터 부족으로 검정 생략"
    return (
        f"로그정규(μ={fit.mu:.2f}, σ={fit.sigma:.2f}), "
        f"KS p={fit.ks_pvalue:.3f} [{q}]"
    )
