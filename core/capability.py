"""
공정능력분석(Process Capability Analysis) 모듈.

강의록 공정능력분석 강의록의 절차를 구현한다.
- 정규성 검정 (Shapiro-Wilk)
- 정규성 불만족 시 Box-Cox 변환 (규격도 동일 변환 적용)
- 단기 공정능력지수 Cp, Cpk  (군내변동 sigma_within)
- 장기 공정능력지수 Pp, Ppk  (전체변동 sigma_overall)
- 공정능력 등급 판정
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import boxcox, shapiro

from . import constants as C


# ---------------------------------------------------------------------------
# 표준편차 추정
# ---------------------------------------------------------------------------
def sigma_within(df: pd.DataFrame, sg_col: str, value_col: str, window: int = 2) -> float:
    """군내변동(단기) 표준편차.

    - 부분군 크기가 2 이상: 합동표준편차(pooled std)를 c4(d)로 보정
    - 부분군 크기가 1     : 이동범위 평균을 d2(window)로 보정
    """
    sizes = df.groupby(sg_col)[value_col].size()
    if (sizes == 1).all():
        x = df[value_col].to_numpy(dtype=float)
        mr = pd.Series(x).rolling(window=window).apply(
            lambda v: v.max() - v.min(), raw=True
        )
        mr_bar = mr.iloc[window - 1:].mean()
        return float(mr_bar / C.d2(window))

    # 합동표준편차: s_p = sqrt( sum (n_i-1) s_i^2 / sum (n_i-1) )
    g = df.groupby(sg_col)[value_col]
    var_i = g.var(ddof=1)
    n_i = g.size()
    dof = (n_i - 1)
    s_p = np.sqrt((dof * var_i).sum() / dof.sum())
    d = int(dof.sum()) + 1
    return float(s_p / C.c4(d))


def sigma_overall(df: pd.DataFrame, value_col: str) -> float:
    """전체변동(장기) 표준편차 = s / c4(n)."""
    x = df[value_col].to_numpy(dtype=float)
    s = np.std(x, ddof=1)
    n = len(x)
    return float(s / C.c4(n))


# ---------------------------------------------------------------------------
# 정규성 검정
# ---------------------------------------------------------------------------
@dataclass
class NormalityResult:
    statistic: float
    p_value: float
    is_normal: bool


def normality_test(values: np.ndarray, alpha: float = 0.05) -> NormalityResult:
    """Shapiro-Wilk 정규성 검정. p >= alpha 이면 정규성 만족."""
    stat, p = shapiro(np.asarray(values, dtype=float))
    return NormalityResult(float(stat), float(p), bool(p >= alpha))


# ---------------------------------------------------------------------------
# 공정능력지수
# ---------------------------------------------------------------------------
@dataclass
class CapabilityResult:
    Cp: float
    Cpk: float
    Pp: float
    Ppk: float
    mean: float
    sigma_within: float
    sigma_overall: float
    USL: float
    LSL: float
    normal: NormalityResult
    transformed: bool = False
    lambda_: float | None = None
    # 변환된 척도에서의 데이터/규격 (그래프용)
    work_values: np.ndarray = field(default=None, repr=False)
    work_usl: float = None
    work_lsl: float = None


def _indices(mean, sw, so, usl, lsl):
    cp = (usl - lsl) / (6 * sw)
    cpk = min((usl - mean) / (3 * sw), (mean - lsl) / (3 * sw))
    pp = (usl - lsl) / (6 * so)
    ppk = min((usl - mean) / (3 * so), (mean - lsl) / (3 * so))
    return cp, cpk, pp, ppk


def process_capability(
    df: pd.DataFrame,
    sg_col: str,
    value_col: str,
    usl: float,
    lsl: float,
    window: int = 2,
    alpha: float = 0.05,
) -> CapabilityResult:
    """공정능력지수(Cp, Cpk, Pp, Ppk)를 계산.

    정규성을 만족하지 않으면 Box-Cox 변환을 적용하고, 규격(USL/LSL)도
    동일 변환하여 변환된 척도에서 지수를 계산한다. (공정능력지수는 무차원
    비율이므로 변환 전후 해석이 동일하다.)
    """
    raw = df[value_col].to_numpy(dtype=float)
    norm = normality_test(raw, alpha)

    transformed = False
    lam = None
    work_df = df
    work_usl, work_lsl = usl, lsl
    work_values = raw

    if not norm.is_normal and np.all(raw > 0):
        # Box-Cox 변환 (양수 데이터에 한함)
        tv, lam = boxcox(raw)
        transformed = True
        work_values = tv
        work_df = df.copy()
        work_df[value_col] = tv
        # 규격도 동일 변환
        work_usl = _boxcox_value(usl, lam)
        work_lsl = _boxcox_value(lsl, lam)

    mean = float(np.mean(work_values))
    sw = sigma_within(work_df, sg_col, value_col, window)
    so = sigma_overall(work_df, value_col)
    cp, cpk, pp, ppk = _indices(mean, sw, so, work_usl, work_lsl)

    return CapabilityResult(
        Cp=cp, Cpk=cpk, Pp=pp, Ppk=ppk,
        mean=mean, sigma_within=sw, sigma_overall=so,
        USL=usl, LSL=lsl, normal=norm,
        transformed=transformed, lambda_=lam,
        work_values=work_values, work_usl=work_usl, work_lsl=work_lsl,
    )


def _boxcox_value(y: float, lam: float) -> float:
    """단일 값에 Box-Cox 변환 적용 (scipy.boxcox 와 동일 정의)."""
    if abs(lam) < 1e-12:
        return float(np.log(y))
    return float((y ** lam - 1) / lam)


# ---------------------------------------------------------------------------
# 공정능력 등급 판정 (강의록 p.9 판정표)
# ---------------------------------------------------------------------------
def grade(cp: float) -> dict:
    """Cp(또는 Cpk) 값에 따른 등급 / 판정 / 시정조치."""
    table = [
        (1.67, 0, "공정능력이 매우 충분", "관리의 간소화를 생각", "±5σ"),
        (1.33, 1, "공정능력이 충분", "현 상태 유지", "±4σ"),
        (1.00, 2, "충분하지는 않지만 괜찮음", "불량발생 가능성 주의", "±3σ"),
        (0.67, 3, "공정능력이 모자람", "전체 선별·공정 개선 필요", "±2σ"),
        (0.00, 4, "공정능력이 매우 부족", "긴급 대책·규격 재검토 필요", "±1σ"),
    ]
    for threshold, g, judge, action, sigma in table:
        if cp >= threshold:
            return {"등급": g, "판정": judge, "시정조치": action, "비고": sigma}
    return {"등급": 4, "판정": "공정능력이 매우 부족", "시정조치": "긴급 대책 필요", "비고": "±1σ"}
