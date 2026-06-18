"""
시간에 따라 가중되는 관리도 (Time Weighted Control Chart).

강의록(통계적공정관리 p.2)에 개념만 소개되고 수식/코드는 제시되지 않은
CUSUM / EWMA 관리도를 표준 SPC 공식으로 구현한다.
Shewhart 관리도가 한 시점의 큰 변동만 잡는 데 비해, 이 관리도들은 작은
변화가 누적되어 반영되므로 작은 추세 변화 탐지에 유용하다.

입력은 다른 관리도와 동일한 계량형 표준 프레임 [부분군, 측정값] 이다.
부분군 크기가 2 이상이면 부분군 평균을, 1이면 개별값을 타점으로 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constants as C


@dataclass
class PointSpec:
    points: pd.Series     # 타점 (부분군 평균 또는 개별값), index = 부분군 라벨
    mu: float             # 중심(목표) 값
    sigma: float          # 타점의 표준편차
    basis: str            # 'individuals' | 'subgroup means'


def _prepare(df: pd.DataFrame, window: int = 2, target: float | None = None) -> PointSpec:
    """타점 계열과 중심/표준편차 추정값을 준비."""
    sg_name, var_name = df.columns[:2]
    sizes = df.groupby(sg_name)[var_name].size()

    if (sizes == 1).all():
        pts = df.set_index(sg_name)[var_name]
        mu = float(pts.mean()) if target is None else float(target)
        mr = pts.rolling(window=window).apply(lambda v: v.max() - v.min(), raw=True)
        sigma = float(mr.iloc[window - 1:].mean() / C.d2(window))
        return PointSpec(pts, mu, sigma, "individuals")

    g = df.groupby(sg_name)[var_name]
    pts = g.mean()
    m = int(g.size().mode().iloc[0])
    r_bar = (g.max() - g.min()).mean()
    sigma_proc = r_bar / C.d2(m)          # 공정 표준편차 추정
    sigma = float(sigma_proc / np.sqrt(m))  # 부분군 평균의 표준편차
    mu = float(pts.mean()) if target is None else float(target)
    return PointSpec(pts, mu, sigma, "subgroup means")


# ---------------------------------------------------------------------------
# EWMA 관리도
# ---------------------------------------------------------------------------
def ewma_chart(df, lam: float = 0.2, L: float = 3.0,
               target: float | None = None, window: int = 2) -> pd.DataFrame:
    """지수가중이동평균(EWMA) 관리도.

    z_i = λ·x_i + (1-λ)·z_(i-1),  z_0 = μ
    한계: μ ± L·σ·sqrt( (λ/(2-λ))·(1-(1-λ)^(2i)) )

    반환: [point(=z), CL, LCL, UCL]  (한계는 점마다 다름)
    """
    spec = _prepare(df, window, target)
    x = spec.points.to_numpy(dtype=float)
    mu, sigma = spec.mu, spec.sigma

    z = np.empty(len(x))
    prev = mu
    for i, xi in enumerate(x):
        prev = lam * xi + (1 - lam) * prev
        z[i] = prev

    i_arr = np.arange(1, len(x) + 1)
    spread = L * sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * i_arr)))

    out = pd.DataFrame(index=spec.points.index)
    out["point"] = z
    out["CL"] = mu
    out["LCL"] = mu - spread
    out["UCL"] = mu + spread
    return out


# ---------------------------------------------------------------------------
# CUSUM 관리도 (tabular 방식)
# ---------------------------------------------------------------------------
def cusum_chart(df, k: float = 0.5, h: float = 5.0,
                target: float | None = None, window: int = 2) -> pd.DataFrame:
    """누적합(CUSUM) 관리도 — tabular(표) 방식.

    K = k·σ (기준값),  H = h·σ (결정구간)
    C_i^+ = max(0, x_i - (μ+K) + C_(i-1)^+)
    C_i^- = max(0, (μ-K) - x_i + C_(i-1)^-)
    C^+ > H  또는  C^- > H  이면 이상 신호.

    반환: [Cplus, Cminus(음수로 표시), H, mH]
    """
    spec = _prepare(df, window, target)
    x = spec.points.to_numpy(dtype=float)
    mu, sigma = spec.mu, spec.sigma
    K = k * sigma
    H = h * sigma

    cp = np.zeros(len(x))
    cm = np.zeros(len(x))
    for i, xi in enumerate(x):
        prev_p = cp[i - 1] if i > 0 else 0.0
        prev_m = cm[i - 1] if i > 0 else 0.0
        cp[i] = max(0.0, xi - (mu + K) + prev_p)
        cm[i] = max(0.0, (mu - K) - xi + prev_m)

    out = pd.DataFrame(index=spec.points.index)
    out["Cplus"] = cp
    out["Cminus"] = -cm          # 그래프에서 아래쪽으로 표시
    out["H"] = H
    out["mH"] = -H
    return out


def cusum_signals(cusum_df: pd.DataFrame) -> list:
    """CUSUM 이상 신호가 발생한 부분군 라벨 목록."""
    H = cusum_df["H"].iloc[0]
    mask = (cusum_df["Cplus"] > H) | (cusum_df["Cminus"] < -H)
    return cusum_df.index[mask].tolist()


def ewma_signals(ewma_df: pd.DataFrame) -> list:
    """EWMA 이상 신호(한계 이탈)가 발생한 부분군 라벨 목록."""
    mask = (ewma_df["point"] > ewma_df["UCL"]) | (ewma_df["point"] < ewma_df["LCL"])
    return ewma_df.index[mask].tolist()
