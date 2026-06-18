"""
데이터 생성 / 적재 / 검증 유틸리티.

강의록의 데이터 형식을 그대로 따른다.
- 계량형(value)   : [부분군명, 측정값]            예) [Date, Length]
- 계수형(count)   : [부분군명, sample_size, 관측값] 예) [Lot, sample_size, Defectives]

웹앱에서 업로드된 CSV를 분석에 사용할 수 있도록 표준 형태로 변환하는
헬퍼와, 데모용 시뮬레이션 데이터 생성 함수를 제공한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 시뮬레이션 데이터 생성 (강의록 generate_value_data / generate_count_data 기반)
# ---------------------------------------------------------------------------
def generate_value_data(
    var_name: str = "value",
    target: float = 0.0,
    sg_name: str = "subgroup",
    num_sg: int = 20,
    sg_size: int = 5,
    sg_std: float = 1.0,
    mean_shift: float = 0.0,
    sg_size_variation: int = 0,
    seed: int | None = None,
) -> pd.DataFrame:
    """계량형 관리도(Xbar-R/S, I-MR)용 데이터 생성.

    각 부분군마다 평균이 ±mean_shift 범위에서 흔들리고,
    정규분포 N(target+shift, sg_std)에서 표본을 추출한다.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(num_sg):
        shift = rng.uniform(-mean_shift, mean_shift)
        n_i = sg_size + rng.integers(-sg_size_variation, sg_size_variation + 1)
        n_i = max(1, int(n_i))
        frames.append(
            pd.DataFrame(
                {
                    sg_name: i + 1,
                    var_name: rng.normal(loc=target + shift, scale=sg_std, size=n_i),
                }
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True)


def generate_count_data(
    var_name: str = "count",
    sg_name: str = "lot",
    num_sg: int = 20,
    sg_size: int = 200,
    p: float = 0.01,
    sg_size_variation: int = 0,
    seed: int | None = None,
) -> pd.DataFrame:
    """계수형 관리도(NP, P, C, U)용 데이터 생성.

    각 부분군에서 표본 크기 sample_size 를 정하고, 이항분포로
    불량/결점 개수를 생성한다.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(num_sg):
        sample_size = sg_size + rng.integers(-sg_size_variation, sg_size_variation + 1)
        sample_size = max(1, int(sample_size))
        frames.append(
            pd.DataFrame(
                {
                    sg_name: i + 1,
                    "sample_size": [sample_size],
                    var_name: rng.binomial(n=sample_size, p=p),
                }
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# 업로드 데이터 → 표준 형태 변환
# ---------------------------------------------------------------------------
def to_value_frame(df: pd.DataFrame, sg_col: str, value_col: str) -> pd.DataFrame:
    """계량형 표준 프레임 [sg, value] 반환.

    부분군 컬럼과 측정값 컬럼이 같으면 ValueError 를 발생시켜
    중복 컬럼으로 인한 오류를 사전에 차단한다.
    """
    if sg_col == value_col:
        raise ValueError("부분군 컬럼과 측정값 컬럼은 서로 달라야 합니다.")
    out = pd.DataFrame(
        {
            sg_col: df[sg_col].to_numpy(),
            value_col: pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
        }
    )
    return out.dropna(subset=[value_col]).reset_index(drop=True)


def to_count_frame(
    df: pd.DataFrame, sg_col: str, size_col: str, value_col: str
) -> pd.DataFrame:
    """계수형 표준 프레임 [sg, sample_size, value] 반환.

    세 컬럼 중 중복이 있으면 ValueError 를 발생시킨다.
    """
    if len({sg_col, size_col, value_col}) < 3:
        raise ValueError("부분군 · 표본크기 · 관측값 컬럼은 서로 모두 달라야 합니다.")
    out = pd.DataFrame(
        {
            sg_col: df[sg_col].to_numpy(),
            size_col: pd.to_numeric(df[size_col], errors="coerce").to_numpy(),
            value_col: pd.to_numeric(df[value_col], errors="coerce").to_numpy(),
        }
    )
    out = out.dropna(subset=[size_col, value_col])
    out[size_col] = out[size_col].astype(int)
    out[value_col] = out[value_col].astype(int)
    return out.reset_index(drop=True)


def subgroup_sizes(df: pd.DataFrame, sg_col: str) -> pd.Series:
    """부분군별 관측치 개수."""
    return df.groupby(sg_col).size()


def is_constant_size(df: pd.DataFrame, sg_col: str) -> bool:
    """모든 부분군 크기가 동일한지 여부."""
    return subgroup_sizes(df, sg_col).nunique() == 1


def recommend_value_chart(df: pd.DataFrame, sg_col: str) -> str:
    """부분군 크기에 따른 계량형 관리도 추천 (강의록 선택기준 트리)."""
    sizes = subgroup_sizes(df, sg_col)
    mode_size = int(sizes.mode().iloc[0])
    if mode_size == 1:
        return "I-MR"
    if mode_size <= 5:
        return "Xbar-R"
    return "Xbar-s"
