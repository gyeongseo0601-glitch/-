"""
관리도(Control Chart) 계산 모듈.

강의록의 generate_value_chart / generate_count_chart 를 재구성하여
관측치(point), 중심선(CL), 상한(UCL), 하한(LCL) 데이터프레임을 생성한다.

반환 형식: 각 관리도는 컬럼 [point, CL, LCL, UCL] 을 갖는 DataFrame.
- 계량형: Xbar-R, Xbar-s, I-MR  → (주관리도, 보조관리도) 튜플
- 계수형: NP, P, C, U          → (관리도,) 튜플

LCL 이 음수가 될 수 있는 계수형/범위형 관리도는 0으로 절단(clip)한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C


# ===========================================================================
# 계량형 관리도
# ===========================================================================
def value_chart(
    df: pd.DataFrame, chart_type: str = "Xbar-R", window: int = 2
):
    """계량형 관리도 생성.

    df 컬럼: [부분군명, 측정값]
    chart_type: 'Xbar-R' | 'Xbar-s' | 'I-MR'
    window: I-MR 의 이동범위 윈도우 크기 (기본 2 = 표준 I-MR)
    """
    sg_name, var_name = df.columns[:2]

    if chart_type == "Xbar-R":
        g = df.groupby(sg_name)[var_name]
        sg = pd.DataFrame(
            {
                "Xbar": g.mean(),
                "R": g.max() - g.min(),
                "n_i": g.size(),
            }
        )
        xbar_bar = sg["Xbar"].mean()
        r_bar = sg["R"].mean()
        m = int(sg["n_i"].mode().iloc[0])  # 최빈 부분군 크기

        a2, d3c, d4c = C.A2(m), C.D3(m), C.D4(m)

        xbar = pd.DataFrame(index=sg.index)
        xbar["point"] = sg["Xbar"]
        xbar["CL"] = xbar_bar
        xbar["LCL"] = xbar_bar - a2 * r_bar
        xbar["UCL"] = xbar_bar + a2 * r_bar

        rchart = pd.DataFrame(index=sg.index)
        rchart["point"] = sg["R"]
        rchart["CL"] = r_bar
        rchart["LCL"] = d3c * r_bar
        rchart["UCL"] = d4c * r_bar
        return xbar, rchart

    if chart_type == "Xbar-s":
        g = df.groupby(sg_name)[var_name]
        sg = pd.DataFrame(
            {
                "Xbar": g.mean(),
                "s": g.std(ddof=1),
                "n_i": g.size(),
            }
        )
        xbar_bar = sg["Xbar"].mean()
        s_bar = sg["s"].mean()
        m = int(sg["n_i"].mode().iloc[0])

        a3, b3, b4 = C.A3(m), C.B3(m), C.B4(m)

        xbar = pd.DataFrame(index=sg.index)
        xbar["point"] = sg["Xbar"]
        xbar["CL"] = xbar_bar
        xbar["LCL"] = xbar_bar - a3 * s_bar
        xbar["UCL"] = xbar_bar + a3 * s_bar

        schart = pd.DataFrame(index=sg.index)
        schart["point"] = sg["s"]
        schart["CL"] = s_bar
        schart["LCL"] = b3 * s_bar
        schart["UCL"] = b4 * s_bar
        return xbar, schart

    if chart_type == "I-MR":
        w = window
        sg = df.set_index(sg_name)
        x = sg[var_name]
        xbar = x.mean()

        # 이동범위 MR_i = (윈도우 내 최대 - 최소)
        mr = x.rolling(window=w).apply(lambda v: v.max() - v.min(), raw=True)
        mr_bar = mr.iloc[w - 1:].mean()

        d2c = C.d2(w)
        d3c, d4c = C.D3(w), C.D4(w)

        i_chart = pd.DataFrame(index=x.index)
        i_chart["point"] = x.values
        i_chart["CL"] = xbar
        i_chart["LCL"] = xbar - 3 * mr_bar / d2c
        i_chart["UCL"] = xbar + 3 * mr_bar / d2c

        mr_chart = pd.DataFrame(index=x.index)
        mr_chart["point"] = mr.values
        mr_chart["CL"] = mr_bar
        mr_chart["LCL"] = max(0.0, d3c) * mr_bar
        mr_chart["UCL"] = d4c * mr_bar
        return i_chart, mr_chart

    raise ValueError(f"알 수 없는 chart_type: {chart_type}")


# ===========================================================================
# 계수형 관리도
# ===========================================================================
def count_chart(df: pd.DataFrame, chart_type: str = "NP"):
    """계수형 관리도 생성.

    df 컬럼: [부분군명, sample_size, 관측값]
    chart_type: 'NP' | 'P' | 'C' | 'U'
    """
    sg_name, n_col, var_name = df.columns[:3]
    data = df.set_index(sg_name)
    n_i = data[n_col]
    x = data[var_name]

    if chart_type == "NP":
        # 불량 개수 관리도 (표본 크기 동일해야 함)
        k = len(data)
        np_bar = x.sum() / k
        p_bar = x.sum() / n_i.sum()
        sd = 3 * np.sqrt(np_bar * (1 - p_bar))
        out = pd.DataFrame(index=data.index)
        out["point"] = x.values
        out["CL"] = np_bar
        out["LCL"] = max(0.0, np_bar - sd)
        out["UCL"] = np_bar + sd
        return (out,)

    if chart_type == "P":
        # 불량률 관리도 (표본 크기 달라도 됨, 점별 한계)
        p_bar = x.sum() / n_i.sum()
        sd = 3 * np.sqrt(p_bar * (1 - p_bar) / n_i)
        out = pd.DataFrame(index=data.index)
        out["point"] = (x / n_i).values
        out["CL"] = p_bar
        out["LCL"] = (p_bar - sd).clip(lower=0).values
        out["UCL"] = (p_bar + sd).values
        return (out,)

    if chart_type == "C":
        # 결점수 관리도 (검사단위 동일해야 함)
        c_bar = x.mean()
        sd = 3 * np.sqrt(c_bar)
        out = pd.DataFrame(index=data.index)
        out["point"] = x.values
        out["CL"] = c_bar
        out["LCL"] = max(0.0, c_bar - sd)
        out["UCL"] = c_bar + sd
        return (out,)

    if chart_type == "U":
        # 단위당 결점수 관리도 (검사단위 달라도 됨, 점별 한계)
        u_bar = x.sum() / n_i.sum()
        sd = 3 * np.sqrt(u_bar / n_i)
        out = pd.DataFrame(index=data.index)
        out["point"] = (x / n_i).values
        out["CL"] = u_bar
        out["LCL"] = (u_bar - sd).clip(lower=0).values
        out["UCL"] = (u_bar + sd).values
        return (out,)

    raise ValueError(f"알 수 없는 chart_type: {chart_type}")


# ===========================================================================
# 이상치 제거 후 관리도 재작성 (강의록: 관리도 재작성 절차)
# ===========================================================================
def out_of_control_subgroups(chart: pd.DataFrame) -> list:
    """관리한계(UCL/LCL)를 벗어난 부분군 인덱스 목록."""
    mask = (chart["point"] > chart["UCL"]) | (chart["point"] < chart["LCL"])
    return chart.index[mask].tolist()


def recompute_value_chart(
    df: pd.DataFrame, chart_type: str = "Xbar-R", window: int = 2, max_iter: int = 20
):
    """이상 부분군을 반복적으로 제거하며 관리한계를 재계산.

    모든 점이 관리상태가 될 때까지(또는 max_iter) 반복하고,
    (최종 주관리도, 최종 보조관리도, 제거된 부분군 목록) 을 반환한다.
    """
    sg_name = df.columns[0]
    work = df.copy()
    removed: list = []

    for _ in range(max_iter):
        main_chart, sub_chart = value_chart(work, chart_type, window)
        ooc_main = out_of_control_subgroups(main_chart)
        ooc_sub = out_of_control_subgroups(sub_chart)
        ooc = sorted(set(ooc_main) | set(ooc_sub))
        if not ooc:
            break
        removed.extend(ooc)
        work = work[~work[sg_name].isin(ooc)].copy()

    main_chart, sub_chart = value_chart(work, chart_type, window)
    return main_chart, sub_chart, removed
