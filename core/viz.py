"""
Plotly 시각화 모듈 (관리도 / 공정능력분석 그래프).

강의록의 plot_variable_control_chart / plot_process_capability 를 재구성하여
Streamlit 에서 사용할 figure 객체를 반환한다. 이상점은 빨간색으로 강조한다.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, probplot

from . import nelson_rules as nr


# ---------------------------------------------------------------------------
# 관리도 시각화
# ---------------------------------------------------------------------------
def control_chart_figure(charts, names, title, mark_rules=True):
    """관리도(들)를 서브플롯으로 시각화.

    charts: 관리도 DataFrame 의 튜플/리스트  (예: (xbar, rchart))
    names : 각 서브플롯의 y축 이름        (예: ['Xbar', 'R'])
    """
    n = len(charts)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                        subplot_titles=[f"{nm} 관리도" for nm in names],
                        vertical_spacing=0.13)

    for i, ch in enumerate(charts):
        row = i + 1
        idx = list(ch.index)

        # 위반 점 탐지
        out_pos = set()
        if mark_rules:
            v = nr.evaluate(ch)
            out_pos = {p for p, s in v.items() if s}

        colors = ["#d62728" if k in out_pos else "#1f77b4"
                  for k in range(len(ch))]

        # 관측치
        fig.add_trace(
            go.Scatter(x=idx, y=ch["point"], mode="lines+markers",
                       marker=dict(size=8, color=colors),
                       line=dict(color="#4c78a8"), name=names[i]),
            row=row, col=1,
        )
        # CL / UCL / LCL
        fig.add_trace(go.Scatter(x=idx, y=ch["CL"], mode="lines",
                                 line=dict(color="green", dash="dashdot"),
                                 name="CL", showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=idx, y=ch["UCL"], mode="lines",
                                 line=dict(color="magenta", dash="dot"),
                                 name="UCL", showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=idx, y=ch["LCL"], mode="lines",
                                 line=dict(color="red", dash="dot"),
                                 name="LCL", showlegend=False), row=row, col=1)

        # 한계값 주석
        last = idx[-1]
        for key, color in [("UCL", "magenta"), ("CL", "green"), ("LCL", "red")]:
            fig.add_annotation(x=last, y=ch[key].iloc[-1],
                               text=f"{key}={ch[key].iloc[-1]:.3f}",
                               showarrow=False, xanchor="left",
                               font=dict(color=color, size=11), row=row, col=1)
        fig.update_yaxes(title_text=names[i], row=row, col=1)
        fig.update_xaxes(title_text="부분군", ticks="outside", row=row, col=1)

    fig.update_layout(template="seaborn", title=title, showlegend=False,
                      height=300 * n + 80, margin=dict(l=50, r=90, t=80, b=50))
    return fig


# ---------------------------------------------------------------------------
# 공정능력분석 시각화
# ---------------------------------------------------------------------------
def ewma_figure(chart, title="EWMA 관리도"):
    """EWMA 관리도 (점마다 다른 한계). 한계 이탈 점은 빨간색."""
    idx = list(chart.index)
    out = set(chart.index[(chart["point"] > chart["UCL"]) |
                          (chart["point"] < chart["LCL"])])
    colors = ["#d62728" if k in out else "#1f77b4" for k in idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=chart["point"], mode="lines+markers",
                             marker=dict(size=8, color=colors),
                             line=dict(color="#4c78a8"), name="EWMA"))
    fig.add_trace(go.Scatter(x=idx, y=chart["CL"], mode="lines",
                             line=dict(color="green", dash="dashdot"), name="CL"))
    fig.add_trace(go.Scatter(x=idx, y=chart["UCL"], mode="lines",
                             line=dict(color="magenta", dash="dot"), name="UCL"))
    fig.add_trace(go.Scatter(x=idx, y=chart["LCL"], mode="lines",
                             line=dict(color="red", dash="dot"), name="LCL"))
    fig.update_layout(template="seaborn", title=title, showlegend=False,
                      xaxis_title="부분군", yaxis_title="EWMA 통계량",
                      height=420, margin=dict(l=50, r=40, t=70, b=50))
    return fig


def cusum_figure(chart, title="CUSUM 관리도"):
    """CUSUM 관리도: C+ (위), C- (아래), 결정구간 ±H."""
    idx = list(chart.index)
    H = chart["H"].iloc[0]
    out_p = set(chart.index[chart["Cplus"] > H])
    out_m = set(chart.index[chart["Cminus"] < -H])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=chart["Cplus"], mode="lines+markers", name="C+",
        line=dict(color="#1f77b4"),
        marker=dict(size=8, color=["#d62728" if k in out_p else "#1f77b4" for k in idx])))
    fig.add_trace(go.Scatter(
        x=idx, y=chart["Cminus"], mode="lines+markers", name="C-",
        line=dict(color="#9467bd"),
        marker=dict(size=8, color=["#d62728" if k in out_m else "#9467bd" for k in idx])))
    fig.add_trace(go.Scatter(x=idx, y=chart["H"], mode="lines",
                             line=dict(color="magenta", dash="dot"), name="+H"))
    fig.add_trace(go.Scatter(x=idx, y=chart["mH"], mode="lines",
                             line=dict(color="red", dash="dot"), name="-H"))
    fig.add_hline(y=0, line_width=1, line_color="green", line_dash="dashdot")
    fig.update_layout(template="seaborn", title=title, showlegend=True,
                      xaxis_title="부분군", yaxis_title="누적합",
                      height=420, margin=dict(l=50, r=40, t=70, b=50))
    return fig


def capability_figure(values, lsl, usl, result, var_name="value"):
    """히스토그램 + 정규분포 곡선 + 규격선 + 지수 주석."""
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=20, histnorm="probability density",
                               name=var_name, opacity=0.6, marker_color="#6baed6"))

    xs = np.linspace(min(values.min(), lsl), max(values.max(), usl), 400)
    fig.add_trace(go.Scatter(x=xs, y=norm.pdf(xs, mean, std), mode="lines",
                             line=dict(color="#08519c", width=2), name="정규분포"))

    for x, label in [(lsl, "LSL"), (usl, "USL")]:
        fig.add_vline(x=x, line_width=1.5, line_dash="dash", line_color="red",
                      annotation_text=label, annotation_position="top")
    fig.add_vline(x=mean, line_width=1, line_dash="dot", line_color="green",
                  annotation_text="평균", annotation_position="bottom")

    text = (f"Cp = {result.Cp:.4f}<br>Cpk = {result.Cpk:.4f}<br>"
            f"Pp = {result.Pp:.4f}<br>Ppk = {result.Ppk:.4f}")
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.98,
                       text=text, align="right", showarrow=False,
                       bordercolor="black", borderwidth=1, bgcolor="white")

    suffix = " (Box-Cox 변환척도)" if result.transformed else ""
    fig.update_layout(template="seaborn",
                      title=f"{var_name} 공정능력분석{suffix}",
                      xaxis_title=var_name, yaxis_title="밀도",
                      height=480, margin=dict(l=40, r=40, t=70, b=40))
    return fig


def qq_figure(values, title="Q-Q Plot"):
    """정규성 확인용 Q-Q plot."""
    z = (np.asarray(values, float) - np.mean(values)) / np.std(values, ddof=1)
    (osm, osr), (slope, intercept, _) = probplot(z, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                             marker=dict(color="#4c78a8", size=6), name="데이터"))
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines",
                             line=dict(color="red", width=2), name="기준선"))
    fig.update_layout(template="seaborn", title=title,
                      xaxis_title="이론 분위수", yaxis_title="표본 분위수",
                      height=380, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def boxcox_compare_figure(raw, transformed, lam):
    """Box-Cox 변환 전후 히스토그램 비교."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["원본 데이터", f"Box-Cox 변환 (λ={lam:.3f})"])
    fig.add_trace(go.Histogram(x=raw, nbinsx=20, marker_color="indianred",
                               opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=transformed, nbinsx=20, marker_color="steelblue",
                               opacity=0.7), row=1, col=2)
    fig.update_layout(template="seaborn", title="Box-Cox 변환 전후 비교",
                      showlegend=False, height=360,
                      margin=dict(l=20, r=20, t=70, b=30))
    return fig
