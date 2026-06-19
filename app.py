"""
SPC & 공정능력분석 웹앱 (Streamlit)
====================================
강의록(통계적공정관리 + 공정능력분석)에 기반한 대화형 분석 대시보드.

실행:  streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    capability as cap,
    constants as C,
    control_charts as cc,
    data_utils as du,
    nelson_rules as nr,
    time_weighted as tw,
    viz,
)

st.set_page_config(page_title="SPC & 공정능력분석", layout="wide", page_icon="📊")


# ===========================================================================
# 데이터 입력 (사이드바)
# ===========================================================================
def sidebar_data():
    st.sidebar.title("📊 SPC & 공정능력분석")
    st.sidebar.caption("강의록 기반 통합 분석 대시보드")
    st.sidebar.divider()

    source = st.sidebar.radio(
        "데이터 소스",
        ["CSV 업로드", "시뮬레이션 데이터 생성", "내장 예제"],
    )

    df = None
    kind = None  # 'value' or 'count'

    if source == "CSV 업로드":
        up = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
        if up is not None:
            df = pd.read_csv(up)
            kind = st.sidebar.radio("데이터 유형", ["계량형(value)", "계수형(count)"])
            kind = "value" if kind.startswith("계량") else "count"

    elif source == "시뮬레이션 데이터 생성":
        kind_sel = st.sidebar.radio("데이터 유형", ["계량형(value)", "계수형(count)"])
        kind = "value" if kind_sel.startswith("계량") else "count"
        with st.sidebar.expander("생성 파라미터", expanded=True):
            if kind == "value":
                num_sg = st.number_input("부분군 수", 5, 200, 30)
                sg_size = st.number_input("부분군 크기", 1, 25, 4)
                target = st.number_input("목표값(target)", value=40.0)
                sg_std = st.number_input("표준편차", 0.01, 100.0, 2.0)
                mean_shift = st.number_input("평균 시프트", 0.0, 50.0, 2.0)
                seed = st.number_input("랜덤시드", 0, 9999, 42)
                df = du.generate_value_data(
                    "value", target, "subgroup", int(num_sg), int(sg_size),
                    sg_std, mean_shift, 0, int(seed))
            else:
                num_sg = st.number_input("부분군 수", 5, 200, 50)
                sg_size = st.number_input("표본 크기", 10, 5000, 300)
                p = st.number_input("불량/결점률 p", 0.001, 0.5, 0.02, format="%.3f")
                var_sz = st.number_input("표본 크기 변동", 0, 500, 0)
                seed = st.number_input("랜덤시드", 0, 9999, 1)
                df = du.generate_count_data(
                    "count", "lot", int(num_sg), int(sg_size), p, int(var_sz), int(seed))

    else:  # 내장 예제
        ex = st.sidebar.selectbox(
            "예제 선택",
            ["PVC 점도 (공정능력)", "웨이퍼 두께 (Xbar-R)",
             "전구 불량 (NP/P)", "충전기 결함 (C/U)"])
        if ex.startswith("PVC"):
            data = np.array(
                [[3576.27, 3630.12, 3576.27, 3630.12, 3355.69, 3363.62],
                 [3504.17, 3514.52, 3747.43, 3666.15, 3709.25, 3317.28],
                 [3440.11, 3494.35, 3962.93, 3514.30, 3273.57, 3336.20],
                 [3638.33, 3719.84, 3617.47, 3450.17, 3378.70, 3475.50],
                 [3661.94, 3485.53, 3499.43, 3605.53, 3390.29, 3519.26]])
            wide = pd.DataFrame(data, columns=[f"pl_{i}" for i in range(1, 7)])
            df = wide.melt(var_name="prod_line", value_name="viscocity")
            kind = "value"
            st.session_state["spec"] = (4000.0, 3000.0)
        elif ex.startswith("웨이퍼"):
            df = du.generate_value_data("Thickness", 40, "Lot", 30, 4, 2, 2, 0, 42)
            kind = "value"
            st.session_state["spec"] = (42.0, 38.0)
        elif ex.startswith("전구"):
            df = du.generate_count_data("Defectives", "Lot", 50, 300, 0.02, 0, 1)
            kind = "count"
        else:
            df = du.generate_count_data("Defects", "Lot", 20, 500, 0.1, 0, 1)
            kind = "count"

    return df, kind, source


# ===========================================================================
# 컬럼 매핑 UI
# ===========================================================================
def column_mapping(df, kind):
    cols = list(df.columns)
    if len(cols) < 2:
        st.error("분석하려면 컬럼이 최소 2개 필요합니다 "
                 "(계량형: 부분군 + 측정값 / 계수형: 부분군 + 표본크기 + 관측값).")
        st.stop()

    # 컬럼 특성 분석: 측정값은 숫자이고 고유값 비율이 높다(거의 다 다른 값),
    # 부분군은 같은 값이 반복된다(고유값 비율이 낮다).
    numeric_cols = [c for c in cols
                    if pd.to_numeric(df[c], errors="coerce").notna().any()]
    non_numeric = [c for c in cols if c not in numeric_cols]

    def uniq_ratio(c):
        return df[c].nunique() / max(len(df), 1)

    def pick_value_col():
        """측정값 컬럼 추정: 숫자 컬럼 중 고유값 비율이 가장 높은 컬럼.
        동점이면 더 뒤쪽 컬럼(측정값은 라벨 뒤에 오는 경우가 많음)."""
        cand = numeric_cols if numeric_cols else cols
        return max(cand, key=lambda c: (uniq_ratio(c), cols.index(c)))

    def pick_subgroup_col(exclude):
        """부분군 컬럼 추정: 비숫자 우선, 아니면 고유값 비율이 가장 낮은 컬럼.
        동점이면 더 앞쪽 컬럼(라벨/ID는 앞에 오는 경우가 많음)."""
        rest = [c for c in cols if c != exclude]
        nn = [c for c in rest if c in non_numeric]
        pool = nn if nn else rest
        if not pool:
            return cols[0]
        return min(pool, key=lambda c: (uniq_ratio(c), cols.index(c)))

    c1, c2, c3 = st.columns(3)
    if kind == "value":
        val_default = pick_value_col()
        sg_default = pick_subgroup_col(exclude=val_default)

        sg = c1.selectbox("부분군 컬럼", cols, index=cols.index(sg_default))
        val = c2.selectbox("측정값 컬럼", cols, index=cols.index(val_default))
        if sg == val:
            st.warning("부분군 컬럼과 측정값 컬럼이 같습니다. 서로 다른 컬럼을 선택하세요.")
            st.stop()
        try:
            std = du.to_value_frame(df, sg, val)
        except Exception as e:
            st.error(f"데이터 변환 중 문제가 발생했습니다: {e}\n\n"
                     "측정값 컬럼에는 숫자 데이터를, 부분군 컬럼에는 라벨을 선택했는지 확인하세요.")
            st.stop()
        if std.empty:
            st.warning(f"측정값 컬럼 '{val}' 에 분석할 숫자 데이터가 없습니다. "
                       "측정값 컬럼에는 숫자 데이터를, 부분군 컬럼에는 라벨을 선택하세요.")
            st.stop()
        return std, sg, val, None
    else:
        sg_default = pick_subgroup_col(exclude=pick_value_col())
        guess_size = cols.index("sample_size") if "sample_size" in cols \
            else (cols.index(sg_default) + 1 if cols.index(sg_default) + 1 < len(cols) else 0)
        sg = c1.selectbox("부분군 컬럼", cols, index=cols.index(sg_default))
        size = c2.selectbox("표본크기 컬럼", cols, index=guess_size)
        val_default = next((c for c in cols if c not in (sg, size)), cols[-1])
        val = c3.selectbox("관측값 컬럼", cols, index=cols.index(val_default))
        if len({sg, size, val}) < 3:
            st.warning("부분군 · 표본크기 · 관측값 컬럼을 서로 다르게 선택하세요.")
            st.stop()
        try:
            std = du.to_count_frame(df, sg, size, val)
        except Exception as e:
            st.error(f"데이터 변환 중 문제가 발생했습니다: {e}\n\n"
                     "표본크기·관측값 컬럼에는 숫자 데이터를 선택했는지 확인하세요.")
            st.stop()
        if std.empty:
            st.warning("선택한 표본크기 / 관측값 컬럼에 숫자 데이터가 없습니다. 컬럼 선택을 확인하세요.")
            st.stop()
        return std, sg, val, size


# ===========================================================================
# 페이지: SPC (통계적 공정관리)
# ===========================================================================
def page_spc(std, kind, sg, val, size):
    st.header("통계적 공정관리 (SPC)")

    if kind == "value":
        rec = du.recommend_value_chart(std, sg)
        st.info(f"부분군 크기 기준 추천 관리도: **{rec}**  ·  "
                "작은 추세 변화 탐지에는 EWMA / CUSUM 사용")
        chart_options = ["Xbar-R", "Xbar-s", "I-MR", "EWMA", "CUSUM"]
        ctype = st.selectbox("관리도 종류", chart_options,
                             index=chart_options.index(rec))

        # ----- 시간가중형 관리도 (EWMA / CUSUM) -----
        if ctype in ("EWMA", "CUSUM"):
            c1, c2, c3 = st.columns(3)
            use_target = c1.checkbox("목표값 직접 지정", value=False)
            target = c1.number_input("목표값(μ)", value=float(std[val].mean())) \
                if use_target else None
            window = c2.number_input("σ 추정 윈도우(개별값일 때)", 2, 10, 2)

            if ctype == "EWMA":
                lam = c3.slider("가중계수 λ", 0.05, 1.0, 0.20, 0.05)
                L = c3.slider("관리한계 폭 L", 2.0, 3.5, 3.0, 0.1)
                chart = tw.ewma_chart(std, lam, L, target, int(window))
                st.plotly_chart(viz.ewma_figure(chart, f"EWMA 관리도 — {val}"),
                                use_container_width=True)
                sig = tw.ewma_signals(chart)
            else:
                k = c3.slider("기준값 계수 k", 0.25, 1.5, 0.5, 0.25)
                h = c3.slider("결정구간 계수 h", 3.0, 6.0, 5.0, 0.5)
                chart = tw.cusum_chart(std, k, h, target, int(window))
                st.plotly_chart(viz.cusum_figure(chart, f"CUSUM 관리도 — {val}"),
                                use_container_width=True)
                sig = tw.cusum_signals(chart)

            if sig:
                st.error(f"{ctype} 이상 신호 발생 부분군: {sig}  "
                         "→ 작은 변화가 누적되어 한계를 이탈했습니다.")
            else:
                st.success(f"{ctype}: 이상 신호 없음 — 관리상태로 판단됩니다.")
            st.caption("시간가중형 관리도는 작은 변화가 누적 반영되므로, "
                       "Shewhart 관리도가 놓치는 미세한 추세 변화 탐지에 유리합니다.")
            return

        window = 2
        if ctype == "I-MR":
            window = st.number_input("이동범위 윈도우(w)", 2, 10, 2)

        # 부분군 크기와 관리도 적합성 가드 (양방향)
        mode_size = int(du.subgroup_sizes(std, sg).mode().iloc[0])
        if ctype in ("Xbar-R", "Xbar-s") and mode_size < 2:
            st.warning(
                f"선택한 데이터는 부분군 크기가 {mode_size}(개별값)입니다. "
                "Xbar-R / Xbar-s 는 부분군 크기가 2 이상일 때만 의미가 있으므로, "
                "**I-MR** 관리도를 선택하세요.")
            return
        if ctype == "I-MR" and mode_size > 1:
            st.warning(
                f"선택한 데이터는 부분군 크기가 {mode_size}입니다. "
                "I-MR 은 부분군 크기가 1(개별값)일 때만 사용합니다. "
                f"부분군 크기 {mode_size}에는 **{rec}** 관리도를 선택하세요.")
            return

        recompute = st.checkbox("이상치 제거 후 관리도 재작성", value=False)
        if recompute:
            main, sub, removed = cc.recompute_value_chart(std, ctype, int(window))
            if removed:
                st.warning(f"제거된 이상 부분군(Lot): {removed}  "
                           f"({len(removed)}개 제거 → 재작성)")
            else:
                st.success("이상점이 없어 현재 관리한계를 채택합니다.")
        else:
            main, sub = cc.value_chart(std, ctype, int(window))

        names = {"Xbar-R": ["Xbar", "R"], "Xbar-s": ["Xbar", "s"],
                 "I-MR": ["I", "MR"]}[ctype]
        fig = viz.control_chart_figure((main, sub), names,
                                       f"{ctype} 관리도 — {val}")
        st.plotly_chart(fig, use_container_width=True)

        _nelson_report(main, names[0])

    else:  # 계수형
        const = du.is_constant_size(std, sg)
        options = ["P", "U"] + (["NP", "C"] if const else [])
        st.info("표본 크기 " + ("**동일** → NP/C 사용 가능" if const
                              else "**변동** → P/U 권장"))
        ctype = st.selectbox("관리도 종류", options)
        if ctype in ("C", "U"):
            work = std.rename(columns={val: "Defects"})
            (chart,) = cc.count_chart(work, ctype)
        else:
            (chart,) = cc.count_chart(std, ctype)
        fig = viz.control_chart_figure((chart,), [ctype],
                                       f"{ctype} 관리도 — {val}")
        st.plotly_chart(fig, use_container_width=True)
        _nelson_report(chart, ctype)


def _nelson_report(chart, name):
    st.subheader("Nelson's Rule 이상판정")
    v = nr.evaluate(chart)
    rep = nr.summary(v)
    if rep.empty:
        st.success(f"{name} 관리도: 위반 규칙 없음 — 관리상태로 판단됩니다.")
    else:
        st.error(f"{name} 관리도: {len(rep)}개 지점에서 이상 신호가 감지되었습니다.")
        st.dataframe(rep, use_container_width=True, hide_index=True)
    with st.expander("Nelson's Rule 8개 규칙 설명"):
        for k, d in nr.RULE_DESCRIPTIONS.items():
            st.write(f"**Rule {k}.** {d}")


# ===========================================================================
# 페이지: 공정능력분석
# ===========================================================================
def page_capability(std, kind, sg, val):
    st.header("공정능력분석 (Process Capability)")
    if kind != "value":
        st.warning("공정능력분석은 계량형(value) 데이터에만 적용됩니다.")
        return

    default = st.session_state.get("spec", (float(std[val].max()), float(std[val].min())))
    c1, c2, c3 = st.columns(3)
    usl = c1.number_input("규격상한 USL", value=float(default[0]))
    lsl = c2.number_input("규격하한 LSL", value=float(default[1]))
    window = c3.number_input("I-MR 윈도우(개별값일 때)", 2, 10, 2)

    if usl <= lsl:
        st.error("USL 은 LSL 보다 커야 합니다.")
        return

    res = cap.process_capability(std, sg, val, usl, lsl, int(window))

    # 정규성 결과
    n = res.normal
    if n.is_normal:
        st.success(f"정규성 검정(Shapiro-Wilk) p-value = {n.p_value:.4f} ≥ 0.05 → 정규성 만족")
    else:
        st.warning(f"정규성 검정 p-value = {n.p_value:.4f} < 0.05 → 정규성 불만족")
        if res.transformed:
            st.info(f"Box-Cox 변환을 적용했습니다 (λ = {res.lambda_:.4f}). "
                    "공정능력지수는 변환된 척도에서 계산됩니다.")

    # 지수 카드
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cp (단기)", f"{res.Cp:.4f}")
    m2.metric("Cpk (단기)", f"{res.Cpk:.4f}")
    m3.metric("Pp (장기)", f"{res.Pp:.4f}")
    m4.metric("Ppk (장기)", f"{res.Ppk:.4f}")

    g = cap.grade(res.Cpk)
    st.markdown(
        f"**판정(Cpk 기준):** 등급 {g['등급']} — {g['판정']}  ·  "
        f"권장조치: {g['시정조치']}  ·  {g['비고']}")

    # 그래프
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(
            viz.capability_figure(res.work_values, res.work_lsl, res.work_usl,
                                  res, var_name=val),
            use_container_width=True)
    with col_b:
        st.plotly_chart(viz.qq_figure(res.work_values, "Q-Q Plot"),
                        use_container_width=True)

    if res.transformed:
        st.plotly_chart(
            viz.boxcox_compare_figure(std[val].to_numpy(), res.work_values, res.lambda_),
            use_container_width=True)

    with st.expander("σ 추정값 및 판정표"):
        st.write(f"- 평균: {res.mean:.4f}")
        st.write(f"- σ_within (군내변동, 단기): {res.sigma_within:.4f}")
        st.write(f"- σ_overall (전체변동, 장기): {res.sigma_overall:.4f}")
        st.table(pd.DataFrame([
            {"Cp 범위": "Cp ≥ 1.67", "등급": 0, "판정": "매우 충분", "σ": "±5σ"},
            {"Cp 범위": "1.67 > Cp ≥ 1.33", "등급": 1, "판정": "충분", "σ": "±4σ"},
            {"Cp 범위": "1.33 > Cp ≥ 1.00", "등급": 2, "판정": "괜찮음", "σ": "±3σ"},
            {"Cp 범위": "1.00 > Cp ≥ 0.67", "등급": 3, "판정": "모자람", "σ": "±2σ"},
            {"Cp 범위": "0.67 > Cp", "등급": 4, "판정": "매우 부족", "σ": "±1σ"},
        ]))


# ===========================================================================
# 페이지: 종합 대시보드
# ===========================================================================
def page_dashboard(std, kind, sg, val, size):
    st.header("종합 대시보드")
    c1, c2, c3 = st.columns(3)
    c1.metric("부분군 수", std[sg].nunique())
    c2.metric("총 관측치", len(std))
    if kind == "value":
        c3.metric("전체 평균", f"{std[val].mean():.3f}")
    else:
        c3.metric("총 관측값", int(std[val].sum()))

    if kind == "value":
        rec = du.recommend_value_chart(std, sg)
        main, sub = cc.value_chart(std, rec)
        ooc = len(cc.out_of_control_subgroups(main)) + len(cc.out_of_control_subgroups(sub))
        v = nr.evaluate(main)
        signals = sum(1 for s in v.values() if s)
        d1, d2c = st.columns(2)
        d1.metric("관리한계 이탈 점", ooc)
        d2c.metric("Nelson 이상 신호 점", signals)
        st.plotly_chart(
            viz.control_chart_figure((main, sub),
                                     {"Xbar-R": ["Xbar", "R"], "Xbar-s": ["Xbar", "s"],
                                      "I-MR": ["I", "MR"]}[rec],
                                     f"{rec} 관리도 (자동 추천)"),
            use_container_width=True)

        spec = st.session_state.get("spec")
        if spec:
            res = cap.process_capability(std, sg, val, spec[0], spec[1])
            st.subheader("공정능력 요약")
            m = st.columns(4)
            m[0].metric("Cp", f"{res.Cp:.3f}")
            m[1].metric("Cpk", f"{res.Cpk:.3f}")
            m[2].metric("Pp", f"{res.Pp:.3f}")
            m[3].metric("Ppk", f"{res.Ppk:.3f}")
    else:
        const = du.is_constant_size(std, sg)
        ctype = "P" if not const else "P"
        (chart,) = cc.count_chart(std, ctype)
        st.plotly_chart(viz.control_chart_figure((chart,), [ctype],
                        f"{ctype} 관리도"), use_container_width=True)

    with st.expander("원본 데이터 보기"):
        st.dataframe(std, use_container_width=True)


# ===========================================================================
# 메인
# ===========================================================================
def main():
    df, kind, source = sidebar_data()

    if df is None:
        st.title("SPC & 공정능력분석 웹앱")
        st.markdown(
            "왼쪽 사이드바에서 **데이터를 업로드**하거나 **시뮬레이션/예제**를 선택하세요.\n\n"
            "- **통계적 공정관리(SPC)**: Xbar-R / Xbar-s / I-MR / NP / P / C / U 관리도 + Nelson's Rule\n"
            "- **공정능력분석**: 정규성 검정 · Box-Cox 변환 · Cp / Cpk / Pp / Ppk · 등급 판정\n"
            "- **종합 대시보드**: 현재 공정 상태 요약\n\n"
            "> 데이터가 바뀌면 모든 분석과 그래프가 자동으로 다시 계산됩니다.")
        return

    st.title("SPC & 공정능력분석 웹앱")
    std, sg, val, size = column_mapping(df, kind)

    tab1, tab2, tab3 = st.tabs(["📈 통계적 공정관리", "🎯 공정능력분석", "🗂 종합 대시보드"])
    with tab1:
        page_spc(std, kind, sg, val, size)
    with tab2:
        page_capability(std, kind, sg, val)
    with tab3:
        page_dashboard(std, kind, sg, val, size)


if __name__ == "__main__":
    main()
