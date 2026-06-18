"""
Nelson's Rules - 관리도 이상판정 기준 (강의록 p.4 의 8개 규칙).

각 점에 대해 위반된 규칙 번호 집합을 반환한다.
시그마 구간은 단일 관리도의 (CL, UCL) 로부터 sigma = (UCL - CL)/3 로 계산한다.
점별 한계(P, U 관리도)도 평균 sigma 로 근사하여 적용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULE_DESCRIPTIONS = {
    1: "1점이 중심선에서 3σ를 벗어남 (관리한계 이탈)",
    2: "9점이 연속으로 중심선 같은 쪽에 존재",
    3: "6점이 연속으로 증가 또는 감소 (추세)",
    4: "14점이 연속으로 교대로 증감 (진동)",
    5: "연속 3점 중 2점이 같은 쪽 2σ를 벗어남",
    6: "연속 5점 중 4점이 같은 쪽 1σ를 벗어남",
    7: "15점이 연속으로 ±1σ 범위 안에 존재",
    8: "8점이 연속으로 ±1σ를 벗어남 (양쪽)",
}


def evaluate(chart: pd.DataFrame) -> dict:
    """관리도 DataFrame[point, CL, LCL, UCL] 에 대해 Nelson 규칙 적용.

    반환: {point_index_position: set(위반 규칙 번호)}
    내부적으로 0-기반 위치 인덱스를 사용한다.
    """
    x = chart["point"].to_numpy(dtype=float)
    cl = chart["CL"].to_numpy(dtype=float)
    sigma = (chart["UCL"].to_numpy(dtype=float) - cl) / 3.0
    n = len(x)

    # NaN(예: I-MR 첫 점) 제외용 마스크
    valid = ~np.isnan(x)

    violations = {i: set() for i in range(n)}

    def above_zone(i, k):
        """점 i 가 CL 기준 양수 방향으로 k*sigma 를 벗어났는가."""
        return valid[i] and sigma[i] > 0 and (x[i] - cl[i]) > k * sigma[i]

    def below_zone(i, k):
        return valid[i] and sigma[i] > 0 and (cl[i] - x[i]) > k * sigma[i]

    # Rule 1: 3σ 이탈
    for i in range(n):
        if not valid[i]:
            continue
        if x[i] > chart["UCL"].iloc[i] or x[i] < chart["LCL"].iloc[i]:
            violations[i].add(1)

    # Rule 2: 9점 연속 같은 쪽
    _run_same_side(x, cl, valid, length=9, target=violations, rule=2)

    # Rule 3: 6점 연속 증가/감소
    _run_trend(x, valid, length=6, target=violations, rule=3)

    # Rule 4: 14점 연속 교대 증감
    _run_alternating(x, valid, length=14, target=violations, rule=4)

    # Rule 5: 연속 3점 중 2점이 같은 쪽 2σ 이탈
    _k_of_m_beyond(above_zone, below_zone, n, valid, k_needed=2, m=3, zone=2,
                   target=violations, rule=5)

    # Rule 6: 연속 5점 중 4점이 같은 쪽 1σ 이탈
    _k_of_m_beyond(above_zone, below_zone, n, valid, k_needed=4, m=5, zone=1,
                   target=violations, rule=6)

    # Rule 7: 15점 연속 ±1σ 이내
    _run_within(x, cl, sigma, valid, length=15, target=violations, rule=7)

    # Rule 8: 8점 연속 ±1σ 밖(양쪽)
    _run_outside(x, cl, sigma, valid, length=8, target=violations, rule=8)

    return violations


def _run_same_side(x, cl, valid, length, target, rule):
    side = np.where(x > cl, 1, np.where(x < cl, -1, 0))
    count = 0
    for i in range(len(x)):
        if not valid[i] or side[i] == 0:
            count = 0
            continue
        if i > 0 and valid[i - 1] and side[i] == side[i - 1]:
            count += 1
        else:
            count = 1
        if count >= length:
            for j in range(i - length + 1, i + 1):
                target[j].add(rule)


def _run_trend(x, valid, length, target, rule):
    up = down = 1
    for i in range(1, len(x)):
        if not (valid[i] and valid[i - 1]):
            up = down = 1
            continue
        if x[i] > x[i - 1]:
            up += 1
            down = 1
        elif x[i] < x[i - 1]:
            down += 1
            up = 1
        else:
            up = down = 1
        if up >= length or down >= length:
            for j in range(i - length + 1, i + 1):
                target[j].add(rule)


def _run_alternating(x, valid, length, target, rule):
    alt = 1
    prev_dir = 0
    for i in range(1, len(x)):
        if not (valid[i] and valid[i - 1]):
            alt = 1
            prev_dir = 0
            continue
        d = np.sign(x[i] - x[i - 1])
        if d != 0 and d == -prev_dir:
            alt += 1
        else:
            alt = 2 if d != 0 else 1
        prev_dir = d
        if alt >= length:
            for j in range(i - length + 1, i + 1):
                target[j].add(rule)


def _k_of_m_beyond(above_zone, below_zone, n, valid, k_needed, m, zone, target, rule):
    for i in range(m - 1, n):
        window = range(i - m + 1, i + 1)
        if not all(valid[j] for j in window):
            continue
        if sum(above_zone(j, zone) for j in window) >= k_needed or \
           sum(below_zone(j, zone) for j in window) >= k_needed:
            for j in window:
                target[j].add(rule)


def _run_within(x, cl, sigma, valid, length, target, rule):
    count = 0
    for i in range(len(x)):
        within = valid[i] and sigma[i] > 0 and abs(x[i] - cl[i]) < sigma[i]
        count = count + 1 if within else 0
        if count >= length:
            for j in range(i - length + 1, i + 1):
                target[j].add(rule)


def _run_outside(x, cl, sigma, valid, length, target, rule):
    count = 0
    for i in range(len(x)):
        outside = valid[i] and sigma[i] > 0 and abs(x[i] - cl[i]) > sigma[i]
        count = count + 1 if outside else 0
        if count >= length:
            for j in range(i - length + 1, i + 1):
                target[j].add(rule)


def summary(violations: dict) -> pd.DataFrame:
    """위반 규칙을 표 형태로 요약."""
    rows = []
    for pos, rules in violations.items():
        if rules:
            rows.append(
                {
                    "위치": pos + 1,
                    "위반 규칙": ", ".join(f"R{r}" for r in sorted(rules)),
                    "설명": "; ".join(RULE_DESCRIPTIONS[r] for r in sorted(rules)),
                }
            )
    return pd.DataFrame(rows)
