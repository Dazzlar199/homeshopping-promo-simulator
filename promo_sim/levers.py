"""프로모션 레버(혜택 항목)와 조합 격자.

5(카드 할인율) x 6(카드사) x 4(무이자 개월) x 5(사은품) x 3(사은품 조건) x 4(적립률) = 7,200개 조합.
값은 모두 데모용 가정이다(assumptions.py 참고).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

CARD_RATES = [0.0, 0.03, 0.05, 0.07, 0.10]
NOINT_MONTHS = [0, 3, 6, 10]
POINT_RATES = [0.0, 0.01, 0.03, 0.05]


@dataclass(frozen=True)
class Issuer:
    name: str
    pay_share: float      # 해당 카드로 결제할 수 있는 구매자 비중(가정)
    issuer_share: float   # 할인액 중 카드사가 내는 몫(가정)


ISSUERS = [
    Issuer("A카드", 0.22, 0.50),
    Issuer("B카드", 0.18, 0.40),
    Issuer("C카드", 0.15, 0.50),
    Issuer("D카드", 0.12, 0.30),
    Issuer("E카드", 0.10, 0.60),
    Issuer("F카드", 0.08, 0.40),
]


@dataclass(frozen=True)
class Gift:
    name: str
    value: int   # 고객이 느끼는 가치(정가 기준, 원)
    cost: int    # 사은품 원가(원)


GIFTS = [
    Gift("없음", 0, 0),
    Gift("보관용기 세트", 19_000, 6_000),
    Gift("여행용 파우치", 29_000, 9_000),
    Gift("미니 전기포트", 49_000, 17_000),
    Gift("프리미엄 텀블러 2개", 69_000, 24_000),
]


@dataclass(frozen=True)
class GiftCondition:
    name: str
    demand_factor: float   # 혜택 체감 정도(전원 증정 = 1.0)
    takeup: float | None   # 주문 중 사은품을 받는 비율. None이면 선착순 한도 사용
    cap: int | None = None


GIFT_CONDITIONS = [
    GiftCondition("전원 증정", 1.0, 1.0),
    GiftCondition("앱 주문 시", 0.6, 0.45),
    GiftCondition("선착순 300명", 0.8, None, 300),
]


def combo_grid() -> pd.DataFrame:
    """7,200개 조합 표. 인덱스 열 이름은 *_i."""
    rows = list(itertools.product(
        range(len(CARD_RATES)), range(len(ISSUERS)), range(len(NOINT_MONTHS)),
        range(len(GIFTS)), range(len(GIFT_CONDITIONS)), range(len(POINT_RATES)),
    ))
    g = pd.DataFrame(rows, columns=["card_i", "issuer_i", "noint_i", "gift_i", "cond_i", "points_i"])
    return attach_lever_values(g)


def attach_lever_values(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["card_rate"] = np.array(CARD_RATES)[g["card_i"]]
    g["issuer"] = [ISSUERS[i].name for i in g["issuer_i"]]
    g["pay_share"] = np.array([x.pay_share for x in ISSUERS])[g["issuer_i"]]
    g["issuer_share"] = np.array([x.issuer_share for x in ISSUERS])[g["issuer_i"]]
    g["noint_months"] = np.array(NOINT_MONTHS)[g["noint_i"]]
    g["gift"] = [GIFTS[i].name for i in g["gift_i"]]
    g["gift_value"] = np.array([x.value for x in GIFTS])[g["gift_i"]]
    g["gift_cost"] = np.array([x.cost for x in GIFTS])[g["gift_i"]]
    g["gift_cond"] = [GIFT_CONDITIONS[i].name for i in g["cond_i"]]
    g["cond_factor"] = np.array([c.demand_factor for c in GIFT_CONDITIONS])[g["cond_i"]]
    g["points_rate"] = np.array(POINT_RATES)[g["points_i"]]
    g["effective_key"] = effective_key(g)
    return g


def effective_key(g: pd.DataFrame) -> pd.Series:
    """실질적으로 같은 조합을 묶는 키. 카드 할인 0%면 카드사, 사은품 없음이면 조건이 의미 없다."""
    issuer = np.where(g["card_i"] == 0, -1, g["issuer_i"])
    cond = np.where(g["gift_i"] == 0, -1, g["cond_i"])
    return (g["card_i"].astype(str) + "-" + pd.Series(issuer, index=g.index).astype(str) + "-"
            + g["noint_i"].astype(str) + "-" + g["gift_i"].astype(str) + "-"
            + pd.Series(cond, index=g.index).astype(str) + "-" + g["points_i"].astype(str))


def describe(row) -> str:
    parts = []
    parts.append(f"{row['issuer']} {row['card_rate']*100:.0f}% 할인" if row["card_rate"] > 0 else "카드 할인 없음")
    parts.append(f"무이자 {row['noint_months']}개월" if row["noint_months"] > 0 else "무이자 없음")
    parts.append(f"사은품 {row['gift']}({row['gift_cond']})" if row["gift_value"] > 0 else "사은품 없음")
    parts.append(f"적립 {row['points_rate']*100:.0f}%" if row["points_rate"] > 0 else "적립 없음")
    return ", ".join(parts)
