"""시간 순서 백테스트(rolling-origin)와 교란 점검.

2025-01~2025-12로 처음 학습하고, 2026-01부터 한 달씩 앞으로 가며
"그 달 이전 자료만으로 학습 -> 그 달 예측"을 반복한다. 미래 자료는 학습에 섞이지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import assumptions as A
from .levers import ISSUERS
from .model import CATEGORIES, GBMBaseline, HierPoissonMAP, RecentMeanBaseline, naive_promo_glm


def wape(y, yhat):
    return float(np.sum(np.abs(y - yhat)) / np.sum(y))


def mape(y, yhat):
    return float(np.mean(np.abs(y - yhat) / np.maximum(y, 1)))


def run(df: pd.DataFrame, first_test_month: str = "2026-01", level: float = 0.8) -> dict:
    df = df.sort_values(["date", "slot"]).reset_index(drop=True)
    month = df["date"].str[:7]
    test_months = sorted(m for m in month.unique() if m >= first_test_month)
    preds = []
    for m in test_months:
        tr, te = df[month < m], df[month == m]
        hier = HierPoissonMAP(level=level).fit(tr)
        lo, hi = hier.interval(te)
        p = pd.DataFrame({
            "month": m, "orders": te["orders"].to_numpy(), "price": te["price"].to_numpy(),
            "hier": hier.predict(te), "lo": lo, "hi": hi,
            "recent": RecentMeanBaseline().fit(tr).predict(te),
            "gbm": GBMBaseline().fit(tr).predict(te),
        })
        preds.append(p)
    P = pd.concat(preds, ignore_index=True)
    y = P["orders"].to_numpy(float)
    models = [("hier", HierPoissonMAP.name), ("gbm", GBMBaseline.name), ("recent", RecentMeanBaseline.name)]
    metrics = []
    for col, label in models:
        yhat = P[col].to_numpy(float)
        metrics.append({
            "모델": label,
            "주문 WAPE": wape(y, yhat),
            "주문 MAPE": mape(y, yhat),
            "주문금액 WAPE": wape(y * P["price"], yhat * P["price"]),
        })
    cover = float(np.mean((y >= P["lo"]) & (y <= P["hi"])))
    width = float(np.mean((P["hi"] - P["lo"]) / np.maximum(P["hier"], 1)))
    monthly = P.groupby("month").apply(
        lambda g: pd.Series({"회차": len(g), "계층 WAPE": wape(g["orders"], g["hier"]),
                             "부스팅 WAPE": wape(g["orders"], g["gbm"]),
                             "최근평균 WAPE": wape(g["orders"], g["recent"]),
                             "80% 구간 적중률": float(np.mean((g["orders"] >= g["lo"]) & (g["orders"] <= g["hi"])))}),
        include_groups=False).reset_index()
    return {
        "n_test": int(len(P)), "test_months": test_months,
        "metrics": pd.DataFrame(metrics), "coverage": cover, "rel_width": width,
        "monthly": monthly, "pred": P,
    }


def card_effect_check(df: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """교란 점검: A카드 10% 할인이 주문을 몇 % 늘리는지(비프라임 시간대 기준) 참값·보정 모델·단순 모델 비교."""
    hier = HierPoissonMAP().fit(df, calibrate=False)
    naive = naive_promo_glm(df)
    share = ISSUERS[0].pay_share
    f_card = -np.log1p(-0.10 * share)
    rows = []
    for c in CATEGORIES:
        el = -truth["elasticity"][c]
        true_lift = np.exp(el * f_card + truth["salience"] * 0.10) - 1
        est_lift = np.exp(hier.slope("f_card", c) * f_card + hier.slope("f_salience", c) * 0.10) - 1
        rows.append({"상품군": c, "참값": true_lift, "보정 모델": est_lift})
    out = pd.DataFrame(rows)
    naive_lift = np.exp(naive["f_card"] * f_card + naive["f_salience"] * 0.10) - 1
    out["단순 모델(상품·시간대 미보정)"] = naive_lift
    return out


def lever_recovery(df: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """무이자·사은품 효과(상품군별 기울기)의 참값 복원."""
    hier = HierPoissonMAP().fit(df, calibrate=False)
    rows = []
    for c in CATEGORIES:
        rows.append({"상품군": c,
                     "무이자 참값": truth["noint"][c], "무이자 추정": hier.slope("f_noint", c),
                     "사은품 참값": truth["gift"][c], "사은품 추정": hier.slope("f_gift", c)})
    rows_df = pd.DataFrame(rows)
    rows_df.attrs["gap_true"] = truth["gap"]
    rows_df.attrs["gap_est"] = hier.coef("b:f_gap")
    return rows_df
