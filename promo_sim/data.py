"""합성 방송 실적 데이터 생성기.

숨겨 둔 참값 효과로 주문 수를 만든다. 모델은 참값을 모르고 데이터만 보고 효과를 추정한다.
과거 혜택 배정은 일부러 치우치게 만든다(프라임 시간대·인기 상품에 큰 혜택이 몰림) — 교란 점검용.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import assumptions as A
from .levers import CARD_RATES, GIFTS, GIFT_CONDITIONS, ISSUERS, NOINT_MONTHS, POINT_RATES, attach_lever_values

FEATURES = ["f_card", "f_salience", "f_noint", "f_gift", "f_points", "f_gap"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = 0
    for cat, c in A.CATEGORIES.items():
        for j in range(A.PRODUCTS_PER_CATEGORY):
            n += 1
            lo, hi = c["price"]
            price = int(round(rng.uniform(lo, hi), -3)) - 100  # 99,900원 같은 끝자리
            rows.append(dict(
                product_id=f"P{n:02d}", name=f"{cat} 상품 {j + 1}", category=cat, price=price,
                commission=c["commission"], base_return=c["ret"],
                true_dev=float(rng.normal(0, 0.3)),
            ))
    return pd.DataFrame(rows)


def promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """혜택 조건을 모델 입력 특성으로 바꾼다. df에는 레버 값과 price, gap_days가 있어야 한다."""
    out = pd.DataFrame(index=df.index)
    out["f_card"] = -np.log1p(-(df["card_rate"] * df["pay_share"]))
    out["f_salience"] = df["card_rate"]
    out["f_noint"] = np.log1p(df["noint_months"]) / np.log(11)
    out["f_gift"] = np.log1p(5 * df["gift_value"] / df["price"] * df["cond_factor"])
    out["f_points"] = df["points_rate"]
    out["f_gap"] = np.log(np.minimum(df["gap_days"], 30) / 30)
    return out


def true_log_mu(df: pd.DataFrame, products: pd.DataFrame) -> np.ndarray:
    p = products.set_index("product_id").loc[df["product_id"]]
    cats = p["category"].to_numpy()
    base = np.array([A.CATEGORIES[c]["base"] for c in cats]) + p["true_dev"].to_numpy()
    el = -np.array([A.CATEGORIES[c]["elasticity"] for c in cats])
    noint = np.array([A.CATEGORIES[c]["noint"] for c in cats])
    gift = np.array([A.CATEGORIES[c]["gift"] for c in cats])
    f = promo_features(df)
    prime = df["slot"].isin(A.PRIME_SLOTS).to_numpy().astype(float)
    month = pd.to_datetime(df["date"]).dt.month.to_numpy()
    season = 0.08 * np.cos(2 * np.pi * (month - 12) / 12)  # 겨울 성수기(임의)
    lm = (base
          + df["slot"].map(A.SLOT_EFFECT).to_numpy()
          + np.array(A.DOW_EFFECT)[pd.to_datetime(df["date"]).dt.day_of_week.to_numpy()]
          + season
          + el * (f["f_card"].to_numpy() + A.TRUE_POINTS_WEIGHT * f["f_points"].to_numpy())
          + A.TRUE_SALIENCE * f["f_salience"].to_numpy() * (1 + A.TRUE_PRIME_CARD_BONUS * prime)
          + noint * f["f_noint"].to_numpy()
          + gift * f["f_gift"].to_numpy()
          + A.TRUE_GAP_EFFECT * f["f_gap"].to_numpy())
    return lm


def _biased_choice(rng, n_options, tilt):
    """tilt>0이면 큰 인덱스(큰 혜택)가 더 잘 뽑힌다."""
    w = np.exp(tilt * np.arange(n_options) / max(n_options - 1, 1))
    return int(rng.choice(n_options, p=w / w.sum()))


def generate(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    products = make_products(rng)
    dates = pd.date_range(A.START_DATE, A.END_DATE, freq="D")
    last_seen: dict[str, pd.Timestamp] = {}
    rows = []
    pop = np.exp(products["true_dev"].to_numpy())
    pick_w = pop / pop.sum()
    for d in dates:
        k = int(min(len(A.SLOTS), max(1, rng.poisson(A.BROADCASTS_PER_DAY))))
        slots = sorted(rng.choice(A.SLOTS, size=k, replace=False))
        prods = rng.choice(len(products), size=k, replace=False, p=pick_w)
        for slot, pi in zip(slots, prods):
            pr = products.iloc[pi]
            prime = slot in A.PRIME_SLOTS
            tilt = (1.6 if prime else -0.4) + 1.5 * pr["true_dev"]
            gap = (d - last_seen[pr["product_id"]]).days if pr["product_id"] in last_seen else 60
            last_seen[pr["product_id"]] = d
            gift_i = _biased_choice(rng, len(GIFTS), 0.8 if prime else -0.6)
            rows.append(dict(
                date=d.date().isoformat(), slot=int(slot), product_id=pr["product_id"],
                card_i=_biased_choice(rng, len(CARD_RATES), tilt),
                issuer_i=int(rng.integers(len(ISSUERS))),
                noint_i=_biased_choice(rng, len(NOINT_MONTHS), 0.5 if pr["price"] > 150_000 else -1.0),
                gift_i=gift_i,
                cond_i=int(rng.integers(len(GIFT_CONDITIONS))),
                points_i=_biased_choice(rng, len(POINT_RATES), -0.8),
                gap_days=int(gap),
            ))
    df = pd.DataFrame(rows)
    df = attach_lever_values(df)
    df = df.merge(products[["product_id", "category", "price"]], on="product_id", how="left")
    lm = true_log_mu(df, products)
    mu = np.exp(lm)
    k = A.DISPERSION_TRUE
    df["orders"] = rng.negative_binomial(k, k / (k + mu))
    df["true_mu"] = mu
    truth = {
        "dispersion": k,
        "elasticity": {c: v["elasticity"] for c, v in A.CATEGORIES.items()},
        "noint": {c: v["noint"] for c, v in A.CATEGORIES.items()},
        "gift": {c: v["gift"] for c, v in A.CATEGORIES.items()},
        "salience": A.TRUE_SALIENCE, "prime_card_bonus": A.TRUE_PRIME_CARD_BONUS,
        "points_weight": A.TRUE_POINTS_WEIGHT, "gap": A.TRUE_GAP_EFFECT,
    }
    return df, products, truth


def save(seed: int = 7, out: Path = DATA_DIR) -> None:
    df, products, truth = generate(seed)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["date", "slot", "product_id", "category", "price", "card_rate", "issuer", "noint_months",
            "gift", "gift_cond", "points_rate", "gap_days", "orders"]
    df[cols].to_csv(out / "broadcasts.csv", index=False)
    products.drop(columns=["true_dev"]).to_csv(out / "products.csv", index=False)
    (out / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")


def load(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """메모리에서 다시 생성한다(시드 고정이라 저장 파일과 같다)."""
    return generate(seed)
