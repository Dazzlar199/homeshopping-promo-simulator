"""조합 계산, 제약 걸러내기, 추천안 3개, 레버별 비교.

계산은 모두 이 파일의 결정적 코드가 한다(난수 시드 고정). 언어 모델은 숫자를 만들지 않는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import assumptions as A
from .levers import (CARD_RATES, GIFT_CONDITIONS, GIFTS, ISSUERS, NOINT_MONTHS, POINT_RATES,
                     attach_lever_values, combo_grid, describe)


@dataclass
class Plan:
    """방송 계획."""
    product_id: str
    date: str
    slot: int
    gap_days: int = 30


@dataclass
class Constraints:
    margin_floor: float = 0.15          # 기대 기여이익 / 기대 주문금액 하한
    card_budget: int = 3_000_000        # 카드 할인 중 우리 부담 기대값 상한(원)
    gift_stock: int = 500               # 사은품 재고(개). 지급 수 상위 90%가 넘으면 탈락
    product_stock: int = 2_000          # 상품 재고(개). 주문은 재고에서 잘린다
    max_stockout_prob: float = 0.2      # 품절 확률 상한
    require_p10_positive: bool = True   # 하위 10% 시나리오에서도 기여이익이 0보다 커야 함
    max_card_rate: float = 0.10
    exclude_issuers: list[str] = field(default_factory=list)
    goal: str = "균형"                   # 균형 / 이익 / 매출 / 위험

    def to_dict(self):
        return asdict(self)


@dataclass
class Economics:
    """주문 표본(행: 조합, 열: 시나리오)에서 금액 지표를 계산한다. 단위: 원."""
    price: float
    commission: float
    base_return: float

    def compute(self, g: pd.DataFrame, q: np.ndarray, stock: int) -> dict[str, np.ndarray]:
        q_eff = np.minimum(q, stock)
        d = (g["card_rate"] * g["pay_share"]).to_numpy()[:, None]
        pts = g["points_rate"].to_numpy()[:, None]
        ret = self.base_return + A.RETURN_PER_DISCOUNT * (d + pts)
        q_net = q_eff * (1 - ret)
        gmv = self.price * q_eff
        commission_rev = self.commission * self.price * q_net
        card_total = d * self.price * q_net
        card_ours = card_total * (1 - g["issuer_share"].to_numpy()[:, None]) * (1 - A.SUPPLIER_SHARE_CARD)
        noint_ours = (A.NOINT_FEE_PER_MONTH * g["noint_months"].to_numpy()[:, None] * A.INSTALLMENT_SHARE
                      * self.price * q_net * (1 - A.SUPPLIER_SHARE_NOINT))
        takeup = np.array([GIFT_CONDITIONS[i].takeup or 0.0 for i in g["cond_i"]])[:, None]
        cap = np.array([GIFT_CONDITIONS[i].cap or 0 for i in g["cond_i"]])[:, None]
        is_cap = cap > 0
        gifts = np.where(is_cap, np.minimum(q_eff, cap), q_eff * takeup)
        gifts = np.where(g["gift_value"].to_numpy()[:, None] > 0, gifts, 0)
        gift_ours = gifts * g["gift_cost"].to_numpy()[:, None] * (1 - A.SUPPLIER_SHARE_GIFT)
        points_ours = pts * self.price * q_net
        contrib = commission_rev - card_ours - noint_ours - gift_ours - points_ours
        return dict(q_eff=q_eff, gmv=gmv, contrib=contrib, card_ours=card_ours, gifts=gifts,
                    stockout=(q >= stock).astype(float))


def evaluate(model, products: pd.DataFrame, plan: Plan, cons: Constraints, n_sims: int = 400,
             seed: int = 0, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """조합마다 주문·주문금액·기여이익의 기대값과 10/50/90% 분위수를 계산한다."""
    g = combo_grid() if grid is None else grid
    pr = products.set_index("product_id").loc[plan.product_id]
    df = g.copy()
    df["product_id"] = plan.product_id
    df["category"] = pr["category"]
    df["price"] = float(pr["price"])
    df["slot"] = int(plan.slot)
    df["date"] = plan.date
    df["gap_days"] = int(plan.gap_days)
    mu = model.predict(df)
    rng = np.random.default_rng(seed)
    k = getattr(model, "k_eff", 10.0)
    shock = rng.gamma(k, 1.0 / k, size=n_sims)              # 모든 조합이 같은 시나리오(공통 난수)를 공유
    lam = mu[:, None] * shock[None, :]
    u = rng.random((len(df), n_sims))
    q = _poisson_from_uniform(lam, u)
    eco = Economics(price=float(pr["price"]), commission=float(pr["commission"]),
                    base_return=float(pr["base_return"]))
    m = eco.compute(g, q, cons.product_stock)
    out = g.copy()
    out["mu"] = mu
    for key in ("q_eff", "gmv", "contrib"):
        out[f"{key}_mean"] = m[key].mean(axis=1)
        out[f"{key}_p10"] = np.quantile(m[key], 0.1, axis=1)
        out[f"{key}_p50"] = np.quantile(m[key], 0.5, axis=1)
        out[f"{key}_p90"] = np.quantile(m[key], 0.9, axis=1)
    out["card_ours_mean"] = m["card_ours"].mean(axis=1)
    out["gifts_p90"] = np.quantile(m["gifts"], 0.9, axis=1)
    out["stockout_prob"] = m["stockout"].mean(axis=1)
    out["margin_ratio"] = out["contrib_mean"] / np.maximum(out["gmv_mean"], 1)
    return apply_constraints(out, cons)


def _poisson_from_uniform(lam: np.ndarray, u: np.ndarray) -> np.ndarray:
    """공통 난수로 포아송 표본을 만든다(평균이 큰 경우 정규 근사, 작은 경우 역함수)."""
    from scipy import stats
    big = lam > 50
    q = np.empty_like(lam)
    z = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
    q[big] = np.maximum(0, np.round(lam[big] + np.sqrt(lam[big]) * z[big]))
    if (~big).any():
        q[~big] = stats.poisson.ppf(u[~big], lam[~big])
    return q


CONSTRAINT_LABELS = {
    "ok_margin": "기여이익률 하한",
    "ok_card": "카드 할인 우리 부담 예산",
    "ok_gift": "사은품 재고",
    "ok_stock": "품절 확률 상한",
    "ok_p10": "하위 10% 기여이익 0 초과",
    "ok_levers": "허용 혜택 범위(카드 할인 상한·제외 카드사)",
}


def apply_constraints(out: pd.DataFrame, cons: Constraints) -> pd.DataFrame:
    out = out.copy()
    out["ok_margin"] = out["margin_ratio"] >= cons.margin_floor
    out["ok_card"] = out["card_ours_mean"] <= cons.card_budget
    out["ok_gift"] = out["gifts_p90"] <= cons.gift_stock
    out["ok_stock"] = out["stockout_prob"] <= cons.max_stockout_prob
    out["ok_p10"] = (out["contrib_p10"] > 0) if cons.require_p10_positive else True
    lev = out["card_rate"] <= cons.max_card_rate + 1e-9
    if cons.exclude_issuers:
        lev &= ~((out["card_rate"] > 0) & out["issuer"].isin(cons.exclude_issuers))
    out["ok_levers"] = lev
    out["feasible"] = out[list(CONSTRAINT_LABELS)].all(axis=1)
    return out


RECS = [
    ("이익 최대", "contrib_mean"),
    ("매출 최대", "gmv_mean"),
    ("위험 최소", "contrib_p10"),
]
GOAL_ORDER = {"이익": [0, 1, 2], "매출": [1, 0, 2], "위험": [2, 0, 1], "균형": [0, 2, 1]}


def recommend(res: pd.DataFrame, goal: str = "균형") -> list[dict]:
    """제약을 통과한 조합 중 목적별 최적안 3개. 실질적으로 같은 조합은 하나로 본다."""
    feas = res[res["feasible"]].drop_duplicates("effective_key")
    recs, used = [], set()
    for label, col in RECS:
        if feas.empty:
            break
        ranked = feas.sort_values([col, "contrib_mean"], ascending=False)
        row, note = ranked.iloc[0], ""
        if row["effective_key"] in used:
            rest = ranked[~ranked["effective_key"].isin(used)]
            if rest.empty:
                continue
            note = "1순위가 앞의 추천안과 같아 다음 순위를 보여 줌"
            row = rest.iloc[0]
        used.add(row["effective_key"])
        recs.append({"label": label, "criterion": col, "row": row, "desc": describe(row), "note": note})
    order = GOAL_ORDER.get(goal, GOAL_ORDER["균형"])
    return [recs[i] for i in order if i < len(recs)]


def infeasibility(res: pd.DataFrame) -> dict[str, int]:
    """제약별로 탈락한 조합 수."""
    return {CONSTRAINT_LABELS[c]: int((~res[c].astype(bool)).sum()) for c in CONSTRAINT_LABELS}


LEVERS = [
    ("카드 할인율", "card_i", [f"{r*100:.0f}%" for r in CARD_RATES]),
    ("카드사", "issuer_i", [x.name for x in ISSUERS]),
    ("무이자", "noint_i", [f"{m}개월" for m in NOINT_MONTHS]),
    ("사은품", "gift_i", [x.name for x in GIFTS]),
    ("사은품 조건", "cond_i", [c.name for c in GIFT_CONDITIONS]),
    ("적립률", "points_i", [f"{p*100:.0f}%" for p in POINT_RATES]),
]
IDX_COLS = ["card_i", "issuer_i", "noint_i", "gift_i", "cond_i", "points_i"]


def lever_comparison(res: pd.DataFrame, base: dict[str, int]) -> pd.DataFrame:
    """기준안에서 레버 하나만 바꿨을 때의 변화(프로모션 유형별 비교)."""
    key = res.set_index(IDX_COLS)
    b = key.loc[tuple(base[c] for c in IDX_COLS)]
    rows = []
    for label, col, names in LEVERS:
        for i, nm in enumerate(names):
            if i == base[col]:
                continue
            idx = dict(base); idx[col] = i
            r = key.loc[tuple(idx[c] for c in IDX_COLS)]
            rows.append({
                "레버": label, "바꾼 값": nm, "기준 값": names[base[col]],
                "주문 변화": r["q_eff_mean"] - b["q_eff_mean"],
                "주문금액 변화(원)": r["gmv_mean"] - b["gmv_mean"],
                "기여이익 변화(원)": r["contrib_mean"] - b["contrib_mean"],
                "제약 통과": bool(r["feasible"]),
            })
    return pd.DataFrame(rows)


def default_base() -> dict[str, int]:
    """비교 기준안: A카드 5% 할인, 무이자 6개월, 사은품 없음, 적립 없음."""
    return {"card_i": 2, "issuer_i": 0, "noint_i": 2, "gift_i": 0, "cond_i": 0, "points_i": 0}


def row_for(res: pd.DataFrame, idx: dict[str, int]) -> pd.Series:
    return res.set_index(IDX_COLS).loc[tuple(idx[c] for c in IDX_COLS)]


def single_combo(**idx) -> pd.DataFrame:
    return attach_lever_values(pd.DataFrame([idx]))
