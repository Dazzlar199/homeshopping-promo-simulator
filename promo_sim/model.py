"""수요 반응 모델과 기준선 2개.

HierPoissonMAP: 계층 사전분포(상품 -> 상품군 -> 전체로 줄어드는 정규 사전분포)를 둔 포아송 회귀를
MAP(사후 최빈값)로 추정한다. 혜택 효과도 상품군별 기울기가 전체 기울기 쪽으로 줄어든다.
불확실성은 음이항 예측 분포로 내고, 학습 기간 끝 20%로 구간 폭을 보정한다.
노트북 수준의 가벼운 계산(L-BFGS 한 번)이라 수 초 안에 끝난다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor

from . import assumptions as A
from .data import FEATURES, promo_features

CAT_SLOPE_FEATURES = ["f_card", "f_salience", "f_noint", "f_gift"]
CATEGORIES = list(A.CATEGORIES)


@dataclass
class Design:
    products: list[str]
    names: list[str] = field(default_factory=list)
    penalty: np.ndarray | None = None

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        f = promo_features(df)
        dt = pd.to_datetime(df["date"])
        cols = [np.ones(len(df))]
        names = ["intercept"]
        pen = [1e-6]
        for c in CATEGORIES:
            cols.append((df["category"] == c).to_numpy(float)); names.append(f"cat:{c}"); pen.append(1 / 0.5**2)
        for p in self.products:
            cols.append((df["product_id"] == p).to_numpy(float)); names.append(f"prod:{p}"); pen.append(1 / 0.35**2)
        for s in A.SLOTS:
            cols.append((df["slot"] == s).to_numpy(float)); names.append(f"slot:{s}"); pen.append(1 / 1.0**2)
        for d in range(7):
            cols.append((dt.dt.day_of_week == d).to_numpy(float)); names.append(f"dow:{d}"); pen.append(1 / 1.0**2)
        for m in range(1, 13):
            cols.append((dt.dt.month == m).to_numpy(float)); names.append(f"month:{m}"); pen.append(1 / 0.5**2)
        for ft in FEATURES:
            cols.append(f[ft].to_numpy()); names.append(f"b:{ft}"); pen.append(1 / 10.0**2)
        for ft in CAT_SLOPE_FEATURES:
            for c in CATEGORIES:
                cols.append(f[ft].to_numpy() * (df["category"] == c).to_numpy(float))
                names.append(f"b:{ft}:{c}"); pen.append(1 / 1.0**2)
        self.names = names
        self.penalty = np.array(pen)
        return np.column_stack(cols)


def _fit_poisson_map(X, y, penalty, init=None):
    theta0 = np.zeros(X.shape[1]) if init is None else init.copy()
    if init is None:
        theta0[0] = np.log(max(y.mean(), 1.0))

    def fg(theta):
        eta = np.clip(X @ theta, -20, 20)
        mu = np.exp(eta)
        nll = np.sum(mu - y * eta) + 0.5 * np.sum(penalty * theta**2)
        grad = X.T @ (mu - y) + penalty * theta
        return nll, grad

    res = optimize.minimize(fg, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return res.x


def _dispersion(y, mu):
    num = np.sum(mu**2)
    den = np.sum((y - mu) ** 2 - mu)
    return float(np.clip(num / max(den, 1e-9), 0.5, 1e4))


def nb_interval(mu, k, level=0.8):
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    p = k / (k + mu)
    return stats.nbinom.ppf(lo_q, k, p), stats.nbinom.ppf(hi_q, k, p)


class HierPoissonMAP:
    name = "계층 포아송(MAP)"

    def __init__(self, level: float = 0.8):
        self.level = level

    def fit(self, df: pd.DataFrame, calibrate: bool = True) -> "HierPoissonMAP":
        df = df.sort_values(["date", "slot"]).reset_index(drop=True)
        self.design = Design(products=sorted(df["product_id"].unique()))
        y = df["orders"].to_numpy(float)
        X = self.design.matrix(df)
        self.theta = _fit_poisson_map(X, y, self.design.penalty)
        mu = np.exp(X @ self.theta)
        self.k = _dispersion(y, mu)
        self.k_eff = self.k
        if calibrate and len(df) > 200:
            cut = int(len(df) * 0.8)
            tr, ca = df.iloc[:cut], df.iloc[cut:]
            d2 = Design(products=self.design.products)
            th = _fit_poisson_map(d2.matrix(tr), tr["orders"].to_numpy(float), d2.penalty, init=self.theta)
            mu_ca = np.exp(d2.matrix(ca) @ th)
            y_ca = ca["orders"].to_numpy(float)
            best, best_gap = self.k, 1e9
            for mult in np.geomspace(0.15, 3.0, 40):
                lo, hi = nb_interval(mu_ca, self.k * mult, self.level)
                cov = np.mean((y_ca >= lo) & (y_ca <= hi))
                gap = abs(cov - self.level)
                if gap < best_gap - 1e-9:
                    best, best_gap = self.k * mult, gap
            self.k_eff = best
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(np.clip(self.design.matrix(df) @ self.theta, -20, 20))

    def interval(self, df: pd.DataFrame, level: float | None = None):
        return nb_interval(self.predict(df), self.k_eff, level or self.level)

    def sample(self, mu: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        k = self.k_eff
        p = k / (k + mu)
        return rng.negative_binomial(k, np.repeat(p[:, None], n, axis=1))

    def coef(self, name: str) -> float:
        return float(self.theta[self.design.names.index(name)])

    def slope(self, feature: str, category: str) -> float:
        total = self.coef(f"b:{feature}")
        if feature in CAT_SLOPE_FEATURES:
            total += self.coef(f"b:{feature}:{category}")
        return total


class RecentMeanBaseline:
    """기준선 1: 같은 상품의 최근 5회 평균(없으면 상품군x시간대 평균, 그다음 전체 평균). 혜택을 보지 않는다."""
    name = "최근 5회 평균"

    def fit(self, df):
        self.hist = df.sort_values(["date", "slot"])
        self.cat_slot = self.hist.groupby(["category", "slot"])["orders"].mean().to_dict()
        self.glob = float(self.hist["orders"].mean())
        self.by_prod = {p: g["orders"].to_numpy() for p, g in self.hist.groupby("product_id")}
        return self

    def predict(self, df):
        out = []
        for _, r in df.iterrows():
            h = self.by_prod.get(r["product_id"])
            if h is not None and len(h):
                out.append(h[-5:].mean())
            else:
                out.append(self.cat_slot.get((r["category"], r["slot"]), self.glob))
        return np.array(out, float)


class GBMBaseline:
    """기준선 2: 그래디언트 부스팅(scikit-learn HistGradientBoosting, 포아송 손실)."""
    name = "그래디언트 부스팅"

    def _X(self, df):
        f = promo_features(df)
        dt = pd.to_datetime(df["date"])
        X = pd.DataFrame({
            "prod": df["product_id"].map(self.prod_codes).fillna(-1).astype(int),
            "cat": df["category"].map({c: i for i, c in enumerate(CATEGORIES)}).astype(int),
            "slot": df["slot"].astype(int), "dow": dt.dt.day_of_week, "month": dt.dt.month,
        })
        for ft in FEATURES:
            X[ft] = f[ft].to_numpy()
        return X

    def fit(self, df):
        self.prod_codes = {p: i for i, p in enumerate(sorted(df["product_id"].unique()))}
        self.m = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.05,
                                               max_leaf_nodes=15, categorical_features=[0, 1], random_state=0)
        self.m.fit(self._X(df), df["orders"])
        return self

    def predict(self, df):
        return self.m.predict(self._X(df))


def naive_promo_glm(df: pd.DataFrame) -> dict:
    """교란 점검용: 상품·시간대 없이 혜택 특성만 넣은 포아송 회귀."""
    f = promo_features(df)
    X = np.column_stack([np.ones(len(df))] + [f[c].to_numpy() for c in FEATURES])
    pen = np.array([1e-6] + [1 / 10.0**2] * len(FEATURES))
    th = _fit_poisson_map(X, df["orders"].to_numpy(float), pen)
    return dict(zip(["intercept"] + FEATURES, th))
