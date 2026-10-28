import numpy as np

from promo_sim.backtest import card_effect_check, wape
from promo_sim.data import generate
from promo_sim.model import GBMBaseline, HierPoissonMAP, RecentMeanBaseline


def test_generator_deterministic():
    a, _, _ = generate(7)
    b, _, _ = generate(7)
    assert len(a) == len(b) and (a["orders"].to_numpy() == b["orders"].to_numpy()).all()
    assert 1400 < len(a) < 1800


def test_model_beats_recent_mean_on_holdout():
    df, _, _ = generate(7)
    tr, te = df[df["date"] < "2026-04-01"], df[df["date"] >= "2026-04-01"]
    y = te["orders"].to_numpy(float)
    hier = HierPoissonMAP().fit(tr)
    w_h = wape(y, hier.predict(te))
    w_r = wape(y, RecentMeanBaseline().fit(tr).predict(te))
    w_g = wape(y, GBMBaseline().fit(tr).predict(te))
    assert w_h < w_r
    assert w_h < 0.4 and w_g < 0.45
    lo, hi = hier.interval(te)
    cov = np.mean((y >= lo) & (y <= hi))
    assert 0.65 < cov < 0.92


def test_recovers_lever_effects_and_shows_confounding():
    df, _, truth = generate(7)
    hier = HierPoissonMAP().fit(df, calibrate=False)
    for c in ["주방가전", "침구·홈데코"]:
        assert abs(hier.slope("f_noint", c) - truth["noint"][c]) < 0.15
        assert abs(hier.slope("f_gift", c) - truth["gift"][c]) < 0.15
    card = card_effect_check(df, truth)
    # 단순 모델은 교란 때문에 할인 효과를 크게 부풀리고, 보정 모델은 참값에 더 가깝다
    err_adj = (card["보정 모델"] - card["참값"]).abs().mean()
    err_naive = (card["단순 모델(상품·시간대 미보정)"] - card["참값"]).abs().mean()
    assert err_adj < err_naive
