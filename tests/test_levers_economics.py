import numpy as np
import pandas as pd

from promo_sim import assumptions as A
from promo_sim.levers import combo_grid
from promo_sim.simulate import Economics, single_combo


def test_grid_size_and_unique():
    g = combo_grid()
    assert len(g) == 7200
    # 카드 0%면 카드사 무의미(1+4x6=25), 사은품 없음이면 조건 무의미(1+4x3=13): 25x4x13x4
    assert g["effective_key"].nunique() == 5200


def test_economics_hand_calc():
    g = single_combo(card_i=4, issuer_i=0, noint_i=1, gift_i=1, cond_i=0, points_i=1)  # A카드 10%, 3개월, 보관용기 전원, 1%
    eco = Economics(price=100_000, commission=0.3, base_return=0.1)
    q = np.array([[1000.0]])
    m = eco.compute(g, q, stock=5000)
    d = 0.10 * 0.22
    ret = 0.1 + A.RETURN_PER_DISCOUNT * (d + 0.01)
    q_net = 1000 * (1 - ret)
    expected = (0.3 * 100_000 * q_net
                - d * 100_000 * q_net * (1 - 0.5) * (1 - A.SUPPLIER_SHARE_CARD)
                - A.NOINT_FEE_PER_MONTH * 3 * A.INSTALLMENT_SHARE * 100_000 * q_net * (1 - A.SUPPLIER_SHARE_NOINT)
                - 1000 * 6_000 * (1 - A.SUPPLIER_SHARE_GIFT)
                - 0.01 * 100_000 * q_net)
    assert np.isclose(m["contrib"][0, 0], expected)
    assert m["gmv"][0, 0] == 100_000 * 1000


def test_stock_caps_orders_and_gift_cap():
    g = single_combo(card_i=0, issuer_i=0, noint_i=0, gift_i=2, cond_i=2, points_i=0)  # 선착순 300명
    eco = Economics(price=50_000, commission=0.3, base_return=0.05)
    m = eco.compute(g, np.array([[900.0, 100.0]]), stock=800)
    assert m["q_eff"][0, 0] == 800 and m["stockout"][0, 0] == 1.0
    assert m["gifts"][0, 0] == 300 and m["gifts"][0, 1] == 100
