from dataclasses import replace

from promo_sim.simulate import Constraints, default_base, lever_comparison, recommend


def _run(sim, **kw):
    plan = sim.plan("P01", "2026-10-13", 21)
    cons = replace(sim.default_constraints(plan), **kw)
    return sim.run(plan, cons), cons


def test_recommendations_respect_constraints(sim):
    r, cons = _run(sim, margin_floor=0.18, gift_stock=400, exclude_issuers=["D카드"])
    assert 1 <= len(r.recs) <= 3
    labels = [x["label"] for x in r.recs]
    assert len(set(labels)) == len(labels)
    keys = [x["row"]["effective_key"] for x in r.recs]
    assert len(set(keys)) == len(keys)          # 추천안끼리 같은 조합이 아님
    for x in r.recs:
        row = x["row"]
        assert row["feasible"]
        assert row["margin_ratio"] >= 0.18
        assert row["gifts_p90"] <= 400
        assert row["card_ours_mean"] <= cons.card_budget
        assert row["stockout_prob"] <= cons.max_stockout_prob
        assert row["contrib_p10"] > 0
        assert not (row["card_rate"] > 0 and row["issuer"] == "D카드")


def test_best_profit_is_max_among_feasible(sim):
    r, _ = _run(sim)
    feas = r.res[r.res["feasible"]]
    profit = next(x for x in r.recs if x["label"] == "이익 최대")
    assert profit["row"]["contrib_mean"] == feas["contrib_mean"].max()


def test_deterministic(sim):
    a, _ = _run(sim)
    b, _ = _run(sim)
    assert [x["desc"] for x in a.recs] == [x["desc"] for x in b.recs]
    assert (a.res["contrib_mean"].to_numpy() == b.res["contrib_mean"].to_numpy()).all()


def test_infeasible_reports_reasons(sim):
    r, _ = _run(sim, margin_floor=0.9)
    assert r.recs == []
    assert r.facts["제약 통과 조합"] == "0개"
    assert int(r.facts["제약별 탈락 조합"]["기여이익률 하한"].replace(",", "").rstrip("개")) > 0


def test_lever_comparison_shape(sim):
    r, _ = _run(sim)
    cmp = lever_comparison(r.res, default_base())
    # 레버별 대안 수 합: 4+5+3+4+2+3 = 21
    assert len(cmp) == 21


def test_goal_changes_order(sim):
    r, _ = _run(sim, goal="위험")
    assert r.recs[0]["label"] == "위험 최소"
    assert recommend(r.res, "매출")[0]["label"] == "매출 최대"
