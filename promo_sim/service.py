"""화면·스크립트가 함께 쓰는 실행 묶음: 데이터 -> 모델 -> 계산 -> 추천 -> 사실표."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agent import apply_conditions
from .data import generate
from .levers import combo_grid
from .model import HierPoissonMAP
from .report import facts as make_facts
from .simulate import (Constraints, Plan, default_base, evaluate, infeasibility, recommend,
                       single_combo)


@dataclass
class Result:
    plan: Plan
    cons: Constraints
    res: pd.DataFrame
    recs: list[dict]
    facts: dict


class Simulator:
    def __init__(self, seed: int = 7):
        self.df, self.products, self.truth = generate(seed)
        self.model = HierPoissonMAP().fit(self.df)
        self.grid = combo_grid()
        self.last: Result | None = None

    def product(self, pid: str) -> pd.Series:
        return self.products.set_index("product_id").loc[pid]

    def gap_days(self, pid: str, date: str) -> int:
        past = self.df[(self.df["product_id"] == pid) & (self.df["date"] < date)]
        if past.empty:
            return 60
        return int((pd.Timestamp(date) - pd.Timestamp(past["date"].max())).days)

    def plan(self, pid: str, date: str, slot: int) -> Plan:
        return Plan(product_id=pid, date=date, slot=int(slot), gap_days=self.gap_days(pid, date))

    def default_constraints(self, plan: Plan) -> Constraints:
        """상품 재고 기본값: 기준안 예상 주문의 1.6배(100개 단위)."""
        base = single_combo(**default_base())
        pr = self.product(plan.product_id)
        base = base.assign(product_id=plan.product_id, category=pr["category"], price=float(pr["price"]),
                           slot=plan.slot, date=plan.date, gap_days=plan.gap_days)
        mu = float(self.model.predict(base)[0])
        return Constraints(product_stock=int(max(300, round(mu * 1.6, -2))))

    def run(self, plan: Plan, cons: Constraints, parsed_note: str = "", seed: int = 0) -> Result:
        res = evaluate(self.model, self.products, plan, cons, seed=seed, grid=self.grid)
        recs = recommend(res, cons.goal)
        pr = self.product(plan.product_id)
        f = make_facts(plan, pr, cons, recs, n_combos=len(res), n_unique=int(res["effective_key"].nunique()),
                       n_feasible=int(res[res["feasible"]]["effective_key"].nunique()),
                       infeas=infeasibility(res), parsed_note=parsed_note)
        self.last = Result(plan, cons, res, recs, f)
        return self.last

    def simulate_fn(self, plan: Plan, cons: Constraints):
        """에이전트 도구용: 조건 dict를 받아 계산하고 사실표를 돌려준다."""
        def _fn(cond: dict) -> dict:
            p, c = apply_conditions(plan, cons, cond)
            if p.slot != plan.slot:
                p = self.plan(p.product_id, p.date, p.slot)
            return self.run(p, c, parsed_note=str(cond.get("summary", ""))).facts
        return _fn
