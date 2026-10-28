import json
from types import SimpleNamespace

import pytest

from promo_sim.agent import PromoAgent, apply_conditions, parse_rules
from promo_sim.report import template_report, verify_numbers


@pytest.fixture(scope="module")
def ctx(sim):
    plan = sim.plan("P01", "2026-10-13", 21)
    cons = sim.default_constraints(plan)
    return sim, plan, cons


def test_template_report_passes_verification(ctx):
    sim, plan, cons = ctx
    f = sim.run(plan, cons).facts
    rep = template_report(f)
    v = verify_numbers(rep, f)
    assert v["passed"] and v["n_numbers"] > 20


def test_fabricated_number_is_caught(ctx):
    sim, plan, cons = ctx
    f = sim.run(plan, cons).facts
    rep = template_report(f) + "\n참고로 예상 기여이익은 12,345,678만 원까지 늘 수 있습니다."
    v = verify_numbers(rep, f)
    assert not v["passed"] and "12345678" in v["unknown"]


def test_parser_cases_rules():
    from pathlib import Path
    cases = [json.loads(x) for x in (Path(__file__).parent / "parser_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(cases) == 100
    ok = 0
    for c in cases:
        got = parse_rules(c["text"])
        if all((float(got[k]) if isinstance(got[k], (int, float)) else got[k]) ==
               (float(c["expected"][k]) if isinstance(c["expected"][k], (int, float)) else c["expected"][k])
               for k in c["expected"]):
            ok += 1
    assert ok >= 95


def test_apply_conditions_keeps_nulls(ctx):
    _, plan, cons = ctx
    p, c = apply_conditions(plan, cons, {"slot": 19, "margin_floor_pct": 20, "card_budget_won": None, "gift_stock": None,
                                         "product_stock": None, "max_card_rate_pct": None, "exclude_issuers": ["B카드"],
                                         "goal": "위험", "summary": ""})
    assert p.slot == 19 and c.margin_floor == 0.2 and c.gift_stock == cons.gift_stock
    assert c.exclude_issuers == ["B카드"] and c.goal == "위험"


# ---------------------------------------------------------------- 가짜 OpenAI 클라이언트(목)
class FakeResponses:
    def __init__(self, report_text):
        self.report_text = report_text
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            args = {"slot": 21, "margin_floor_pct": 18, "card_budget_won": None, "gift_stock": 400, "product_stock": None,
                    "max_card_rate_pct": None, "exclude_issuers": ["D카드"], "goal": "이익", "summary": "가짜 해석"}
            item = SimpleNamespace(type="function_call", name="run_simulation", arguments=json.dumps(args, ensure_ascii=False), call_id="c1")
            return SimpleNamespace(output=[item], output_text="")
        text = self.report_text(kw) if callable(self.report_text) else self.report_text
        return SimpleNamespace(output=[], output_text=text)


def _facts_from_call(kw):
    return json.loads(kw["input"][-1]["output"])


def test_agent_mock_tool_loop_and_verified_report(ctx):
    sim, plan, cons = ctx
    fr = FakeResponses(lambda kw: template_report(_facts_from_call(kw)))
    agent = PromoAgent(client=SimpleNamespace(responses=fr), model="gpt-test", use_ai=True)
    out = agent.run("밤 9시, 기여이익률 18%, 사은품 400개, D카드 빼고 이익 최대", sim.simulate_fn(plan, cons))
    assert out["calls"] == 2 and out["verification"]["passed"]
    assert out["source"].startswith("OpenAI gpt-test")
    assert out["conditions"]["gift_stock"] == 400
    assert fr.calls[0]["tools"][0]["name"] == "run_simulation"
    assert fr.calls[1]["input"][-1]["type"] == "function_call_output"


def test_agent_mock_fabricated_report_falls_back(ctx):
    sim, plan, cons = ctx
    fr = FakeResponses("추천안 1의 예상 기여이익은 98,765만 원입니다.")
    agent = PromoAgent(client=SimpleNamespace(responses=fr), model="gpt-test", use_ai=True)
    out = agent.run("이익 최대로", sim.simulate_fn(plan, cons))
    assert out["verification"]["passed"]
    assert "틀 리포트" in out["source"]
    assert any("대조 실패" in x for x in out["log"])


def test_agent_api_error_falls_back_to_rules(ctx):
    sim, plan, cons = ctx

    class Boom:
        def create(self, **kw):
            raise ConnectionError("network down")
    agent = PromoAgent(client=SimpleNamespace(responses=Boom()), model="gpt-test", use_ai=True)
    out = agent.run("밤 9시 방송, 보수적으로", sim.simulate_fn(plan, cons))
    assert out["source"] == "규칙 해석 + 틀 리포트" and out["verification"]["passed"]
    assert out["conditions"]["goal"] == "위험"


def test_no_key_runs_offline(ctx, monkeypatch):
    sim, plan, cons = ctx
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = PromoAgent()
    assert agent.use_ai is False and agent.client is None
    out = agent.run("저녁 7시 방송, 매출을 최대한 올려", sim.simulate_fn(plan, cons))
    assert out["calls"] == 0 and out["conditions"]["slot"] == 19 and out["verification"]["passed"]
