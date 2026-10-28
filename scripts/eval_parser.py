"""요청 해석 정확도 채점. 기본은 규칙 해석기, --ai면 OpenAI 해석(문장당 API 1회, --limit로 개수 제한)."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promo_sim.agent import CONDITION_SCHEMA, PARSE_RULES, PromoAgent, load_env, parse_rules  # noqa: E402

FIELDS = ["slot", "margin_floor_pct", "card_budget_won", "gift_stock", "product_stock",
          "max_card_rate_pct", "exclude_issuers", "goal"]


def norm(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def score(cases, parse):
    ok_case, ok_field, total_field, fails = 0, 0, 0, []
    for c in cases:
        got = parse(c["text"])
        diffs = [k for k in FIELDS if norm(got.get(k)) != norm(c["expected"][k])]
        total_field += len(FIELDS); ok_field += len(FIELDS) - len(diffs)
        if not diffs:
            ok_case += 1
        else:
            fails.append((c["text"], diffs))
    return ok_case / len(cases), ok_field / total_field, fails


def ai_parser():
    agent = PromoAgent(use_ai=True)

    def parse(text):
        resp = agent._create(input=[{"role": "system", "content": "홈쇼핑 담당자 요청을 프로모션 계산 조건으로 바꿔라. " + PARSE_RULES},
                                    {"role": "user", "content": text}],
                             text={"format": {"type": "json_schema", "name": "conditions", "schema": CONDITION_SCHEMA, "strict": True}})
        return json.loads(resp.output_text)
    return parse, agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", help="결과를 JSON으로 저장할 경로(예: docs/parser_eval_ai.json)")
    a = ap.parse_args()
    cases = [json.loads(x) for x in (ROOT / "tests" / "parser_cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()][: a.limit]
    if a.ai:
        load_env()
        parse, agent = ai_parser()
    else:
        parse, agent = parse_rules, None
    case_acc, field_acc, fails = score(cases, parse)
    print(f"{'AI' if a.ai else '규칙'} 해석: 문장 {len(cases)}개, 문장 단위 정확도 {case_acc*100:.1f}%, 항목 단위 {field_acc*100:.1f}%"
          + (f", API 호출 {agent.calls}회" if agent else ""))
    for t, d in fails[:10]:
        print(" 틀림:", t, d)
    if a.out:
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "parser": "ai" if a.ai else "rules",
               "model": __import__("promo_sim.agent", fromlist=["model_name"]).model_name() if a.ai else None,
               "n_cases": len(cases), "case_accuracy": round(case_acc, 4), "field_accuracy": round(field_acc, 4),
               "n_fields": len(cases) * len(FIELDS), "api_calls": agent.calls if agent else 0,
               "fails": [{"text": t, "wrong_fields": d} for t, d in fails]}
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("저장:", a.out)
