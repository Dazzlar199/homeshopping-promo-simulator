"""요청 해석 테스트 문장 100개를 만든다(tests/parser_cases.jsonl). 시드 고정.

규칙 해석기 회귀 테스트와, 키가 있을 때 AI 해석 채점(scripts/eval_parser.py --ai)에 같이 쓴다.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "parser_cases.jsonl"

SLOT = [("밤 9시", 21), ("21시", 21), ("저녁 7시", 19), ("오전 10시", 10), ("오후 1시", 13),
        ("아침 6시", 6), ("8시", 8), ("오후 4시", 16), ("밤 11시", 23), ("19시", 19)]
MARGIN = [("기여이익률 {v}% 아래로는 안 돼", None), ("마진 하한 {v}%", None), ("이익률은 {v}% 이상", None),
          ("기여이익률 하한 {v}%", None)]
GIFT = ["사은품 재고 {v}개", "사은품은 {v}개밖에 없어", "사은품 재고는 {v}개"]
STOCK = ["상품 재고 {v}개", "재고가 {v}개", "상품재고 {v}개"]
BUDGET = [("카드 예산 {v}만원", 10_000), ("카드 부담 예산 {v}만 원", 10_000), ("카드할인 예산은 {v}만원", 10_000)]
MAXCARD = ["카드 할인 최대 {v}%까지", "카드 할인율 {v}% 이하", "카드 할인은 {v}% 넘지 않게"]
EXCL = ["{v}는 빼 줘", "{v} 제외", "{v}은 빼고"]
GOAL = [("이익을 최대로", "이익"), ("매출을 최대한 올려", "매출"), ("보수적으로 손해 안 나게", "위험"),
        ("주문을 많이 받고 싶어", "매출"), ("수익 위주로", "이익"), ("", "균형")]


def case(rng):
    exp = {"slot": None, "margin_floor_pct": None, "card_budget_won": None, "gift_stock": None,
           "product_stock": None, "max_card_rate_pct": None, "exclude_issuers": [], "goal": "균형"}
    parts = []
    s, h = rng.choice(SLOT); parts.append(f"{s} 방송"); exp["slot"] = h
    for kind in rng.sample(["margin", "gift", "stock", "budget", "maxcard", "excl"], rng.randint(1, 4)):
        if kind == "margin":
            v = rng.choice([10, 12, 15, 18, 20, 22]); parts.append(rng.choice(MARGIN)[0].format(v=v)); exp["margin_floor_pct"] = float(v)
        elif kind == "gift":
            v = rng.choice([200, 300, 400, 500, 800]); parts.append(rng.choice(GIFT).format(v=v)); exp["gift_stock"] = v
        elif kind == "stock":
            v = rng.choice([1200, 1500, 2000, 2500, 3000]); parts.append(rng.choice(STOCK).format(v=f"{v:,}" if rng.random() < .5 else v)); exp["product_stock"] = v
        elif kind == "budget":
            v = rng.choice([100, 200, 300, 500]); t, mult = rng.choice(BUDGET); parts.append(t.format(v=v)); exp["card_budget_won"] = v * mult
        elif kind == "maxcard":
            v = rng.choice([3, 5, 7]); parts.append(rng.choice(MAXCARD).format(v=v)); exp["max_card_rate_pct"] = float(v)
        elif kind == "excl":
            v = rng.choice(["A카드", "B카드", "C카드", "D카드", "E카드", "F카드"]); parts.append(rng.choice(EXCL).format(v=v)); exp["exclude_issuers"] = [v]
    g, gl = rng.choice(GOAL)
    if g:
        parts.append(g)
    exp["goal"] = gl
    return {"text": ", ".join(parts) + ".", "expected": exp}


if __name__ == "__main__":
    rng = random.Random(42)
    rows = [case(rng) for _ in range(100)]
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)}개 저장: {OUT}")
