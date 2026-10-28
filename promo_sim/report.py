"""리포트 사실표, 틀(template) 리포트, 숫자 대조 검사.

리포트에 쓰는 숫자는 facts()가 미리 문자열로 만들어 둔다. 언어 모델이 쓴 리포트든 틀 리포트든
verify_numbers()로 리포트 안 모든 숫자가 사실표에 있는지 대조하고, 하나라도 없으면 통과시키지 않는다.
"""
from __future__ import annotations

import re

import pandas as pd

LEVEL_PCT = 80
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
ALWAYS_OK = {"1", "2", "3", "10", "80", "90"}   # 목록 번호, 구간 수준(10/80/90%)


def won(x: float) -> str:
    """원 단위 금액을 만 원 단위 문자열로(예: 12,345만 원)."""
    return f"{round(x / 10_000):,}만 원"


def cnt(x: float) -> str:
    return f"{round(x):,}건"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def rec_facts(rec: dict) -> dict:
    r = rec["row"]
    return {
        "추천안": rec["label"],
        "조합": rec["desc"],
        "예상 주문": cnt(r["q_eff_mean"]),
        "주문 80% 구간": f"{cnt(r['q_eff_p10'])} ~ {cnt(r['q_eff_p90'])}",
        "예상 주문금액": won(r["gmv_mean"]),
        "예상 기여이익": won(r["contrib_mean"]),
        "기여이익 80% 구간": f"{won(r['contrib_p10'])} ~ {won(r['contrib_p90'])}",
        "기여이익률": pct(r["margin_ratio"]),
        "카드 할인 우리 부담": won(r["card_ours_mean"]),
        "사은품 지급(상위 90%)": f"{round(r['gifts_p90']):,}개",
        "품절 확률": pct(r["stockout_prob"], 0),
        "비고": rec.get("note", ""),
    }


def facts(plan, product: pd.Series, cons, recs: list[dict], n_combos: int, n_unique: int,
          n_feasible: int, infeas: dict[str, int], parsed_note: str = "") -> dict:
    return {
        "방송": f"{product['name']}({product['category']}), {plan.date} {plan.slot}시",
        "요청 해석": parsed_note,
        "조합 수": f"{n_combos:,}개",
        "실질적으로 다른 조합": f"{n_unique:,}개",
        "제약 통과 조합": f"{n_feasible:,}개",
        "제약": {
            "기여이익률 하한": pct(cons.margin_floor, 0),
            "카드 할인 우리 부담 예산": won(cons.card_budget),
            "사은품 재고": f"{cons.gift_stock:,}개",
            "상품 재고": f"{cons.product_stock:,}개",
            "품절 확률 상한": pct(cons.max_stockout_prob, 0),
        },
        "제약별 탈락 조합": {k: f"{v:,}개" for k, v in infeas.items()},
        "추천안": [rec_facts(r) for r in recs],
    }


def template_report(f: dict) -> str:
    lines = [f"[방송] {f['방송']}"]
    if f.get("요청 해석"):
        lines.append(f"[요청 해석] {f['요청 해석']}")
    lines.append(f"[계산] 조합 {f['조합 수']}(실질적으로 다른 조합 {f['실질적으로 다른 조합']}) 중 "
                 f"제약을 모두 통과한 조합은 {f['제약 통과 조합']}입니다.")
    if not f["추천안"]:
        worst = max(f["제약별 탈락 조합"].items(), key=lambda kv: int(kv[1].replace(",", "").rstrip("개")))
        lines.append(f"[결과] 제약을 모두 지키는 조합이 없습니다. 가장 많이 걸린 제약은 '{worst[0]}'({worst[1]} 탈락)입니다. "
                     "이 제약을 완화하거나 재고·예산을 다시 확인해 주세요.")
        return "\n".join(lines)
    for i, r in enumerate(f["추천안"], 1):
        lines.append(f"\n추천안 {i}. {r['추천안']}: {r['조합']}")
        lines.append(f"- 예상 주문 {r['예상 주문']}(80% 구간 {r['주문 80% 구간']}), 예상 주문금액 {r['예상 주문금액']}")
        lines.append(f"- 예상 기여이익 {r['예상 기여이익']}(80% 구간 {r['기여이익 80% 구간']}), 기여이익률 {r['기여이익률']}")
        lines.append(f"- 카드 할인 우리 부담 {r['카드 할인 우리 부담']}, 사은품 지급 상위 90% {r['사은품 지급(상위 90%)']}, 품절 확률 {r['품절 확률']}")
        if r["비고"]:
            lines.append(f"- 비고: {r['비고']}")
    lines.append("\n모든 숫자는 합성 데이터로 학습한 모델의 계산값이며, 승인 전 담당자 확인이 필요합니다.")
    return "\n".join(lines)


def _tokens(text: str) -> set[str]:
    return {t.replace(",", "") for t in _NUM.findall(text)}


def allowed_tokens(f) -> set[str]:
    out: set[str] = set(ALWAYS_OK)

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                out.update(_tokens(str(k)))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            out.update(_tokens(str(o)))
    walk(f)
    return out


def verify_numbers(report: str, f: dict) -> dict:
    """리포트 숫자가 모두 사실표에 있는지 대조한다."""
    found = _tokens(report)
    ok_set = allowed_tokens(f)
    bad = sorted(t for t in found if t not in ok_set)
    return {"passed": not bad, "n_numbers": len(found), "unknown": bad}
