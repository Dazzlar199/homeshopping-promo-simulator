"""조정 에이전트: 담당자의 말로 된 요청 -> 계산 도구 호출 -> 리포트 -> 숫자 대조.

- OPENAI_API_KEY가 있을 때만 OpenAI Responses API(함수 호출)를 쓴다.
- 키가 없거나 호출이 실패하면 규칙 기반 해석기와 틀 리포트로 같은 흐름을 끝까지 돈다.
- 언어 모델은 숫자를 만들지 않는다. 계산은 run_simulation 도구(파이썬 코드)가 하고,
  리포트 숫자는 report.verify_numbers로 대조해 하나라도 다르면 틀 리포트로 바꾼다.
- 키 값은 어디에도 출력·기록하지 않는다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .levers import ISSUERS
from .report import template_report, verify_numbers
from .simulate import Constraints, Plan

DEFAULT_MODEL = "gpt-5.4-nano"     # 작고 싼 모델. OPENAI_MODEL 환경변수로 바꿀 수 있다
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
GOALS = ["균형", "이익", "매출", "위험"]
ISSUER_NAMES = [x.name for x in ISSUERS]


def load_env(path: Path = ENV_PATH) -> None:
    """.env의 KEY=VALUE를 환경변수로 올린다(이미 있는 값은 두고, 아무것도 출력하지 않는다)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v


def ai_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def model_name() -> str:
    return os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


# ---------------------------------------------------------------- 조건 스키마
def _nullable(t):
    return {"type": [t, "null"]}


CONDITION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "slot": {**_nullable("integer"), "description": "방송 시작 시각(6,8,10,13,16,19,21,23 중). 말이 없으면 null"},
        "margin_floor_pct": {**_nullable("number"), "description": "기여이익률 하한(%). 예: 15"},
        "card_budget_won": {**_nullable("integer"), "description": "카드 할인 우리 부담 예산(원)"},
        "gift_stock": {**_nullable("integer"), "description": "사은품 재고(개)"},
        "product_stock": {**_nullable("integer"), "description": "상품 재고(개)"},
        "max_card_rate_pct": {**_nullable("number"), "description": "카드 할인율 상한(%)"},
        "exclude_issuers": {"type": "array", "items": {"type": "string", "enum": ISSUER_NAMES},
                            "description": "빼야 할 카드사"},
        "goal": {"type": "string", "enum": GOALS, "description": "이익/매출/위험(보수적)/균형"},
        "summary": {"type": "string", "description": "요청을 어떻게 조건으로 바꿨는지 한 문장"},
    },
    "required": ["slot", "margin_floor_pct", "card_budget_won", "gift_stock", "product_stock",
                 "max_card_rate_pct", "exclude_issuers", "goal", "summary"],
}

TOOLS = [{
    "type": "function",
    "name": "run_simulation",
    "description": "프로모션 조합 7,200개를 계산하고 제약을 걸러 추천안 3개와 사실표를 돌려준다. 숫자는 이 도구만 만든다.",
    "parameters": CONDITION_SCHEMA,
    "strict": True,
}]

PARSE_RULES = (
    "조건 규칙: 담당자가 말하지 않은 숫자는 추측해 채우지 말고 null로 둔다(기본값이 쓰인다). "
    "목표(goal): 이익·수익을 최대로 → 이익, 매출·주문을 많이 → 매출, 보수적·손해 안 나게·위험 줄이기 → 위험, 목표 말이 없으면 → 균형. "
    "'이익률 15% 이상'처럼 하한을 말한 것은 목표가 아니라 margin_floor_pct다. "
)
SYSTEM = (
    "너는 홈쇼핑 프로모션 결정 보조 에이전트다. 담당자 요청을 읽고 run_simulation 도구를 정확히 한 번 호출한다. "
    + PARSE_RULES +
    "도구 결과를 받으면 한국어로 짧은 리포트를 쓴다. "
    "리포트에 쓰는 숫자는 도구 결과에 있는 문자열을 그대로 옮기고, 새 숫자를 계산하거나 반올림하거나 지어내지 않는다. "
    "추천안마다 조합, 예상 주문과 구간, 예상 기여이익과 구간, 기여이익률, 고른 이유 한 줄을 쓴다. "
    "추천안이 없으면 어떤 제약 때문인지 쓴다. 이모지와 과장 표현은 쓰지 않고, 끝에 제안이나 질문을 덧붙이지 않는다."
)


# ---------------------------------------------------------------- 규칙 기반 해석기(키 없을 때)
def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _won(amount: str, unit: str) -> int:
    v = _num(amount)
    return int(v * {"억": 1e8, "천만": 1e7, "만": 1e4, "원": 1}.get(unit, 1))


def parse_rules(text: str) -> dict:
    t = text.replace(" ", "")
    out: dict[str, Any] = {k: None for k in CONDITION_SCHEMA["properties"]}
    out["exclude_issuers"] = []
    m = re.search(r"(오전|오후|밤|저녁|아침)?(\d{1,2})시", t)
    if m:
        h = int(m.group(2))
        if m.group(1) in ("오후", "밤", "저녁") and h < 12:
            h += 12
        out["slot"] = h
    m = re.search(r"(?:마진|기여이익률|이익률)(?:하한)?(?:은|을|이|는)?(\d+(?:\.\d+)?)%", t)
    if m:
        out["margin_floor_pct"] = _num(m.group(1))
    m = re.search(r"카드(?:할인)?(?:부담)?예산(?:은|을|이|는)?(\d[\d,.]*)(억|천만|만|원)", t)
    if m:
        out["card_budget_won"] = _won(m.group(1), m.group(2))
    m = re.search(r"사은품(?:재고)?(?:은|을|이|는)?(\d[\d,]*)개", t)
    if m:
        out["gift_stock"] = int(_num(m.group(1)))
    m = re.search(r"(?:상품재고|재고)(?:은|을|이|는|가)?(\d[\d,]*)개", t.replace("사은품재고", "_"))
    if m:
        out["product_stock"] = int(_num(m.group(1)))
    m = re.search(r"카드(?:할인)?(?:율)?(?:은|을|는)?(?:최대|상한)?(\d+(?:\.\d+)?)%(?:까지|이하|넘지)", t)
    if m:
        out["max_card_rate_pct"] = _num(m.group(1))
    for name in ISSUER_NAMES:
        if re.search(name + r"(?:는|은)?(?:빼|제외)", t):
            out["exclude_issuers"].append(name)
    if re.search(r"보수|안전|위험|손해", t):
        out["goal"] = "위험"
    elif re.search(r"매출|취급고|주문을?많이|판매량", t):
        out["goal"] = "매출"
    elif re.search(r"이익(?!률)|수익|마진을?최대", t):
        out["goal"] = "이익"
    else:
        out["goal"] = "균형"
    changed = [k for k, v in out.items() if v not in (None, []) and k not in ("goal", "summary")]
    out["summary"] = f"규칙 해석: {', '.join(changed) if changed else '바뀐 조건 없음'}, 목표 {out['goal']}"
    return out


def apply_conditions(plan: Plan, cons: Constraints, c: dict) -> tuple[Plan, Constraints]:
    """해석한 조건을 계획·제약에 덮어쓴다(null은 기존 값 유지)."""
    from .assumptions import SLOTS
    p, k = plan, cons
    if c.get("slot") in SLOTS:
        p = replace(p, slot=int(c["slot"]))
    upd: dict[str, Any] = {}
    if c.get("margin_floor_pct") is not None:
        upd["margin_floor"] = float(c["margin_floor_pct"]) / 100
    if c.get("card_budget_won") is not None:
        upd["card_budget"] = int(c["card_budget_won"])
    if c.get("gift_stock") is not None:
        upd["gift_stock"] = int(c["gift_stock"])
    if c.get("product_stock") is not None:
        upd["product_stock"] = int(c["product_stock"])
    if c.get("max_card_rate_pct") is not None:
        upd["max_card_rate"] = float(c["max_card_rate_pct"]) / 100
    if c.get("exclude_issuers"):
        upd["exclude_issuers"] = list(c["exclude_issuers"])
    if c.get("goal") in GOALS:
        upd["goal"] = c["goal"]
    return p, replace(k, **upd)


# ---------------------------------------------------------------- 에이전트
SimulateFn = Callable[[dict], dict]   # 조건 dict -> 사실표 dict


class PromoAgent:
    def __init__(self, client=None, model: str | None = None, use_ai: bool | None = None):
        self.model = model or model_name()
        self.use_ai = ai_enabled() if use_ai is None else use_ai
        self.client = client
        if self.use_ai and self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        self.calls = 0

    def _create(self, **kw):
        self.calls += 1
        if self.model.startswith(("gpt-5", "o3", "o4")):
            kw.setdefault("reasoning", {"effort": "low"})
        return self.client.responses.create(model=self.model, **kw)

    def run(self, request: str, simulate: SimulateFn) -> dict:
        """요청 하나를 끝까지 처리한다. 반환: 조건, 사실표, 리포트, 대조 결과, 출처, 호출 수, 기록."""
        log: list[str] = []
        if self.use_ai:
            try:
                return self._run_ai(request, simulate, log)
            except Exception as e:  # 네트워크·한도·형식 오류 -> 규칙 경로로
                log.append(f"AI 경로 실패, 규칙 경로로 전환: {type(e).__name__}")
        cond = parse_rules(request)
        f = simulate(cond)
        rep = template_report(f)
        return dict(conditions=cond, facts=f, report=rep, verification=verify_numbers(rep, f),
                    source="규칙 해석 + 틀 리포트", calls=self.calls, log=log)

    def _run_ai(self, request: str, simulate: SimulateFn, log: list[str]) -> dict:
        msgs: list[Any] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}]
        resp = self._create(input=msgs, tools=TOOLS, tool_choice={"type": "function", "name": "run_simulation"})
        call = next((it for it in resp.output if getattr(it, "type", None) == "function_call"), None)
        if call is None:
            raise RuntimeError("도구 호출 없음")
        cond = json.loads(call.arguments)
        log.append(f"도구 호출: run_simulation {json.dumps(cond, ensure_ascii=False)}")
        f = simulate(cond)
        msgs += list(resp.output)
        msgs.append({"type": "function_call_output", "call_id": call.call_id,
                     "output": json.dumps(f, ensure_ascii=False)})
        resp2 = self._create(input=msgs, tools=TOOLS, tool_choice="none")
        rep = (resp2.output_text or "").strip()
        ver = verify_numbers(rep, f)
        if ver["passed"] and rep:
            return dict(conditions=cond, facts=f, report=rep, verification=ver,
                        source=f"OpenAI {self.model} (도구 호출 + 리포트)", calls=self.calls, log=log)
        log.append(f"리포트 숫자 대조 실패: {ver['unknown']} -> 틀 리포트로 교체")
        rep = template_report(f)
        return dict(conditions=cond, facts=f, report=rep, verification=verify_numbers(rep, f),
                    source=f"OpenAI {self.model} 해석 + 틀 리포트(대조 실패로 교체)", calls=self.calls, log=log)
