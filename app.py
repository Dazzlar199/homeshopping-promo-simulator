"""홈쇼핑 프로모션 시뮬레이터 — Streamlit 화면(합성 데이터 데모).

실행: .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_sim import assumptions as A
from promo_sim.agent import PromoAgent, ai_enabled, load_env, model_name
from promo_sim.levers import CARD_RATES, GIFT_CONDITIONS, GIFTS, ISSUERS, NOINT_MONTHS, POINT_RATES
from promo_sim.report import template_report, verify_numbers, won
from promo_sim.service import Simulator
from promo_sim.simulate import Constraints, default_base, lever_comparison, row_for

ROOT = Path(__file__).resolve().parent
APPROVALS = ROOT / "data" / "approvals.jsonl"
BACKTEST_JSON = ROOT / "docs" / "backtest.json"

st.set_page_config(page_title="홈쇼핑 프로모션 시뮬레이터", layout="wide")
load_env()


@st.cache_resource(show_spinner="합성 데이터 생성·모델 학습 중")
def get_sim() -> Simulator:
    return Simulator()


sim = get_sim()
products = sim.products

st.title("홈쇼핑 프로모션 시뮬레이터")
st.caption("합성 데이터 데모입니다. 실제 회사의 수치는 쓰지 않았고, 가정값과 출처는 '가정·출처' 탭에 있습니다.")
if ai_enabled():
    st.success(f"AI 연결 켜짐: OpenAI {model_name()} (요청 해석·리포트 작성). 숫자는 계산 코드만 만듭니다.")
else:
    st.info("AI 연결 꺼짐(OPENAI_API_KEY 없음): 규칙 해석과 틀 리포트로 같은 계산·추천을 합니다.")

# ------------------------------------------------------------------ 입력
with st.sidebar:
    st.header("방송 계획")
    labels = {r.product_id: f"{r.product_id} {r.name} ({r.price:,}원)" for r in products.itertuples()}
    pid = st.selectbox("상품", list(labels), format_func=labels.get, index=0)
    d = st.date_input("방송일", value=date(2026, 10, 13))
    slot = st.selectbox("시간대", A.SLOTS, index=A.SLOTS.index(21), format_func=lambda h: f"{h}시")
    plan = sim.plan(pid, d.isoformat(), slot)
    dc = sim.default_constraints(plan)
    st.header("제약")
    margin = st.slider("기여이익률 하한(%)", 0, 30, int(dc.margin_floor * 100))
    budget = st.number_input("카드 할인 우리 부담 예산(원)", 0, 50_000_000, dc.card_budget, step=500_000)
    gift_stock = st.number_input("사은품 재고(개)", 0, 10_000, dc.gift_stock, step=50)
    stock = st.number_input("상품 재고(개)", 100, 20_000, dc.product_stock, step=100)
    stockout = st.slider("품절 확률 상한(%)", 0, 100, int(dc.max_stockout_prob * 100))
    p10 = st.checkbox("하위 10% 시나리오도 기여이익 0 초과", value=True)
    max_card = st.selectbox("카드 할인율 상한", CARD_RATES, index=len(CARD_RATES) - 1, format_func=lambda r: f"{r*100:.0f}%")
    excl = st.multiselect("제외할 카드사", [x.name for x in ISSUERS])
    goal = st.radio("목표", ["균형", "이익", "매출", "위험"], horizontal=True)

cons = Constraints(margin_floor=margin / 100, card_budget=int(budget), gift_stock=int(gift_stock),
                   product_stock=int(stock), max_stockout_prob=stockout / 100, require_p10_positive=p10,
                   max_card_rate=float(max_card), exclude_issuers=excl, goal=goal)

tab_run, tab_cmp, tab_bt, tab_src, tab_log = st.tabs(["추천·승인", "유형별 비교", "백테스트", "가정·출처", "승인 기록"])

# ------------------------------------------------------------------ 말로 요청
with tab_run:
    req = st.text_area("요청(말로 적기, 선택)", placeholder="예: 밤 9시 방송, 기여이익률 18% 아래로는 안 돼. 사은품 재고 400개, D카드는 빼고 이익 최대로.")
    c1, c2 = st.columns([1, 3])
    run_req = c1.button("요청 해석해서 계산", disabled=not req.strip())
    fp = (plan, cons, req)
    if run_req:
        out = PromoAgent().run(req, sim.simulate_fn(plan, cons))
        out["result"] = sim.last
        st.session_state["agent_out"], st.session_state["agent_fp"] = out, fp
    out = st.session_state.get("agent_out") if st.session_state.get("agent_fp") == fp else None
    if out is None:
        result = sim.run(plan, cons)
        f = result.facts
        report = template_report(f)
        ver = verify_numbers(report, f)
        source, conds, calls = "화면 입력 + 틀 리포트", None, 0
    else:
        result = out["result"]
        f, report, ver = out["facts"], out["report"], out["verification"]
        source, conds, calls = out["source"], out["conditions"], out["calls"]

    if conds:
        with st.expander("요청을 이렇게 해석했습니다", expanded=True):
            st.json(conds)
    m1, m2, m3 = st.columns(3)
    m1.metric("조합 수", f["조합 수"])
    m2.metric("실질적으로 다른 조합", f["실질적으로 다른 조합"])
    m3.metric("제약 통과", f["제약 통과 조합"])

    recs = result.recs
    if not recs:
        st.error("제약을 모두 지키는 조합이 없습니다.")
        st.table(pd.Series(f["제약별 탈락 조합"], name="탈락 조합 수"))
    else:
        cols = st.columns(len(recs))
        for col, r, rf in zip(cols, recs, f["추천안"]):
            with col:
                st.subheader(r["label"])
                st.write(r["desc"])
                st.metric("예상 기여이익", rf["예상 기여이익"], help="80% 구간: " + rf["기여이익 80% 구간"])
                st.caption(f"기여이익 80% 구간 {rf['기여이익 80% 구간']} · 기여이익률 {rf['기여이익률']}")
                st.caption(f"예상 주문 {rf['예상 주문']} ({rf['주문 80% 구간']}) · 주문금액 {rf['예상 주문금액']}")
                st.caption(f"카드 우리 부담 {rf['카드 할인 우리 부담']} · 사은품 상위 90% {rf['사은품 지급(상위 90%)']} · 품절 확률 {rf['품절 확률']}")
                if rf["비고"]:
                    st.caption("비고: " + rf["비고"])

    st.markdown("#### 리포트")
    st.caption(f"작성: {source}" + (f" · API 호출 {calls}회" if calls else ""))
    if ver["passed"]:
        st.success(f"숫자 대조 통과: 리포트 숫자 {ver['n_numbers']}개가 모두 계산 결과에 있습니다.")
    else:
        st.error(f"숫자 대조 실패: {ver['unknown']}")
    st.text(report)

    st.markdown("#### 담당자 승인")
    if recs:
        with st.form("approve"):
            choice = st.radio("대상 추천안", [r["label"] for r in recs], horizontal=True)
            decision = st.radio("결정", ["승인", "수정 요청", "반려"], horizontal=True)
            who = st.text_input("담당자")
            memo = st.text_input("메모")
            if st.form_submit_button("기록"):
                rec = next(r for r in recs if r["label"] == choice)
                entry = {"time": datetime.now().isoformat(timespec="seconds"), "who": who, "decision": decision,
                         "memo": memo, "plan": vars(plan), "constraints": cons.to_dict(), "recommendation": choice,
                         "combo": rec["desc"], "expected_contrib": won(rec["row"]["contrib_mean"]),
                         "report_source": source, "numbers_verified": ver["passed"]}
                APPROVALS.parent.mkdir(exist_ok=True)
                with APPROVALS.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                st.success(f"{decision} 기록했습니다. 자동 집행은 하지 않습니다.")

    with st.expander("제약 통과 조합 상위 20개(기여이익 순)"):
        feas = result.res[result.res["feasible"]].drop_duplicates("effective_key")
        show = feas.sort_values("contrib_mean", ascending=False).head(20)
        st.dataframe(pd.DataFrame({
            "카드": [f"{r.issuer} {r.card_rate*100:.0f}%" if r.card_rate else "없음" for r in show.itertuples()],
            "무이자": show["noint_months"].map(lambda m: f"{m}개월"),
            "사은품": [f"{r.gift}({r.gift_cond})" if r.gift_value else "없음" for r in show.itertuples()],
            "적립": show["points_rate"].map(lambda p: f"{p*100:.0f}%"),
            "예상 주문": show["q_eff_mean"].round(0).astype(int),
            "기여이익(만 원)": (show["contrib_mean"] / 1e4).round(0).astype(int),
            "기여이익 하위 10%(만 원)": (show["contrib_p10"] / 1e4).round(0).astype(int),
            "기여이익률": (show["margin_ratio"] * 100).round(1),
        }), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ 유형별 비교
with tab_cmp:
    st.write("기준안에서 혜택 하나만 바꿨을 때 주문·주문금액·기여이익이 어떻게 달라지는지 봅니다.")
    b = default_base()
    c = st.columns(6)
    b["card_i"] = c[0].selectbox("카드 할인", range(len(CARD_RATES)), index=b["card_i"], format_func=lambda i: f"{CARD_RATES[i]*100:.0f}%")
    b["issuer_i"] = c[1].selectbox("카드사", range(len(ISSUERS)), index=b["issuer_i"], format_func=lambda i: ISSUERS[i].name)
    b["noint_i"] = c[2].selectbox("무이자", range(len(NOINT_MONTHS)), index=b["noint_i"], format_func=lambda i: f"{NOINT_MONTHS[i]}개월")
    b["gift_i"] = c[3].selectbox("사은품", range(len(GIFTS)), index=b["gift_i"], format_func=lambda i: GIFTS[i].name)
    b["cond_i"] = c[4].selectbox("사은품 조건", range(len(GIFT_CONDITIONS)), index=b["cond_i"], format_func=lambda i: GIFT_CONDITIONS[i].name)
    b["points_i"] = c[5].selectbox("적립", range(len(POINT_RATES)), index=b["points_i"], format_func=lambda i: f"{POINT_RATES[i]*100:.0f}%")
    br = row_for(result.res, b)
    st.caption(f"기준안: 예상 주문 {br['q_eff_mean']:,.0f}건, 기여이익 {won(br['contrib_mean'])}, 기여이익률 {br['margin_ratio']*100:.1f}%")
    cmp = lever_comparison(result.res, b)
    cmp["주문금액 변화(만 원)"] = (cmp.pop("주문금액 변화(원)") / 1e4).round(0).astype(int)
    cmp["기여이익 변화(만 원)"] = (cmp.pop("기여이익 변화(원)") / 1e4).round(0).astype(int)
    cmp["주문 변화"] = cmp["주문 변화"].round(0).astype(int)
    st.dataframe(cmp, hide_index=True, use_container_width=True)
    st.bar_chart(cmp.assign(항목=cmp["레버"] + ": " + cmp["바꾼 값"]).set_index("항목")["기여이익 변화(만 원)"])

# ------------------------------------------------------------------ 백테스트
with tab_bt:
    if BACKTEST_JSON.exists():
        bt = json.loads(BACKTEST_JSON.read_text(encoding="utf-8"))
        st.write(f"시간 순서 백테스트: {bt['test_period']}, 예측 회차 {bt['n_test']}회. 매달 그 이전 자료만으로 다시 학습했습니다.")
        st.dataframe(pd.DataFrame(bt["metrics"]), hide_index=True)
        st.write(f"80% 예측 구간 적중률 {bt['coverage']*100:.1f}% (목표 80%)")
        st.dataframe(pd.DataFrame(bt["monthly"]), hide_index=True)
        st.markdown("교란 점검: A카드 10% 할인의 주문 증가율(비프라임 시간대 참값 기준)")
        st.dataframe(pd.DataFrame(bt["card_effect"]), hide_index=True)
    else:
        st.warning("docs/backtest.json이 없습니다. `python scripts/run_backtest.py`를 먼저 실행하세요.")

# ------------------------------------------------------------------ 가정·출처
with tab_src:
    st.dataframe(pd.DataFrame([{"가정": a, "값의 근거": b, "출처": A.SOURCES[s]["label"],
                                "링크": " ".join(A.SOURCES[s]["urls"])} for a, b, s in A.ASSUMPTION_TABLE]),
                 hide_index=True, use_container_width=True)
    st.caption("임의 설정 값은 실제 현장 데이터가 들어오면 바꿔야 합니다.")

with tab_log:
    if APPROVALS.exists():
        rows = [json.loads(x) for x in APPROVALS.read_text(encoding="utf-8").splitlines() if x.strip()]
        st.dataframe(pd.DataFrame(rows)[["time", "who", "decision", "recommendation", "combo", "expected_contrib", "memo"]],
                     hide_index=True, use_container_width=True)
    else:
        st.write("아직 기록이 없습니다.")
