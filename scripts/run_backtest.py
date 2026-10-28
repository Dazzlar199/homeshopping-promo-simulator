"""시간 순서 백테스트를 돌려 docs/backtest.json, docs/backtest.md를 만든다(노트북 수준, 수십 초)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promo_sim import backtest as B  # noqa: E402
from promo_sim.data import generate  # noqa: E402


def pct(x):
    return f"{x*100:.1f}%"


if __name__ == "__main__":
    df, products, truth = generate()
    r = B.run(df)
    tm = df[df["date"].str[:7] >= r["test_months"][0]]
    floor = B.wape(tm["orders"].to_numpy(float), tm["true_mu"].to_numpy())
    card = B.card_effect_check(df, truth)
    lev = B.lever_recovery(df, truth)
    out = {
        "test_period": f"{r['test_months'][0]} ~ {r['test_months'][-1]}", "n_test": r["n_test"],
        "n_rows": int(len(df)), "noise_floor_wape": floor, "coverage": r["coverage"], "rel_width": r["rel_width"],
        "metrics": r["metrics"].round(4).to_dict("records"),
        "monthly": r["monthly"].round(4).to_dict("records"),
        "card_effect": card.round(4).to_dict("records"),
        "lever_recovery": lev.round(3).to_dict("records"),
        "gap": {"true": lev.attrs["gap_true"], "est": round(lev.attrs["gap_est"], 3)},
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    m = r["metrics"]
    lines = [
        "# 백테스트 결과 (합성 데이터)", "",
        f"- 데이터: 방송 {len(df):,}회(2025-01-01 ~ 2026-09-30), 상품 30개",
        f"- 방식: 2026-01부터 한 달씩, 그 달 이전 자료만으로 다시 학습해 그 달을 예측(rolling-origin). 예측 회차 {r['n_test']}회",
        "- 합성 데이터라 실제 성능이 아니라 방법이 작동하는지 확인하는 용도다. 실제 데이터에서는 오차가 더 클 수 있다.", "",
        "## 예측 오차", "", "| 모델 | 주문 WAPE | 주문 MAPE | 주문금액 WAPE |", "|---|---:|---:|---:|",
    ]
    for row in m.itertuples(index=False):
        lines.append(f"| {row[0]} | {pct(row[1])} | {pct(row[2])} | {pct(row[3])} |")
    lines += ["", f"- 계층 모델 80% 예측 구간 적중률: {pct(r['coverage'])} (목표 80%), 평균 구간 폭은 예측값의 {pct(r['rel_width'])}",
              f"- 하한 참고: 참값 평균(데이터를 만든 식)으로 예측해도 같은 기간 주문 WAPE는 {pct(floor)}다. 남은 오차 대부분은 방송마다 생기는 우연한 변동이다.", "",
              "## 월별", "", "| 월 | 회차 | 계층 WAPE | 부스팅 WAPE | 최근평균 WAPE | 80% 구간 적중률 |", "|---|---:|---:|---:|---:|---:|"]
    for row in r["monthly"].itertuples(index=False):
        lines.append(f"| {row[0]} | {int(row[1])} | {pct(row[2])} | {pct(row[3])} | {pct(row[4])} | {pct(row[5])} |")
    lines += ["", "## 교란 점검: A카드 10% 할인의 주문 증가율", "",
              "과거 기록은 프라임 시간대·인기 상품에 큰 할인이 몰리도록 만들었다. 상품·시간대를 함께 넣은 모델과 넣지 않은 단순 모델을 비교한다(참값은 비프라임 시간대 기준).", "",
              "| 상품군 | 참값 | 보정 모델 | 단순 모델 |", "|---|---:|---:|---:|"]
    for row in card.itertuples(index=False):
        lines.append(f"| {row[0]} | {pct(row[1])} | {pct(row[2])} | {pct(row[3])} |")
    lines += ["", "## 혜택 효과 복원(상품군별 기울기)", "", "| 상품군 | 무이자 참값 | 무이자 추정 | 사은품 참값 | 사은품 추정 |", "|---|---:|---:|---:|---:|"]
    for row in lev.itertuples(index=False):
        lines.append(f"| {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.2f} | {row[4]:.2f} |")
    lines += ["", f"- 직전 방송 간격(당겨쓰기) 효과: 참값 {lev.attrs['gap_true']:.2f}, 추정 {lev.attrs['gap_est']:.2f}"]
    (ROOT / "docs" / "backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
