"""합성 데이터 가정값과 출처.

공개 자료로 확인한 값은 출처를 적고, 확인할 수 없는 값은 "임의 설정"으로 표시한다.
특정 회사의 실제 수치는 하나도 쓰지 않는다.
"""
from __future__ import annotations

SOURCES = {
    "ftc_commission": {
        "label": "공정거래위원회 2025년 대형유통업체 판매수수료율 등 실태조사(2024년 거래 기준, 2025-12 발표) 보도 — TV홈쇼핑 실질수수료율 27.7%",
        "urls": [
            "https://news.mtn.co.kr/news-detail/2025122513185560099",
            "https://view.asiae.co.kr/article/2025122514260320018",
        ],
        "note": "보도 기사로 확인. 공정위 보도자료 원문 대조는 남음. 상품군별 최신 수치는 확인 못 해 상품군 편차는 임의 설정.",
    },
    "elasticity_meta": {
        "label": "Bijmolt, van Heerde & Pieters (2005), New Empirical Generalizations on the Determinants of Price Elasticity, Journal of Marketing Research 42(2), 141-156 — 81개 연구 1,851개 가격탄력성 평균 -2.62",
        "urls": ["https://repository.tilburguniversity.edu/items/813834f9-4626-45f8-ad0a-2920a545e089"],
        "note": "초록 기준. 상품군별 탄력성을 평균 -2.62 주변(-1.8~-3.4)에 임의 배치.",
    },
    "arbitrary": {
        "label": "임의 설정(공개 근거 없음, 본선 현장 데이터로 교체)",
        "urls": [],
        "note": "카드사 결제 비중·분담률, 사은품 가치·원가, 무이자 수수료, 반품률, 시간대 효과 등.",
    },
}

# 방송 시간대(시작 시각)와 시간대 효과(로그 배수, 임의 설정)
SLOTS = [6, 8, 10, 13, 16, 19, 21, 23]
SLOT_EFFECT = {6: -0.45, 8: -0.15, 10: 0.05, 13: -0.05, 16: -0.10, 19: 0.20, 21: 0.35, 23: 0.0}
PRIME_SLOTS = {19, 21}

# 요일 효과(월~일, 임의 설정)
DOW_EFFECT = [-0.05, -0.05, -0.02, 0.0, 0.05, 0.12, 0.10]

# 상품군: 가격 범위(원), 판매수수료율, 기본 반품률, 가격탄력성(참값), 무이자 민감도, 사은품 민감도
# 수수료율은 TV홈쇼핑 평균 27.7%(ftc_commission) 주변에 상품군별 편차를 임의로 둠.
CATEGORIES = {
    "주방가전": dict(price=(99_000, 399_000), commission=0.25, ret=0.08, elasticity=-2.2, noint=0.35, gift=0.30, base=6.4),
    "생활가전": dict(price=(149_000, 599_000), commission=0.24, ret=0.07, elasticity=-1.8, noint=0.45, gift=0.25, base=6.0),
    "이미용": dict(price=(49_000, 199_000), commission=0.30, ret=0.10, elasticity=-2.8, noint=0.15, gift=0.40, base=6.6),
    "패션잡화": dict(price=(39_000, 159_000), commission=0.31, ret=0.18, elasticity=-3.4, noint=0.10, gift=0.35, base=6.9),
    "건강식품": dict(price=(39_000, 129_000), commission=0.33, ret=0.05, elasticity=-2.6, noint=0.05, gift=0.45, base=7.0),
    "침구·홈데코": dict(price=(79_000, 299_000), commission=0.28, ret=0.12, elasticity=-2.4, noint=0.25, gift=0.30, base=6.3),
}

PRODUCTS_PER_CATEGORY = 5          # 6 x 5 = 30개 상품
START_DATE = "2025-01-01"
END_DATE = "2026-09-30"
BROADCASTS_PER_DAY = 2.4           # 약 1,500회
DISPERSION_TRUE = 9.0              # 음이항 분포 산포(클수록 포아송에 가까움, 임의 설정)

# 비용 분담(임의 설정)
SUPPLIER_SHARE_CARD = 0.5          # 카드사 몫을 뺀 할인액 중 협력사 부담 비율
SUPPLIER_SHARE_GIFT = 0.5          # 사은품 원가 중 협력사 부담 비율
NOINT_FEE_PER_MONTH = 0.004        # 무이자 할부 수수료(개월당, 결제액 대비)
INSTALLMENT_SHARE = 0.6            # 할부를 고르는 구매자 비중
SUPPLIER_SHARE_NOINT = 0.5
RETURN_PER_DISCOUNT = 0.5          # 실질 할인 1%p당 반품률 +0.5%p (충동구매 반품 가정)

# 참값 효과(합성 데이터 생성에만 씀. 모델은 이 값을 모른다)
TRUE_SALIENCE = 1.5                # 할인 '표시' 자체의 효과(할인율에 곱함)
TRUE_POINTS_WEIGHT = 0.5           # 적립금은 현금 할인의 절반 가치로 느낀다
TRUE_GAP_EFFECT = 0.18             # 같은 상품 직전 방송과 가까우면 수요 감소(당겨쓰기)
TRUE_PRIME_CARD_BONUS = 0.6        # 프라임 시간대에서 카드 할인 표시 효과가 더 큼(모델에 없는 상호작용)

ASSUMPTION_TABLE = [
    ("방송 시간대·요일 효과", "임의 설정", "arbitrary"),
    ("상품군 판매수수료율 24~33%", "TV홈쇼핑 평균 27.7% 주변에 상품군 편차 임의", "ftc_commission"),
    ("가격탄력성 -1.8~-3.4", "메타분석 평균 -2.62 주변", "elasticity_meta"),
    ("카드사별 결제 비중 8~22%, 카드사 분담 30~60%", "임의 설정", "arbitrary"),
    ("사은품 가치 1.9만~6.9만 원, 원가는 가치의 약 1/3", "임의 설정", "arbitrary"),
    ("무이자 수수료 개월당 0.4%, 할부 선택 60%", "임의 설정", "arbitrary"),
    ("상품군 반품률 5~18%, 실질 할인 1%p당 +0.5%p", "임의 설정", "arbitrary"),
    ("협력사 분담 50%(카드 할인 잔여분·사은품·무이자)", "임의 설정", "arbitrary"),
    ("음이항 산포 9", "임의 설정", "arbitrary"),
]
