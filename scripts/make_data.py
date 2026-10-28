"""합성 데이터를 data/에 저장한다(시드 고정, 다시 실행해도 같은 파일)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from promo_sim.data import save  # noqa: E402

if __name__ == "__main__":
    save()
    print("data/broadcasts.csv, data/products.csv, data/truth.json 저장")
