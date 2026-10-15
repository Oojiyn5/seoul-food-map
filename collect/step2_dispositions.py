"""2단계: 식약처 행정처분결과(식품접객업, 서비스ID I2630) 수집 → 서울만 저장 → 연희동 식당과 매칭.

사전 준비:
  식품안전나라(https://www.foodsafetykorea.go.kr/api/) 회원가입 → 인증키 발급
  .streamlit/secrets.toml 에  foodsafety_api_key = "..."  추가

출력:
  data/raw/dispositions_raw.json      전국 원본 (재실행 시 재사용, --refresh 로 갱신)
  data/seoul_dispositions.csv         서울 처분 이력
  data/yeonhui_dispositions.csv       연희동 식당과 매칭된 처분 이력 (mgt_no 포함)

주의: 공개기한이 지난 처분은 지도에 표시하지 않는다(식품위생법상 공표기간).
"""
import json
import re
import sys

import pandas as pd

from common import DATA, RAW, TARGET_GU, get_json, secret

SERVICE = "I2630"
PAGE = 1000  # API 1회 최대 건수
RAW_PATH = RAW / "dispositions_raw.json"

# 응답 필드명 후보 (명세가 바뀌어도 동작하도록)
NAME_KEYS = ["BSSH_NM", "PRCSCITYPOINT_BSSHNM", "업소명"]
ADDR_KEYS = ["ADDR", "SITE_ADDR", "주소"]


def fetch_all(key):
    rows, start = [], 1
    while True:
        url = f"http://openapi.foodsafetykorea.go.kr/api/{key}/{SERVICE}/json/{start}/{start + PAGE - 1}"
        body = get_json(url).get(SERVICE, {})
        result = body.get("RESULT", {})
        if result.get("CODE") not in ("INFO-000", None):
            if result.get("CODE") == "INFO-200":  # 데이터 없음 = 끝
                break
            sys.exit(f"API 오류: {result}")
        page_rows = body.get("row", [])
        rows.extend(page_rows)
        total = int(body.get("total_count", 0))
        print(f"  {len(rows):,} / {total:,}")
        if not page_rows or len(rows) >= total:
            break
        start += PAGE
    return rows


def pick(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def norm_name(s):
    """상호 비교용: 공백·괄호 내용·특수문자 제거."""
    s = re.sub(r"\(.*?\)|\[.*?\]", "", str(s))
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s).lower()


def road_key(addr):
    """'서울특별시 서대문구 연희로 123-4, 1층' → '연희로 123-4' (도로명+건물번호)."""
    m = re.match(r"서울특별시\s+\S+구\s+(\S+\s+[\d-]+)", str(addr))
    return m.group(1) if m else None


def main():
    if RAW_PATH.exists() and "--refresh" not in sys.argv:
        rows = json.loads(RAW_PATH.read_text("utf-8"))
        print(f"저장된 원본 사용: {len(rows):,}건 (갱신하려면 --refresh)")
    else:
        print("행정처분 API 수집 중...")
        rows = fetch_all(secret("foodsafety_api_key"))
        RAW_PATH.write_text(json.dumps(rows, ensure_ascii=False), "utf-8")

    df = pd.DataFrame(rows)
    print(f"응답 필드: {list(df.columns)}")
    name_col, addr_col = pick(df.columns, NAME_KEYS), pick(df.columns, ADDR_KEYS)
    if not name_col or not addr_col:
        sys.exit("업소명/주소 필드를 찾지 못했습니다. 위 필드 목록을 보고 NAME_KEYS/ADDR_KEYS 를 수정하세요.")

    # "경기도 시흥시 서울대학로" 같은 주소가 섞이지 않도록 시작 부분으로 판별
    seoul = df[df[addr_col].str.startswith("서울특별시", na=False)].copy()
    seoul.to_csv(DATA / "seoul_dispositions.csv", index=False, encoding="utf-8-sig")
    print(f"서울 처분: {len(seoul):,}건")

    # 연희동 식당과 매칭: 같은 구 + 상호 일치, 동명 업소가 여럿이면 도로명 주소로 고른다
    shops = pd.read_csv(DATA / "yeonhui_restaurants.csv", dtype=str)
    shops["key"] = shops["name"].map(norm_name)
    gu = seoul[seoul[addr_col].str.contains(TARGET_GU, na=False)].copy()
    gu["key"] = gu[name_col].map(norm_name)
    matched = gu.merge(shops[["mgt_no", "name", "addr_jibun", "addr_road", "key"]], on="key", how="inner")
    matched["addr_match"] = matched[addr_col].map(road_key) == matched["addr_road"].map(road_key)
    n_cand = matched.groupby("DSPSDTLS_SEQ")["mgt_no"].transform("nunique")
    any_addr = matched.groupby("DSPSDTLS_SEQ")["addr_match"].transform("any")
    matched = matched[(n_cand == 1) | matched["addr_match"] | ~any_addr]

    matched.to_csv(DATA / "yeonhui_dispositions.csv", index=False, encoding="utf-8-sig")
    print(f"{TARGET_GU} 처분 {len(gu):,}건 중 연희동 식당과 매칭 {len(matched):,}건 "
          f"(식당 {matched['mgt_no'].nunique()}곳)")
    print("※ 동명 업소 오매칭 가능성이 있으니 addr 컬럼으로 육안 검수 권장")


if __name__ == "__main__":
    main()
