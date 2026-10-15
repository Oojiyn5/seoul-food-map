"""3단계: 연희동 영업 중 식당 → Google Places(New) Text Search로 평점·리뷰 수·리뷰(최대 5개) 수집.

리뷰 수(userRatingCount)는 제한이 없으므로 베이지안 평점 신뢰도 계산에 쓰고,
리뷰 본문 5개는 키워드 분석의 보조 자료로만 쓴다.

사전 준비: Google Cloud에서 'Places API (New)' 사용 설정. 키는 secrets.toml 의 gmaps_api_key.
실행:  python step3_google_places.py [--limit 20]   (처음엔 --limit 로 소량 테스트 권장)

출력:
  data/cache/google_places.json   mgt_no별 원본 응답 (재실행 시 이어서 수집)
  data/yeonhui_google.csv         매칭 결과 요약
"""
import argparse
import math
import re
from difflib import SequenceMatcher

import pandas as pd

from common import DATA, TARGET_DONG, JsonCache, get_json, secret

URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = ",".join(f"places.{f}" for f in [
    "id", "displayName", "formattedAddress", "location", "rating",
    "userRatingCount", "priceLevel", "primaryType", "reviews", "businessStatus",
])
MAX_DIST_M = 150     # 인허가 좌표와 이 거리 이내만 같은 가게로 인정
MIN_NAME_SIM = 0.7   # 상호 유사도 하한 (0.5에선 "이화정→이화원" 같은 오매칭이 많았음)


def norm(s):
    s = re.sub(r"\(.*?\)", "", str(s))
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s).lower()


def name_sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    # 포함 관계는 짧은 쪽이 2글자 이상일 때만 인정 ("닐" ⊂ "하우스오브바이닐" 방지)
    if min(len(a), len(b)) >= 2 and (a in b or b in a):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def haversine(lat1, lng1, lat2, lng2):
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def search(key, name, lat, lng):
    body = {"textQuery": f"{name} {TARGET_DONG}", "languageCode": "ko", "regionCode": "KR", "pageSize": 5}
    if not math.isnan(lat):
        body["locationBias"] = {"circle": {"center": {"latitude": lat, "longitude": lng}, "radius": 300.0}}
    headers = {"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS}
    return get_json(URL, method="POST", json=body, headers=headers).get("places", [])


def best_match(shop, places):
    best, best_score = None, 0.0
    for p in places:
        sim = name_sim(shop["name"], p.get("displayName", {}).get("text", ""))
        loc = p.get("location", {})
        dist = (haversine(shop["lat"], shop["lng"], loc["latitude"], loc["longitude"])
                if loc and not math.isnan(shop["lat"]) else 0.0)
        if sim >= MIN_NAME_SIM and dist <= MAX_DIST_M and sim > best_score:
            best, best_score = {**p, "_sim": sim, "_dist": dist}, sim
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    key = secret("gmaps_api_key")
    shops = pd.read_csv(DATA / "yeonhui_restaurants.csv", dtype={"mgt_no": str})
    shops = shops[shops["closed"] == 0]
    if args.limit:
        shops = shops.head(args.limit)

    cache = JsonCache("google_places")
    todo = shops[~shops["mgt_no"].isin(cache.data)]
    print(f"대상 {len(shops)}곳, 신규 요청 {len(todo)}곳")
    for i, (_, shop) in enumerate(todo.iterrows(), 1):
        try:
            cache[shop["mgt_no"]] = search(key, shop["name"], shop["lat"], shop["lng"])
        except RuntimeError as e:
            print(f"  ! {shop['name']}: {e}\n  중단합니다(받은 결과는 저장됨).")
            break
        except Exception as e:
            print(f"  ! {shop['name']}: {e}")
            continue
        if i % 20 == 0:
            cache.save()
            print(f"  {i}/{len(todo)}")
    cache.save()

    rows = []
    for _, shop in shops.iterrows():
        m = best_match(shop, cache.data.get(shop["mgt_no"], []))
        rows.append({
            "mgt_no": shop["mgt_no"],
            "name": shop["name"],
            "place_id": m and m["id"],
            "g_name": m and m["displayName"]["text"],
            "rating": m and m.get("rating"),
            "rating_count": m and m.get("userRatingCount", 0),
            "price_level": m and m.get("priceLevel"),
            "primary_type": m and m.get("primaryType"),
            "n_reviews_text": len(m.get("reviews", [])) if m else 0,
            "match_sim": m and round(m["_sim"], 2),
            "match_dist_m": m and round(m["_dist"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "yeonhui_google.csv", index=False, encoding="utf-8-sig")
    print(f"매칭 성공 {out['place_id'].notna().sum()}/{len(out)}곳 → data/yeonhui_google.csv")


if __name__ == "__main__":
    main()
