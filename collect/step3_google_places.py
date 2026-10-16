"""3단계: AI 분석 지역 영업 중 식당 → Google Places(New) Text Search로 평점·리뷰 수·영업시간·전화 수집.

무료 한도 안에서만 받는다. 리뷰 본문(Atmosphere 필드)은 요청하지 않아서 'Text Search Enterprise' 요금 항목으로
잡히고, 이 항목은 월 1,000건까지 무료다(2026-10 기준). 여유를 두고 한 달에 MONTHLY_BUDGET 건만 부른다.
호출 수는 data/cache/google_usage.json 에 월별로 센다. 다음 달에 다시 실행하면 그 달 한도만큼 이어서 받는다.
받는 순서: 블로그 관련 글이 많은 가게부터 (구글에 등록돼 있을 가능성이 크고, 맛 점수가 많이 쓰이는 가게).

연희동 391곳은 예전에 리뷰 본문까지 받아 둔 캐시를 그대로 쓴다(감성분석 정답 데이터로 사용).
전화·영업시간·지도 링크도 Text Search 응답에 들어 있어서 새 지역은 Place Details를 따로 부르지 않는다(5단계가 캐시에서 정리).

사전 준비: Google Cloud에서 'Places API (New)' 사용 설정. 키는 secrets.toml 의 gmaps_api_key.
실행:  python step3_google_places.py [--budget 50]   (--budget: 이번 실행에서 부를 최대 건수)

출력:
  data/cache/google_places.json   mgt_no별 원본 응답 (재실행 시 이어서 수집)
  data/area_google.csv            매칭 결과 요약 (평점·리뷰 수)
"""
import argparse
import json
import math
import re
from difflib import SequenceMatcher

import pandas as pd

from common import CACHE, DATA, JsonCache, get_json, secret

URL = "https://places.googleapis.com/v1/places:searchText"
FIELDS = ",".join(f"places.{f}" for f in [
    "id", "displayName", "formattedAddress", "location", "rating",
    "userRatingCount", "priceLevel", "primaryType", "businessStatus",
    "googleMapsUri", "nationalPhoneNumber", "websiteUri", "regularOpeningHours",
])
MONTHLY_BUDGET = 900  # 무료 1,000건에서 재시도·테스트 여유를 뺀 값
USAGE = CACHE / "google_usage.json"
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


def search(key, name, place, lat, lng):
    body = {"textQuery": f"{name} {place}", "languageCode": "ko", "regionCode": "KR", "pageSize": 5}
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


def blog_counts(shops):
    """가게별 상호 언급 블로그 글 수 (4단계 결과). 없으면 0."""
    path = DATA / "area_blogs.csv"
    if not path.exists():
        return pd.Series(0, index=shops["mgt_no"])
    b = pd.read_csv(path, dtype={"mgt_no": str}, usecols=["mgt_no", "mentions_name"])
    return b[b["mentions_name"]].groupby("mgt_no").size().reindex(shops["mgt_no"], fill_value=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=None, help="이번 실행에서 부를 최대 건수 (월 한도 안에서)")
    args = ap.parse_args()

    shops = pd.read_csv(DATA / "area_restaurants.csv", dtype={"mgt_no": str})
    shops = shops[shops["closed"] == 0].reset_index(drop=True)
    cache = JsonCache("google_places")

    month = pd.Timestamp.now().strftime("%Y-%m")
    usage = json.loads(USAGE.read_text("utf-8")) if USAGE.exists() else {}
    left = MONTHLY_BUDGET - usage.get(month, 0)
    if args.budget is not None:
        left = min(left, args.budget)
    n_posts = blog_counts(shops)
    todo = shops[~shops["mgt_no"].isin(cache.data)].copy()
    todo["n_posts"] = n_posts.reindex(todo["mgt_no"]).values
    todo = todo[todo["n_posts"] > 0].sort_values("n_posts", ascending=False)  # 블로그 글이 없는 가게는 구글에도 거의 없다
    print(f"대상 {len(shops)}곳, 아직 안 받은 곳(블로그 글 있음) {len(todo)}곳, "
          f"{month} 사용 {usage.get(month, 0)}건 → 이번에 최대 {max(left, 0)}건")

    key = secret("gmaps_api_key")
    calls = 0
    for _, shop in todo.head(max(left, 0)).iterrows():
        try:
            cache[shop["mgt_no"]] = search(key, shop["name"], shop["area_name"], shop["lat"], shop["lng"])
        except RuntimeError as e:
            print(f"  ! {shop['name']}: {e}\n  중단합니다(받은 결과는 저장됨).")
            calls += 1
            break
        except Exception as e:
            print(f"  ! {shop['name']}: {e}")
            calls += 1
            continue
        calls += 1
        if calls % 20 == 0:
            cache.save()
            usage[month] = usage.get(month, 0) + 20
            USAGE.write_text(json.dumps(usage), "utf-8")
            print(f"  {calls}/{min(left, len(todo))}", flush=True)
    cache.save()
    usage[month] = usage.get(month, 0) + calls % 20
    USAGE.write_text(json.dumps(usage), "utf-8")
    print(f"이번 실행 {calls}건, {month} 누적 {usage[month]}건 (월 한도 {MONTHLY_BUDGET})")

    rows = []
    for _, shop in shops.iterrows():
        if shop["mgt_no"] not in cache:
            continue
        m = best_match(shop, cache[shop["mgt_no"]])
        rows.append({
            "mgt_no": shop["mgt_no"],
            "name": shop["name"],
            "place_id": m and m["id"],
            "g_name": m and m["displayName"]["text"],
            "rating": m and m.get("rating"),
            "rating_count": m and m.get("userRatingCount", 0),
            "price_level": m and m.get("priceLevel"),
            "primary_type": m and m.get("primaryType"),
            "business_status": m and m.get("businessStatus"),
            "n_reviews_text": len(m.get("reviews", [])) if m else 0,
            "match_sim": m and round(m["_sim"], 2),
            "match_dist_m": m and round(m["_dist"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "area_google.csv", index=False, encoding="utf-8-sig")
    print(f"구글 검색한 {len(out)}곳 중 매칭 {out['place_id'].notna().sum()}곳 → data/area_google.csv")


if __name__ == "__main__":
    main()
