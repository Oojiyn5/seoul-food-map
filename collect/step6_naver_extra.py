"""6단계: 예약·웨이팅·주차 정보용 네이버 블로그 추가 검색.

캐치테이블·테이블링은 공개 API가 없고, 구글 '인기 시간대'도 API에 없으므로
"{상권} {상호} 예약/웨이팅/주차" 검색 결과 요약문에서 언급을 찾는다(분석은 analysis/shop_info.py).
주제별 첫 페이지(100건)만 받는다.

실행:  python step6_naver_extra.py [--limit 20]
출력:
  data/cache/naver_extra.json     "mgt_no|주제"별 원본 (재실행 시 이어서 수집)
  data/area_blogs_extra.csv       글 단위 정리본 (topic 열 추가, 형식은 area_blogs.csv와 같음)
"""
import argparse
import time

import pandas as pd

from common import DATA, JsonlCache, get_json, secret
from step4_naver_blog import AD_PATTERN, URL, base_name, clean, is_ambiguous, mentions, name_keys

TOPICS = ["예약", "웨이팅", "주차"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    headers = {
        "X-NCP-APIGW-API-KEY-ID": secret("naver_client_id"),
        "X-NCP-APIGW-API-KEY": secret("naver_client_secret"),
    }
    shops = pd.read_csv(DATA / "area_restaurants.csv", dtype={"mgt_no": str})
    shops = shops[shops["closed"] == 0]
    if args.limit:
        shops = shops.head(args.limit)

    cache = JsonlCache("naver_extra")
    jobs = [(s["mgt_no"], s["name"], s["area_name"], t) for _, s in shops.iterrows() for t in TOPICS
            if f"{s['mgt_no']}|{t}" not in cache]
    print(f"대상 {len(shops)}곳 × {len(TOPICS)}주제, 신규 요청 {len(jobs)}건")
    for i, (mgt_no, name, place, topic) in enumerate(jobs, 1):
        params = {"query": f"{place} {base_name(name)} {topic}", "display": 100, "start": 1, "sort": "sim"}
        try:
            cache[f"{mgt_no}|{topic}"] = get_json(URL, params=params, headers=headers).get("items", [])
        except RuntimeError as e:
            print(f"  ! {name} {topic}: {e}\n  중단합니다(받은 결과는 저장됨).")
            break
        except Exception as e:
            print(f"  ! {name} {topic}: {e}")
            continue
        time.sleep(0.12)  # 초당 10회 제한 여유
        if i % 50 == 0:
            cache.save()
            print(f"  {i}/{len(jobs)} ({time.strftime('%H:%M')})", flush=True)
    cache.save()

    rows = []
    for _, shop in shops.iterrows():
        keys = name_keys(shop["name"])
        for topic in TOPICS:
            for it in cache.data.get(f"{shop['mgt_no']}|{topic}", []):
                title, desc = clean(it.get("title")), clean(it.get("description"))
                rows.append({
                    "mgt_no": shop["mgt_no"], "name": shop["name"], "area_name": shop["area_name"], "topic": topic,
                    "title": title, "description": desc, "link": it.get("link"),
                    "postdate": it.get("postdate"), "blogger": it.get("bloggername"),
                    "mentions_name": mentions(keys, title + " " + desc),
                    "name_ambiguous": is_ambiguous(shop["name"]),
                    "ad_suspect": bool(AD_PATTERN.search(title + " " + desc)),
                })
    out = pd.DataFrame(rows).drop_duplicates(["mgt_no", "link"])
    out.to_csv(DATA / "area_blogs_extra.csv", index=False, encoding="utf-8-sig")
    rel = out[out["mentions_name"]]
    print(f"글 {len(out):,}건 중 상호 언급 {len(rel):,}건 → data/area_blogs_extra.csv")
    print(rel.groupby("topic").size().to_string())


if __name__ == "__main__":
    main()
