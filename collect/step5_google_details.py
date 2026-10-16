"""5단계: 구글에 매칭된 식당의 편의 정보(주차·예약·영업시간 등)를 Place Details로 수집.

구글 '인기 시간대(혼잡도)'는 공식 API에 없어서 받지 않는다(스크래핑은 약관 위반).
붐비는 정도는 블로그 웨이팅 언급으로 대신한다(6단계).

새 지역은 무료 한도 때문에 부르지 않고(3단계 Text Search 응답의 전화·영업시간을 씀), 받아 둔 연희동 캐시만 정리한다.
새로 부르려면 --fetch (Place Details Enterprise + Atmosphere 요금 항목, 월 1,000건 무료).

실행:  python step5_google_details.py [--fetch] [--limit 20]
출력:
  data/cache/google_details.json  place_id별 원본 응답 (재실행 시 이어서 수집)
  data/area_details.csv          가게별 정리본 (새 지역은 3단계 응답의 전화·영업시간·지도 링크)
"""
import argparse

import pandas as pd

from common import DATA, JsonCache, get_json, secret

URL = "https://places.googleapis.com/v1/places/{}"
FIELDS = ",".join([
    "id", "googleMapsUri", "nationalPhoneNumber", "websiteUri", "regularOpeningHours",
    "parkingOptions", "reservable", "goodForGroups", "goodForChildren", "allowsDogs",
    "outdoorSeating", "takeout", "delivery", "dineIn", "servesVegetarianFood", "restroom",
    "accessibilityOptions", "paymentOptions",
])
BOOL_FIELDS = ["reservable", "goodForGroups", "goodForChildren", "allowsDogs", "outdoorSeating",
               "takeout", "delivery", "dineIn", "servesVegetarianFood", "restroom"]
PARKING = ["freeParkingLot", "paidParkingLot", "freeStreetParking", "paidStreetParking",
           "valetParking", "freeGarageParking", "paidGarageParking"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fetch", action="store_true", help="캐시에 없는 가게를 API로 새로 받기 (유료 가능)")
    args = ap.parse_args()

    key = secret("gmaps_api_key")
    g = pd.read_csv(DATA / "area_google.csv", dtype={"mgt_no": str})
    g = g[g["place_id"].notna()]
    if args.limit:
        g = g.head(args.limit)

    cache = JsonCache("google_details")
    todo = g[~g["place_id"].isin(cache.data)] if args.fetch else g.iloc[:0]
    print(f"대상 {len(g)}곳, 신규 요청 {len(todo)}곳")
    headers = {"X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELDS}
    for i, pid in enumerate(todo["place_id"], 1):
        try:
            cache[pid] = get_json(URL.format(pid), params={"languageCode": "ko"}, headers=headers)
        except RuntimeError as e:
            print(f"  ! {pid}: {e}\n  중단합니다(받은 결과는 저장됨).")
            break
        except Exception as e:
            print(f"  ! {pid}: {e}")
            continue
        if i % 20 == 0:
            cache.save()
            print(f"  {i}/{len(todo)}")
    cache.save()

    # 상세 캐시에 없는 가게(새 지역)는 3단계 Text Search 응답에 든 전화·영업시간·지도 링크를 쓴다
    searched = {p["id"]: p for places in JsonCache("google_places").data.values() for p in places}
    rows = []
    for _, r in g.iterrows():
        d = cache.data.get(r["place_id"]) or searched.get(r["place_id"])
        if d is None:
            continue
        row = {"mgt_no": r["mgt_no"], "name": r["name"], "place_id": r["place_id"],
               "maps_url": d.get("googleMapsUri"), "phone": d.get("nationalPhoneNumber"),
               "website": d.get("websiteUri")}
        for f in BOOL_FIELDS:
            row[f] = d.get(f)  # 값이 없으면 None (= 구글에 정보 없음, '불가'가 아님)
        park = d.get("parkingOptions") or {}
        row["parking"] = ",".join(p for p in PARKING if park.get(p)) or (
            "없음" if park and not any(park.values()) else None)
        row["hours"] = " / ".join((d.get("regularOpeningHours") or {}).get("weekdayDescriptions", [])) or None
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "area_details.csv", index=False, encoding="utf-8-sig")
    fill = out.drop(columns=["mgt_no", "name", "place_id"]).notna().mean().map("{:.0%}".format)
    print(f"{len(out)}곳 → data/area_details.csv\n[필드별 값이 있는 비율]\n{fill.to_string()}")


if __name__ == "__main__":
    main()
