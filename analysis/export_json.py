"""지도용 데이터 내보내기: 분석 결과를 모아 web/restaurants.json 으로 저장한다.

⑤ 베이지안 평점: 리뷰가 적은 가게의 평점은 연희동 평균 쪽으로 당긴다.
   평점 = (C·m + n·r) / (C + n),  m = 연희동 리뷰 수 가중 평균, C = PRIOR_N
   하한 = 평점 − 1.28·σ/√(n + C)  (80% 신뢰구간 아래쪽, σ = 리뷰 별점 표준편차)
위생: 공표기간(최근 2년) 행정처분이 있으면 경고, 없으면 기준일과 함께 "공개된 처분 없음"

입력:  data/yeonhui_*.csv, data/yeonhui_shop_emb.npz, data/cache/google_places.json
출력:  web/restaurants.json
실행:  py export_json.py
"""
import base64
import json
import re
import sys
from urllib.parse import quote

import numpy as np
import pandas as pd

from prep import DATA, ROOT

sys.path.insert(0, str(ROOT / "collect"))
from step4_naver_blog import base_name, is_ambiguous  # noqa: E402

WEB = ROOT / "web"
WEB.mkdir(exist_ok=True)
DISPOSITION_DATE = "2026-10-14"   # step2 행정처분 조회일
PRIOR_N = 20                      # 베이지안 평점: 리뷰 20개 분량의 연희동 평균을 더한다
FEATURE_TEXT = {"지점(○○점)": ("지점(체인)", "개인 상호(지점 아님)")}  # SHAP 요인 표시 문구 (값 1, 값 0)


DAY_IDX = {"일요일": 0, "월요일": 1, "화요일": 2, "수요일": 3, "목요일": 4, "금요일": 5, "토요일": 6}  # JS getDay()와 같은 순서
TIME = re.compile(r"(오전|오후)?\s*(\d{1,2}):(\d{2})")


def to_min(mer, h, m):
    h, m = int(h), int(m)
    if mer == "오후" and h != 12:
        h += 12
    if mer == "오전" and h == 12:
        h = 0
    return h * 60 + m


def parse_hours(s):
    """구글 영업시간 문자열 → 요일별 [시작분, 끝분] 목록 (일요일=0). 자정을 넘기면 끝분 > 1440.
    예: '월요일: 오전 11:00 ~ 오후 3:00, 오후 5:00~10:00' → [[660, 900], [1020, 1320]].
    해석 못 한 요일은 None (= 모름)."""
    if not isinstance(s, str) or not s:
        return None
    week = [None] * 7
    for part in s.split(" / "):
        day, _, spec = part.partition(": ")
        if day not in DAY_IDX:
            continue
        spec = spec.strip()
        if spec == "휴무일":
            week[DAY_IDX[day]] = []
            continue
        if "24시간" in spec:
            week[DAY_IDX[day]] = [[0, 1440]]
            continue
        ranges = []
        for rng in spec.split(","):
            t = TIME.findall(rng)
            if len(t) != 2:
                ranges = None
                break
            (m1, h1, n1), (m2, h2, n2) = t
            start = to_min(m1, h1, n1)
            end = to_min(m2 or m1, h2, n2)        # 끝 시각에 오전/오후가 없으면 시작 것을 따른다
            if end <= start and not m2 and to_min("오후", h2, n2) > start:
                end = to_min("오후", h2, n2)       # '오전 11:00~3:00' 같은 경우
            if end <= start:
                end += 1440                        # 자정 넘김
            ranges.append([start, end])
        week[DAY_IDX[day]] = ranges
    return week if any(w is not None for w in week) else None


def read(name):
    return pd.read_csv(DATA / f"{name}.csv", dtype={"mgt_no": str})


def nz(v):
    """NaN → None (JSON null)."""
    if isinstance(v, (float, np.floating)) and np.isnan(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def review_sd():
    c = json.loads((DATA / "cache" / "google_places.json").read_text("utf-8"))
    stars, status = [], {}
    for places in c.values():
        for p in places:
            status[p["id"]] = p.get("businessStatus")
            stars += [r["rating"] for r in p.get("reviews", [])]
    return float(np.std(stars)), status


def factor_text(s, is_branch):
    if not isinstance(s, str) or not s:
        return []
    out = []
    for f in s.split(", "):
        if f in FEATURE_TEXT:
            f = FEATURE_TEXT[f][0 if is_branch else 1]
        out.append(f)
    return out


def merge_same_place(shops, blog):
    """한 가게가 인허가를 여러 개 가진 경우를 하나로 합친다. 블로그 글이 겹쳐 지도에 두 번 나오기 때문.
    - 같은 건물이고 상호가 서로 포함됨 (예: (주)카덴 2층 / (주)우동카덴 1층)
    - 상호가 같고 60m 이내 (예: 한우명가 연희로 46-1 / 48, 별관을 따로 인허가)
    남기는 기록은 오래된 인허가(영업 연수를 제대로 보여주려고), 블로그·구글 정보는 글이 많은 쪽을 쓴다."""
    key = lambda n: re.sub(r"\(주\)|주식회사|㈜|\(.*?\)|\s", "", n)  # noqa: E731
    shops = shops.assign(_bld=shops["addr_road"].str.split(",").str[0], _key=shops["name"].map(key))
    n_posts = blog.set_index("mgt_no")["n_posts"]
    recs = list(shops.to_dict("records"))
    drop, note, borrow = set(), {}, {}
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if a["mgt_no"] in drop or b["mgt_no"] in drop or min(len(a["_key"]), len(b["_key"])) < 2:
                continue
            same_bld = a["_bld"] == b["_bld"] and (a["_key"] in b["_key"] or b["_key"] in a["_key"])
            near_same = a["_key"] == b["_key"] and haversine(a["lat"], a["lng"], b["lat"], b["lng"]) <= 60
            if not (same_bld or near_same):
                continue
            keep, lose = (a, b) if a["open_date"] <= b["open_date"] else (b, a)
            drop.add(lose["mgt_no"])
            note[keep["mgt_no"]] = f"'{lose['name']}'({lose['addr_road']}) 인허가와 같은 가게로 보고 합쳐 표시"
            if n_posts.get(lose["mgt_no"], 0) > n_posts.get(keep["mgt_no"], 0):
                borrow[keep["mgt_no"]] = lose["mgt_no"]
    print(f"같은 가게로 보이는 인허가 {len(drop)}건 합침: {', '.join(shops.loc[shops['mgt_no'].isin(drop), 'name'])}")
    return shops[~shops["mgt_no"].isin(drop)].drop(columns=["_bld", "_key"]), note, borrow


def haversine(lat1, lng1, lat2, lng2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lng2 - lng1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def main():
    shops = read("yeonhui_restaurants")
    shops = shops[shops["closed"] == 0]
    surv = read("yeonhui_survival").set_index("mgt_no")
    goog = read("yeonhui_google").set_index("mgt_no")
    blog = read("yeonhui_blog_nlp")
    info = read("yeonhui_info").set_index("mgt_no")
    disp = read("yeonhui_dispositions")
    shops, merged_note, borrow = merge_same_place(shops, blog)
    blog = blog.set_index("mgt_no")

    # ⑤ 베이지안 평점
    sd, status = review_sd()
    g = goog[goog["rating"].notna() & (goog["rating_count"] > 0)]
    m = np.average(g["rating"], weights=g["rating_count"])
    goog["bayes"] = (PRIOR_N * m + goog["rating_count"] * goog["rating"]) / (PRIOR_N + goog["rating_count"])
    goog["bayes_low"] = goog["bayes"] - 1.28 * sd / np.sqrt(goog["rating_count"] + PRIOR_N)
    print(f"베이지안 평점: 연희동 평균 {m:.2f}, 별점 표준편차 {sd:.2f}, 평점 있는 가게 {len(g)}곳")

    # 검색용 임베딩 (int8로 줄여 base64)
    z = np.load(DATA / "yeonhui_shop_emb.npz")
    emb = dict(zip(z["mgt_no"], z["emb"].astype(np.float32)))

    is_branch = shops["name"].str.contains(r"\S{2,}점\)?$|\S+점\s*\(", regex=True)
    out = []
    for (_, s), branch in zip(shops.iterrows(), is_branch):
        mg = s["mgt_no"]
        src = borrow.get(mg, mg)  # 합친 가게는 글이 많은 인허가의 구글·블로그 정보를 쓴다
        name = re.sub(r"^\s*(\(주\)|주식회사|㈜)\s*|\s*(\(주\)|주식회사|㈜)\s*$", "", s["name"]).strip() or s["name"]
        r = {"id": mg, "name": name, "license_name": s["name"], "category": nz(s["category"]), "main_food": nz(s.get("main_food")),
             "lat": round(float(s["lat"]), 6), "lng": round(float(s["lng"]), 6),
             "addr": nz(s["addr_road"]) or nz(s["addr_jibun"]), "open_date": s["open_date"][:10],
             "note": merged_note.get(mg)}

        if mg in surv.index:
            v = surv.loc[mg]
            r["survival"] = {"score": int(v["survival_score"]), "risk_3y": float(v["risk_3y"]),
                             "age": int(v["age_years"]),
                             "up": factor_text(v["risk_up"], branch), "down": factor_text(v["risk_down"], branch)}
        else:
            r["survival"] = {"score": None, "age": int(2026 - int(s["open_date"][:4])),
                             "note": "1990년 이전 개업: 오래 영업해 모델 대상에서 제외"}

        if src in goog.index and pd.notna(goog.at[src, "place_id"]):
            v = goog.loc[src]
            r["google"] = {"rating": nz(v["rating"]), "count": int(v["rating_count"]) if pd.notna(v["rating_count"]) else 0,
                           "bayes": nz(round(v["bayes"], 2)), "bayes_low": nz(round(v["bayes_low"], 2)),
                           "price": nz(v["price_level"]), "type": nz(v["primary_type"]),
                           "status": status.get(v["place_id"])}

        if src in blog.index:
            v = blog.loc[src]
            r["blog"] = {"n": int(v["n_posts"]), "neg_share": round(float(v["neg_share"]), 3),
                         "controversy_pct": float(v["controversy_pct"]), "low_evidence": bool(v["low_evidence"]),
                         # 짧은 상호('우주', '가을')는 다른 이야기 글이 섞여 키워드가 엉뚱해서 숨긴다
                         "keywords": [] if is_ambiguous(s["name"]) else
                                     [k for k in str(v["keywords"]).split(", ") if k and k != "nan"],
                         "last_post": str(int(v["last_post"]))}

        v = info.loc[src]
        r["info"] = {k: nz(v[c]) for k, c in [
            ("reserve", "reserve_info"), ("reserve_links", "reserve_links"), ("waiting", "waiting"),
            ("busy_when", "busy_when"), ("parking", "parking_info"), ("phone", "phone"), ("hours", "hours"),
            ("website", "website"), ("google_map", "maps_url"), ("naver_map", "naver_map_url"),
            ("groups", "goodForGroups"), ("dogs", "allowsDogs"), ("outdoor", "outdoorSeating"),
            ("kids", "goodForChildren"), ("takeout", "takeout"), ("vegetarian", "servesVegetarianFood")]}
        r["info"]["naver_map"] = "https://map.naver.com/p/search/" + quote(f"연희동 {base_name(name)}")
        r["info"]["hours_min"] = parse_hours(nz(v["hours"]))
        # 폐업 의심: 구글 '영업 종료' 표시, 또는 제목에 상호+폐업 표현이 있는 블로그 글
        if pd.notna(v.get("closure_posts")) and v["closure_posts"] > 0:
            d = str(v["closure_date"])
            r["closure"] = {"source": "블로그", "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}", "link": v["closure_link"],
                            "title": v["closure_title"], "posts": int(v["closure_posts"])}
        if (r.get("google") or {}).get("status") == "CLOSED_PERMANENTLY":
            r["closure"] = {"source": "구글", "title": "구글 지도에 '영업 종료'로 표시됨"}
        r["info"] = {k: x for k, x in r["info"].items() if x not in (None, "", "정보 부족")}

        d = disp[disp["mgt_no"] == mg]
        r["hygiene"] = ({"warning": True, "items": d[["DSPS_DCSNDT", "DSPS_TYPECD_NM", "VILTCN"]].astype(str).values.tolist()}
                        if len(d) else {"warning": False,
                                        "text": f"최근 2년 공개된 행정처분 없음 (기준일 {DISPOSITION_DATE})"})

        if src in emb:
            r["emb"] = base64.b64encode(np.clip(np.round(emb[src] * 127 / np.abs(emb[src]).max()), -127, 127)
                                        .astype(np.int8).tobytes()).decode()
        out.append(r)

    meta = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d"), "count": len(out),
            "embedding_model": "Xenova/multilingual-e5-small", "prior_mean_rating": round(float(m), 2),
            "disposition_date": DISPOSITION_DATE}
    path = WEB / "restaurants.json"
    path.write_text(json.dumps({"meta": meta, "restaurants": out}, ensure_ascii=False, separators=(",", ":")), "utf-8")
    has = lambda k: sum(k in r for r in out)  # noqa: E731
    print(f"{len(out)}곳 → {path} ({path.stat().st_size / 1024:.0f}KB). "
          f"생존점수 {sum(r['survival']['score'] is not None for r in out)}, 구글 {has('google')}, "
          f"블로그 {has('blog')}, 검색 임베딩 {has('emb')}")


if __name__ == "__main__":
    main()
