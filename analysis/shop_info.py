"""가게 이용 정보: 예약 방법 · 웨이팅 · 붐비는 때 · 주차 (블로그 언급 + 구글 Place Details).

캐치테이블·테이블링·네이버 지도는 공개 API가 없고 구글 '인기 시간대'도 API에 없어서,
블로그 요약문에서 언급을 찾아 "근거 글 수 · 최근 언급일 · 링크"와 함께 보여준다.
예약 방식은 바뀔 수 있으므로 최근 언급일을 꼭 같이 표시한다.

입력:  data/area_blog_posts.csv (blog_nlp.py 결과, 상호 언급 글), data/area_details.csv,
       data/area_restaurants.csv
출력:  data/area_info.csv, analysis/output/shop_info_summary.txt
실행:  py shop_info.py
"""
import re
import sys
from urllib.parse import quote

import pandas as pd

from prep import DATA, OUT, ROOT

sys.path.insert(0, str(ROOT / "collect"))
from step4_naver_blog import base_name  # noqa: E402

# 예약·대기 방법: (표시 이름, 패턴). 요약문은 200자 정도라 언급이 있을 때만 판단한다
RESERVE = [
    ("캐치테이블", r"캐치\s*테이블|catch\s*table"),
    ("테이블링", r"테이블링|tabling"),
    ("네이버 예약", r"네이버\s*(예약|booking)|N\s*예약|네이버로\s*예약"),
    ("전화 예약", r"전화\s*(로\s*)?예약|예약\s*(은|는)?\s*전화"),
    ("예약 불가", r"예약\s*(불가|안\s*(돼|되|받)|이\s*안\s*(돼|되)|(을|를)?\s*받지\s*않|X)|노\s*예약|no\s*reservation"),
    ("예약 필수", r"예약\s*(필수|제(?!\s*(아니|아님|X|x))|하고\s*가(야|세요))|100%\s*예약"),
]
WAIT_YES = r"웨이팅|대기\s*(시간|가\s*길|줄)|줄\s*(을\s*)?서|오픈\s*런|대기\s*(명단|등록)"
WAIT_NO = r"웨이팅\s*(없|X|x|0)|바로\s*(입장|착석|앉)|대기\s*(없|0)|웨이팅\s*없이"
WHEN = {"점심": r"점심|런치", "저녁": r"저녁|디너", "주말": r"주말|토요일|일요일", "평일": r"평일",
        "오픈 직후": r"오픈\s*런|오픈\s*(전|시간|하자마자)"}
PARK_NO = r"주차\s*(불가|가?\s*안\s*(돼|되)|공간\s*(이|은)?\s*(없|협소)|(가|는|이)?\s*(어렵|어려|힘들|힘듦|불편))|주차장\s*(이|은)?\s*없|주차\s*X"
PARK_PUBLIC = r"공영\s*주차장"
PARK_YES = r"주차\s*(가능|공간\s*(있|넉넉)|(가|도)?\s*(돼|되|편)|할\s*수\s*있)|전용\s*주차|발레|발렛|주차장\s*(이|은)?\s*(있|완비|넓)"

CLOSE_TITLE = r"폐업|문\s*(을\s*)?닫|영업\s*(을\s*)?종료"   # '마지막 영업'은 예능 회차 기사 등에 걸려서 뺌
PARTICLE = r"(?:은|는|이|가|을|를|의|에|에서|도|과|와|이다|본점|점)?"


def name_word(name):
    """상호가 '독립된 단어'로 나올 때만 맞는 정규식. '목란' 본점 폐업 → 맞음, 우주떡집 → '우주' 안 맞음."""
    name = name.strip()
    if len(re.sub(r"\s", "", name)) < 2:
        return None
    return rf"(?<![가-힣A-Za-z0-9]){re.escape(name)}{PARTICLE}(?![가-힣A-Za-z0-9])"

lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def evidence(g, pattern):
    """패턴이 나온 글 수, 최근 언급일, 최근 글 링크."""
    hit = g[g["text"].str.contains(pattern, regex=True, flags=re.I)]
    if hit.empty:
        return 0, None, None
    last = hit.loc[hit["postdate"].idxmax()]
    return len(hit), str(last["postdate"]), last["link"]


def label_date(d):
    return f"{d[:4]}-{d[4:6]}" if d else ""


def main():
    allp = pd.read_csv(DATA / "area_blog_posts.csv", dtype={"mgt_no": str}, low_memory=False)
    allp["text"] = allp["title"].fillna("") + " " + allp["description"].fillna("")
    # 사실 정보(예약·주차)는 광고글에도 맞게 쓰여 있으므로 광고 여부와 관계없이 쓴다. 식당 이야기가 아닌 글만 뺀다
    allp["off_topic"] = allp["off_topic"].astype(str).eq("True")
    p = allp[~allp["off_topic"]]
    log(f"상호 언급 글 {len(p):,}건, {p['mgt_no'].nunique()}곳")

    # 폐업 소식: 제목에 상호와 폐업 표현이 함께 있는 글 (본문에만 있으면 다른 가게 이야기인 경우가 많았음)
    closure = {}
    for m, g in allp.groupby("mgt_no"):
        word = name_word(base_name(g["name"].iloc[0]))
        if word is None:
            continue
        titles = g["title"].fillna("")
        hit = g[titles.str.contains(CLOSE_TITLE, regex=True) & titles.str.contains(word, regex=True)]
        if len(hit):
            last = hit.loc[hit["postdate"].idxmax()]
            closure[m] = (len(hit), str(last["postdate"]), last["link"], last["title"])

    rows = []
    for m, g in p.groupby("mgt_no"):
        r = {"mgt_no": m, "n_posts": len(g)}
        if m in closure:
            n, last, link, title = closure.pop(m)
            r.update(closure_posts=n, closure_date=last, closure_link=link, closure_title=title)
        found = []
        for label, pat in RESERVE:
            n, last, link = evidence(g, pat)
            r[f"rsv_{label}"] = n
            if n:
                found.append((n, label, last, link))
        # 서로 다른 방법이 섞여 언급되므로(예: 테이블링 31건, 예약 불가 3건) 가장 많은 방법의 25% 이상만 보여준다.
        # 글이 많은 가게는 한 건만으로 판단하지 않는다
        top_n = max((f[0] for f in found), default=0)
        min_n = max(2 if len(g) >= 20 else 1, 0.25 * top_n)
        shown = sorted((f for f in found if f[0] >= min_n), reverse=True)
        r["reserve_methods"] = " · ".join(f"{lab}({n}건, 최근 {label_date(last)})" for n, lab, last, _ in shown)
        r["reserve_links"] = " ".join(link for *_, link in shown)

        n_yes, _, _ = evidence(g, WAIT_YES)
        n_no, _, _ = evidence(g, WAIT_NO)
        n_yes -= n_no  # "웨이팅 없이"도 '웨이팅'을 포함하므로 뺀다
        r["wait_share"] = max(n_yes, 0) / len(g)
        r["wait_none_posts"] = n_no
        waits = g[g["text"].str.contains(WAIT_YES, regex=True) & ~g["text"].str.contains(WAIT_NO, regex=True)]
        when = {k: int(waits["text"].str.contains(v, regex=True).sum()) for k, v in WHEN.items()}
        top = [k for k, v in sorted(when.items(), key=lambda x: -x[1]) if v >= 2][:2]
        r["busy_when"] = ", ".join(top)

        pn, pl, _ = evidence(g, PARK_NO)
        pp, _, _ = evidence(g, PARK_PUBLIC)
        py_, _, _ = evidence(g, PARK_YES)
        r.update(park_no=pn, park_public=pp, park_yes=py_)
        rows.append(r)
    # 주제 필터로 글이 모두 빠진 가게도 폐업 소식은 남긴다
    rows += [{"mgt_no": m, "n_posts": 0, "closure_posts": n, "closure_date": d, "closure_link": lk, "closure_title": t}
             for m, (n, d, lk, t) in closure.items()]
    info = pd.DataFrame(rows).set_index("mgt_no")
    for c in ["closure_posts", "closure_date", "closure_link", "closure_title"]:
        if c not in info:
            info[c] = None   # 폐업 소식이 하나도 없어도 열은 있어야 export가 동작한다

    # 웨이팅 정도: 분석 지역 안 상대 비교 (글 10건 이상인 가게만 판단)
    enough = info["n_posts"] >= 10
    q = info.loc[enough, "wait_share"].quantile([0.5, 0.85])
    info["waiting"] = "정보 부족"
    info.loc[enough, "waiting"] = pd.cut(info.loc[enough, "wait_share"], [-1, q[0.5], q[0.85], 2],
                                         labels=["드묾", "가끔", "잦음"]).astype(str)

    # 구글 Place Details 합치기
    shops = pd.read_csv(DATA / "area_restaurants.csv", dtype={"mgt_no": str})
    shops = shops[shops["closed"] == 0].set_index("mgt_no")[["name", "area_name"]]
    det = pd.read_csv(DATA / "area_details.csv", dtype={"mgt_no": str}).set_index("mgt_no")
    out = shops.join(info).join(det.drop(columns=["name", "place_id"]))
    out["n_posts"] = out["n_posts"].fillna(0).astype(int)
    out[["park_yes", "park_no", "park_public"]] = out[["park_yes", "park_no", "park_public"]].fillna(0)

    # 주차 판단: 구글 값 > 블로그 다수결. 근거가 없으면 비워 둔다
    def parking(r):
        g = r.get("parking")
        if isinstance(g, str) and g != "없음":
            return "가능 (구글)"
        yes, no, pub = r["park_yes"], r["park_no"], r["park_public"]
        if yes >= 2 and yes > no:
            return f"가능 (블로그 {int(yes)}건)"
        if no >= 2 and no > yes:
            return "어려움" + (" · 근처 공영주차장" if pub >= 1 else "") + f" (블로그 {int(no)}건)"
        if g == "없음":
            return "어려움 (구글)"
        if pub >= 2:
            return f"근처 공영주차장 이용 (블로그 {int(pub)}건)"
        return ""
    out["parking_info"] = out.apply(parking, axis=1)
    # 구글 reservable은 '예약 가능' 정보로만 덧붙인다 (방법은 모름)
    g_rsv = out["reservable"].map({True: "예약 가능(구글)", False: "예약 안 받음(구글)"}).fillna("")
    out["reserve_info"] = [" · ".join(x for x in (a if isinstance(a, str) else "", b) if x)
                           for a, b in zip(out["reserve_methods"], g_rsv)]
    out["naver_map_url"] = ["https://map.naver.com/p/search/" + quote(f"{a} {n}") for n, a in zip(out["name"], out["area_name"])]

    log(f"\n[정보가 있는 가게 비율, 영업 중 {len(out)}곳 기준]")
    log(f"  예약 방법 {(out['reserve_info'] != '').mean():.0%}, 웨이팅 판단 {(out['waiting'].isin(['드묾', '가끔', '잦음'])).mean():.0%}, "
        f"주차 {(out['parking_info'] != '').mean():.0%}, 전화 {out['phone'].notna().mean():.0%}, 영업시간 {out['hours'].notna().mean():.0%}")
    for label, _ in RESERVE:
        log(f"  {label}: {(out[f'rsv_{label}'].fillna(0) >= 1).sum()}곳 언급")
    log("\n[웨이팅 잦음 상위]")
    top = out[out["waiting"] == "잦음"].sort_values("wait_share", ascending=False).head(10)
    for n, r in top.iterrows():
        log(f"  {r['name']}: 웨이팅 언급 {r['wait_share']:.0%}, {r['busy_when']} / 예약 {r['reserve_info']} / 주차 {r['parking_info']}")

    cl = out[out["closure_posts"].fillna(0) > 0]
    log(f"\n[폐업 소식] 제목에 상호+폐업 표현이 있는 글이 있는 가게 {len(cl)}곳")
    for _, r in cl.iterrows():
        log(f"  {r['name']}: {int(r['closure_posts'])}건, 최근 {r['closure_date']} | {r['closure_title'][:60]}")

    out.reset_index().to_csv(DATA / "area_info.csv", index=False, encoding="utf-8-sig")
    log(f"\n→ data/area_info.csv ({len(out)}곳)")
    (OUT / "shop_info_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
