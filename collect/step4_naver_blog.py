"""4단계: 네이버 블로그 검색 API로 식당별 블로그 글(제목+요약문) 수집.

Google 리뷰 5개 제한을 대체하는 데이터. 이후 단계에서
  - 광고글 필터 → 감성분석 → 감성 점수 분산 = '호불호 지수'
  - 키워드 추출 / 임베딩 검색
에 사용한다. 본문 크롤링은 하지 않고 API가 주는 요약문만 쓴다.

사전 준비:
  네이버 클라우드 콘솔 > NAVER API HUB > Application 등록 (사용 API: 검색)
  (2026년부터 개발자센터 검색 API는 API HUB로 이관. 개발자센터 키는 사용 불가)
  .streamlit/secrets.toml 에 naver_client_id = "...", naver_client_secret = "..." 추가
  (API HUB 요금·호출 한도는 콘솔에서 확인. 개발자센터 시절의 하루 25,000회 무료와 다를 수 있음)

실행:  python step4_naver_blog.py [--per-shop 300] [--limit 20]

출력:
  data/cache/naver_blog.json  mgt_no별 원본 (재실행 시 이어서 수집)
  data/area_blogs.csv         글 단위 정리본
"""
import argparse
import html
import re
import time

import pandas as pd

from common import DATA, JsonlCache, get_json, secret

URL = "https://naverapihub.apigw.ntruss.com/search/v1/blog"
DISPLAY = 100  # 1회 최대

# 1차 광고 의심 문구 (이후 분류 모델 학습 시 약한 라벨로 활용)
AD_PATTERN = re.compile(r"원고료|소정의|협찬|제공받아|제공 받아|체험단|무료로 제공|지원받아|업체로부터")


def clean(s):
    return html.unescape(re.sub(r"<.*?>", "", s or "")).strip()


def compact(s):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", re.sub(r"\(.*?\)", "", str(s))).lower()


def base_name(name):
    """검색어용 상호: 괄호 내용과 지점명('연희점' 등)을 뺀다. 블로그는 보통 지점명 없이 쓴다."""
    s = re.sub(r"주식회사|㈜|\(주\)", " ", str(name))   # '주식회사 목란' → '목란'
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\s*\S*점$", "", s.strip()) if re.search(r"\s\S+점$", s.strip()) else s
    return re.sub(r"\s+", " ", s).strip() or str(name)


def name_keys(name):
    """글에 이 중 하나가 있으면 상호 언급으로 본다: 기본 상호, 괄호 안 이름(영문명 등)."""
    keys = {compact(base_name(name))}
    for m in re.findall(r"\((.*?)\)", str(name)):
        # 괄호 안이 영문명이거나 띄어쓰기 없는 별칭일 때만. 지점명('연희점')·설명('우동 돈까스')은 제외
        if re.search(r"[A-Za-z]", m) or (" " not in m.strip() and not m.strip().endswith("점")):
            keys.add(compact(m))
    return {k for k in keys if len(k) >= 2}


def is_ambiguous(name):
    """상호가 두 글자 이하라 일반 단어와 헷갈리는 경우 ('가을', '우주'). 분석 단계에서 더 엄격히 거른다."""
    return max((len(k) for k in name_keys(name)), default=0) <= 2


def mentions(keys, text):
    t = compact(text)
    return any(k in t for k in keys)


def fetch_shop(headers, name, place, per_shop):
    items = []
    keys = name_keys(name)
    for start in range(1, min(per_shop, 1000) + 1, DISPLAY):
        params = {"query": f"{place} {base_name(name)}", "display": DISPLAY, "start": start, "sort": "sim"}
        body = get_json(URL, params=params, headers=headers)
        page = body.get("items", [])
        items.extend(page)
        time.sleep(0.12)  # 초당 10회 제한 여유
        if len(page) < DISPLAY or start + DISPLAY > int(body.get("total", 0)):
            break
        # 관련 글이 거의 없는 페이지면 뒤 페이지는 더 관련이 없으므로 그만 받는다
        hit = sum(mentions(keys, clean(it.get("title")) + clean(it.get("description"))) for it in page)
        if hit < 0.2 * len(page):
            break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-shop", type=int, default=300, help="식당당 최대 수집 글 수 (최대 1000)")
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

    cache = JsonlCache("naver_blog")
    todo = shops[~shops["mgt_no"].isin(cache.data)]
    print(f"대상 {len(shops)}곳, 신규 요청 {len(todo)}곳")
    fails = 0
    for i, (_, shop) in enumerate(todo.iterrows(), 1):
        try:
            cache[shop["mgt_no"]] = fetch_shop(headers, shop["name"], shop["area_name"], args.per_shop)
            fails = 0
        except RuntimeError as e:
            print(f"  ! {shop['name']}: {e}")
            print("  중단합니다(받은 결과는 저장됨). 내일 다시 실행하면 이어서 받아요.")
            break
        except Exception as e:
            print(f"  ! {shop['name']}: {e}")
            fails += 1
            if fails >= 5:
                print("  연속 5번 실패해서 중단합니다(받은 결과는 저장됨).")
                break
            continue
        if i % 10 == 0:
            cache.save()
            print(f"  {i}/{len(todo)} ({time.strftime('%H:%M')})", flush=True)
    cache.save()

    rows = []
    for _, shop in shops.iterrows():
        keys = name_keys(shop["name"])
        for it in cache.data.get(shop["mgt_no"], []):
            title, desc = clean(it.get("title")), clean(it.get("description"))
            rows.append({
                "mgt_no": shop["mgt_no"],
                "name": shop["name"],
                "area_name": shop["area_name"],
                "title": title,
                "description": desc,
                "link": it.get("link"),
                "postdate": it.get("postdate"),
                "blogger": it.get("bloggername"),
                # 검색 결과에 다른 가게 글이 섞이므로, 상호가 실제로 언급된 글만 표시
                "mentions_name": mentions(keys, title + " " + desc),
                "name_ambiguous": is_ambiguous(shop["name"]),
                "ad_suspect": bool(AD_PATTERN.search(title + " " + desc)),
            })
    out = pd.DataFrame(rows).drop_duplicates(["mgt_no", "link"])
    out.to_csv(DATA / "area_blogs.csv", index=False, encoding="utf-8-sig")

    rel = out[out["mentions_name"]]
    per_shop = rel.groupby("mgt_no").size()
    print(f"글 {len(out):,}건 중 상호 언급 {len(rel):,}건, 광고 의심 {rel['ad_suspect'].mean():.1%}")
    per_shop = per_shop.reindex(shops["mgt_no"], fill_value=0)  # 관련 글 0건인 식당도 센다
    print(f"식당당 관련 글 수: 중앙값 {per_shop.median():.0f}, 10건 미만 식당 {(per_shop < 10).sum()}곳")
    amb = shops["name"].map(is_ambiguous)
    print(f"짧은 상호(일반 단어와 헷갈림) {amb.sum()}곳: 분석 때 별도 확인 필요")


if __name__ == "__main__":
    main()
