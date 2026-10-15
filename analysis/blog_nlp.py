"""③④ 블로그 분석: 광고 필터 → 감성분석 → 호불호 지수, 키워드, 검색용 가게 임베딩.

1. 관련 글: 상호가 언급된 글만. 짧은 상호('가을', '우주')는 제목에 상호가 있어야 인정
2. 광고 필터: 협찬 문구(정규식) + 같은 가게에 대해 다른 블로거가 거의 같은 문장을 쓴 글(체험단 원고)
   주제 필터: 전체 글 평균과 가장 동떨어진 하위 2% 글(식당 이야기 아님)
3. 감성분석: 구글 리뷰 별점(★4~5 긍정, ★1~2 부정)을 라벨로 e5 임베딩 + 로지스틱 회귀 학습 → 블로그 글에 적용
4. 호불호 지수: 가게별 부정 글 비율. 글이 적으면 전체 평균 쪽으로 당긴다(베타 사전분포)
5. 키워드: Kiwi 명사 후보 → 가게별 TF-IDF 상위 → KeyBERT 방식 MMR로 겹치지 않게 5개
6. 검색용 임베딩: 광고 아닌 글 임베딩의 평균 (브라우저에서 같은 e5 모델로 질문을 임베딩해 비교)

입력:  data/yeonhui_blogs.csv, data/yeonhui_blogs_extra.csv, data/cache/google_places.json
출력:  data/yeonhui_blog_nlp.csv, data/yeonhui_blog_posts.csv, data/yeonhui_shop_emb.npz,
       analysis/output/blog_nlp_summary.txt
실행:  py blog_nlp.py   (임베딩은 analysis/checkpoints/blog_emb.npz 에 저장, 재실행 시 이어서 계산)
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from prep import DATA, OUT, ROOT

sys.path.insert(0, str(ROOT / "collect"))
from step4_naver_blog import compact, mentions, name_keys  # noqa: E402

CKPT = ROOT / "analysis" / "checkpoints"
CKPT.mkdir(exist_ok=True)
EMB_PATH = CKPT / "blog_emb.npz"
MODEL = "intfloat/multilingual-e5-small"   # 브라우저용 Xenova/multilingual-e5-small 과 같은 모델
DUP_SIM = 0.6        # 글자 n-gram 유사도가 이 이상인 글이 다른 블로거에게 2건 이상 있으면 체험단 원고로 본다
PRIOR_K = 10         # 호불호 지수 축소 강도 (글 10건 분량의 전체 평균)
MIN_POSTS = 10       # 이보다 글이 적으면 '근거 부족'
OFF_TOPIC_Q = 0.02   # 전체 평균과 유사도 하위 2% 글은 식당 이야기가 아닌 것으로 본다 (하위 1~3% 글을 읽어 보고 정함)
AREA = r"연희|신촌|연대|연세|홍대|홍제|서대문|연남|명지대|사러가|궁동|홍연"   # 연희동 주변 지명
STOP = set("""연희동 연희 연남동 신촌 홍대 서대문구 서울 맛집 방문 후기 리뷰 포스팅 블로그 오늘 정도 생각 사람 가게 식당
메뉴 음식 추천 위치 시간 영업시간 주소 전화 전화번호 주차 주차장 예약 웨이팅 근처 이번 다음 친구 정말 진짜 하나 가격
사진 인스타그램 인스타 내돈내산 이곳 여기 저희 우리 동네 느낌 분위기 하루 주말 평일 점심 저녁 시작 마무리 정보 이용
라스트오더 휴무 매주 브레이크타임 브레이크 타임 도보 거리 지하철 출구 번출구 연희로 홍연길 연희맛로 서울특별시
입구 마트 정류장 버스 요일 월요일 화요일 수요일 목요일 금요일 토요일 일요일 오픈 마감 공지 사장 사장님 이용시간 소개 진행 유명 장소 정기 나들이 고지 공간 감성 방문기 리뷰어 이야기 오더 라스트 오랜만 매장 안녕 가능 하우스 서대문 마포 연대 연세대 주식회사 법인""".split())
# 지점명('연희점', '본점'), 도로명('성산로'), 지명('석남동', '서구', '인천광역시'). 같은 이름의 다른 지역 가게 글이 섞일 때 나온다
# '면'은 냉면·라면 때문에 빼고, 동·시·구로 끝나는 음식(우동, 텐동, 스시, 아구)은 예외
NOT_KEYWORD = re.compile(r"점$|로$|길$|(?<!우|규|텐|츠|케|타|센|코|비|후)동$|(?<!스|홍)시$|(?<!아)구$|군$|읍$|광역시|특별시")

lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ---------------------------------------------------------------- 데이터
def load_posts():
    main = pd.read_csv(DATA / "yeonhui_blogs.csv", dtype={"mgt_no": str})
    extra_path = DATA / "yeonhui_blogs_extra.csv"
    parts = [main.assign(topic="일반")]
    if extra_path.exists():
        parts.append(pd.read_csv(extra_path, dtype={"mgt_no": str}))
    b = pd.concat(parts, ignore_index=True).drop_duplicates(["mgt_no", "link"])
    b[["title", "description"]] = b[["title", "description"]].fillna("")
    log(f"수집 글 {len(b):,}건 (일반 검색 + 예약·웨이팅·주차 추가 검색, 중복 제거)")

    keys = {n: name_keys(n) for n in b["name"].unique()}
    in_title = np.array([mentions(keys[n], t) for n, t in zip(b["name"], b["title"])])
    ok = b["mentions_name"] & (~b["name_ambiguous"] | in_title)
    log(f"상호 언급 글 {b['mentions_name'].sum():,}건 → 짧은 상호는 제목 언급만 인정 → {ok.sum():,}건")
    b = b[ok].reset_index(drop=True)
    # 연희동 주변 지명이 없는 글은 같은 이름의 다른 가게·체인 다른 지점 글일 가능성이 커서 뺀다 (전체의 약 3%)
    local = (b["title"] + " " + b["description"]).str.contains(AREA, regex=True)
    log(f"연희동 주변 지명이 있는 글만 → {local.sum():,}건 (체인 다른 지점·동명 가게 글 {(~local).sum():,}건 제외)")
    b = b[local].reset_index(drop=True)
    b["text"] = (b["title"] + ". " + b["description"]).str.slice(0, 400)
    return b


def google_reviews():
    """구글 리뷰(한국어) 별점 → 감성 라벨. 검색 후보 가게의 리뷰도 모두 쓴다(라벨 데이터로만 사용)."""
    c = json.loads((DATA / "cache" / "google_places.json").read_text("utf-8"))
    rows = {}
    for places in c.values():
        for p in places:
            for r in p.get("reviews", []):
                t = r.get("originalText") or r.get("text") or {}
                if t.get("languageCode") == "ko" and len(t.get("text", "")) >= 10:
                    rows[r["name"]] = (t["text"][:400], r["rating"])
    g = pd.DataFrame(rows.values(), columns=["text", "rating"])
    g = g[g["rating"] != 3].reset_index(drop=True)
    g["neg"] = (g["rating"] <= 2).astype(int)
    return g


def embed(texts):
    """e5 임베딩. 계산한 문장은 checkpoints에 저장해 두고 새 문장만 계산한다."""
    texts = list(dict.fromkeys(texts))
    have = {}
    if EMB_PATH.exists():
        z = np.load(EMB_PATH, allow_pickle=True)
        have = dict(zip(z["texts"].tolist(), z["emb"]))
    todo = [t for t in texts if t not in have]
    log(f"임베딩: 저장됨 {len(texts) - len(todo):,}건, 새로 계산 {len(todo):,}건")
    if todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL)
        step = 2000
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            e = model.encode(["passage: " + t for t in chunk], batch_size=64, normalize_embeddings=True)
            have.update(zip(chunk, e.astype(np.float32)))
            np.savez(EMB_PATH, texts=np.array(list(have), dtype=object), emb=np.stack(list(have.values())))
            print(f"  {min(i + step, len(todo)):,}/{len(todo):,}", flush=True)
    return have


# ---------------------------------------------------------------- 분석
def campaign_flags(b):
    """같은 가게 글 중 다른 블로거 2명 이상과 문장이 거의 같은 글 → 체험단 원고 의심."""
    flag = np.zeros(len(b), bool)
    for _, idx in b.groupby("mgt_no").indices.items():
        if len(idx) < 3:
            continue
        sub = b.iloc[idx]
        if not sub["description"].str.strip().any():
            continue
        x = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3)).fit_transform(sub["description"])
        sim = (x @ x.T).toarray()
        same_blogger = sub["blogger"].values[:, None] == sub["blogger"].values[None, :]
        sim[same_blogger] = 0
        flag[idx] = (sim >= DUP_SIM).sum(1) >= 2
    return flag


def keywords(b, emb_of, model_encode):
    """가게별 키워드 5개: Kiwi 명사 → 가게 내 등장 글 수 × 가게 간 희소성(TF-IDF) 상위 15개 → MMR."""
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    toks = [{t.form for t in kiwi.tokenize(s) if t.tag in ("NNG", "NNP", "SL") and len(t.form) >= 2}
            for s in b["text"]]
    b = b.assign(toks=toks)
    shop_terms = {}
    for m, g in b.groupby("mgt_no"):
        name_k = name_keys(g["name"].iloc[0])
        cnt = pd.Series([w for ts in g["toks"] for w in ts]).value_counts()
        # 상호 조각('PROT', 'AUDIE.')은 글자만 남겨 상호 키와 비교해서 뺀다
        cnt = cnt[[w not in STOP and not NOT_KEYWORD.search(w) and len(compact(w)) >= 2
                   and not any(compact(w) in k or k in compact(w) for k in name_k) for w in cnt.index]]
        shop_terms[m] = cnt[cnt >= max(2, 0.05 * len(g))] / len(g)  # 글의 5% 이상(최소 2건)에 나온 명사
    df = pd.Series([w for s in shop_terms.values() for w in s.index]).value_counts()
    idf = np.log(len(shop_terms) / df)
    out = {}
    for m, tf in shop_terms.items():
        if tf.empty:
            out[m] = ""
            continue
        cand = (tf * idf[tf.index]).sort_values(ascending=False).head(15).index.tolist()
        ce = model_encode(cand)
        centroid = np.mean([emb_of[t] for t in b.loc[b["mgt_no"] == m, "text"]], axis=0)
        rel = ce @ centroid
        chosen = [int(np.argmax(rel))]
        while len(chosen) < min(5, len(cand)):  # MMR: 가게와 관련 크고 이미 고른 단어와 덜 겹치게
            red = (ce @ ce[chosen].T).max(1)
            score = 0.5 * rel - 0.5 * red
            score[chosen] = -np.inf
            chosen.append(int(np.argmax(score)))
        out[m] = ", ".join(cand[i] for i in chosen)
    return out


def main():
    b = load_posts()
    g = google_reviews()
    log(f"감성 라벨: 구글 리뷰(한국어, ★3 제외) {len(g):,}건, 부정(★1~2) {g['neg'].sum()}건")

    emb_of = embed(b["text"].tolist() + g["text"].tolist())
    Xb = np.stack([emb_of[t] for t in b["text"]])
    Xg = np.stack([emb_of[t] for t in g["text"]])

    # 광고 필터
    b["ad_campaign"] = campaign_flags(b)
    b["is_ad"] = b["ad_suspect"] | b["ad_campaign"]
    log(f"\n[광고 필터] 협찬 문구 {b['ad_suspect'].sum():,}건, 체험단 원고 의심 {b['ad_campaign'].sum():,}건 "
        f"→ 제외 {b['is_ad'].mean():.1%}")
    ex = b[b["ad_campaign"]].groupby("mgt_no").head(2).head(6)
    for _, r in ex.iterrows():
        log(f"  예) [{r['name']}] {r['description'][:70]}")

    # 주제 필터: 전체 글 평균과 가장 동떨어진 하위 2%는 식당 이야기가 아닌 글(전시회, 건물 경매 기사, 다른 지역 지점)이 대부분.
    # 감성·키워드·검색에서는 빼고, 폐업 기사 찾기(shop_info.py)에는 쓰도록 표시만 남긴다
    mu = Xb.mean(0)
    b["topic_sim"] = Xb @ (mu / np.linalg.norm(mu))
    b["off_topic"] = b["topic_sim"] < b["topic_sim"].quantile(OFF_TOPIC_Q)
    log(f"[주제 필터] 식당 이야기가 아닌 글 {b['off_topic'].sum():,}건 제외 (유사도 하위 {OFF_TOPIC_Q:.0%})")
    use = ~b["is_ad"] & ~b["off_topic"]

    # 감성분석: 구글 별점으로 학습, 5겹 교차검증
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    p_cv = cross_val_predict(clf, Xg, g["neg"], cv=cv, method="predict_proba")[:, 1]
    log(f"\n[감성분석] 구글 리뷰 5겹 교차검증 AUC {roc_auc_score(g['neg'], p_cv):.3f} "
        f"(부정 재현율 {(p_cv[g['neg'] == 1] >= 0.5).mean():.0%}, 긍정을 부정으로 {(p_cv[g['neg'] == 0] >= 0.5).mean():.0%})")
    clf.fit(Xg, g["neg"])
    b["p_neg"] = clf.predict_proba(Xb)[:, 1]
    b["neg"] = b["p_neg"] >= 0.5
    for label, r in [("가장 부정적", b.nlargest(3, "p_neg")), ("가장 긍정적", b.nsmallest(3, "p_neg"))]:
        for _, x in r.iterrows():
            log(f"  {label} {x['p_neg']:.2f} [{x['name']}] {x['description'][:70]}")

    # 호불호 지수: 광고 아닌 글의 부정 비율을 전체 평균 쪽으로 축소
    clean = b[use]
    m0 = clean["neg"].mean()
    s = clean.groupby("mgt_no").agg(n_posts=("neg", "size"), n_neg=("neg", "sum"),
                                    sent_mean=("p_neg", lambda p: 1 - p.mean()),
                                    sent_sd=("p_neg", "std"), last_post=("postdate", "max"))
    s["neg_share"] = (s["n_neg"] + PRIOR_K * m0) / (s["n_posts"] + PRIOR_K)
    s["controversy_pct"] = s["neg_share"].rank(pct=True).round(3)   # 연희동 안 상대 순위 (높을수록 호불호)
    s["low_evidence"] = s["n_posts"] < MIN_POSTS
    s = s.join(b.groupby("mgt_no")["is_ad"].sum().rename("n_ads"))
    log(f"\n[호불호] 광고 제외 글 {len(clean):,}건, 전체 부정 비율 {m0:.1%}, "
        f"글 {MIN_POSTS}건 미만 가게 {s['low_evidence'].sum()}곳")

    # 검증: 블로그 감성 평균과 구글 평점의 순위 상관 (리뷰 20개 이상, 글 10건 이상 가게)
    gg = pd.read_csv(DATA / "yeonhui_google.csv", dtype={"mgt_no": str}).set_index("mgt_no")
    v = s.join(gg[["rating", "rating_count"]])
    v = v[(v["rating_count"] >= 20) & ~v["low_evidence"]]
    rho, p = spearmanr(v["sent_mean"], v["rating"])
    log(f"  검증: 블로그 감성 평균 vs 구글 평점 순위상관 {rho:.2f} (p={p:.3f}, {len(v)}곳)")

    # 키워드 + 검색 임베딩
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL)
    enc = lambda ws: st.encode(["query: " + w for w in ws], normalize_embeddings=True)  # noqa: E731
    log("\n키워드 추출 중...")
    s["keywords"] = pd.Series(keywords(clean, emb_of, enc))
    ids = s.index.tolist()
    shop_emb = np.stack([Xb[(b["mgt_no"] == m).values & use.values].mean(0) for m in ids])
    shop_emb /= np.linalg.norm(shop_emb, axis=1, keepdims=True)
    np.savez_compressed(DATA / "yeonhui_shop_emb.npz", mgt_no=np.array(ids), emb=shop_emb.astype(np.float16))
    names = b.drop_duplicates("mgt_no").set_index("mgt_no")["name"]
    for m in s.sort_values("n_posts", ascending=False).index[:8]:
        log(f"  {names[m]}: {s.at[m, 'keywords']}")

    s.insert(0, "name", names[s.index])
    s.round(3).to_csv(DATA / "yeonhui_blog_nlp.csv", encoding="utf-8-sig")
    b.drop(columns=["text"]).to_csv(DATA / "yeonhui_blog_posts.csv", index=False, encoding="utf-8-sig")
    log(f"\n→ data/yeonhui_blog_nlp.csv ({len(s)}곳), yeonhui_blog_posts.csv, yeonhui_shop_emb.npz")
    (OUT / "blog_nlp_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
