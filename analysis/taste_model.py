"""맛 점수 보완: 구글 평점이 없는 가게의 평점을 블로그 글로 추정한다.

구글은 무료 한도 때문에 일부 가게만 받는다. 구글 평점이 있는 가게(리뷰 MIN_REVIEWS개 이상)로
"블로그 글 → 구글 베이지안 평점" 회귀를 학습하고, 평점이 없는 가게에 적용한다.

특성: 가게 블로그 임베딩(e5 평균, 384차원) + 감성 평균·부정 비율 + 관련 글 수(log) + 광고 비율 + 업태·지역
모델: Ridge 회귀 (5겹 교차검증으로 규제 강도 선택). 성능은 교차검증 순위상관(Spearman)으로 보고한다.

입력:  data/area_google.csv, data/area_blog_nlp.csv, data/area_shop_emb.npz, data/area_restaurants.csv
출력:  data/area_taste.csv (mgt_no, taste_source, taste_rating), analysis/output/taste_model_summary.txt
실행:  py taste_model.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prep import DATA, OUT

PRIOR_N = 20       # export_json.py 와 같은 베이지안 평점 강도
MIN_REVIEWS = 10   # 학습 정답으로 쓸 최소 구글 리뷰 수 (적으면 평점이 운에 흔들림)

lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(str(s))


def bayes(goog):
    g = goog[goog["rating"].notna() & (goog["rating_count"] > 0)]
    m = np.average(g["rating"], weights=g["rating_count"])
    return (PRIOR_N * m + goog["rating_count"] * goog["rating"]) / (PRIOR_N + goog["rating_count"]), m


def main():
    shops = pd.read_csv(DATA / "area_restaurants.csv", dtype={"mgt_no": str})
    shops = shops[shops["closed"] == 0].set_index("mgt_no")
    goog = pd.read_csv(DATA / "area_google.csv", dtype={"mgt_no": str}).set_index("mgt_no")
    goog["bayes"], m = bayes(goog)
    blog = pd.read_csv(DATA / "area_blog_nlp.csv", dtype={"mgt_no": str}).set_index("mgt_no")
    z = np.load(DATA / "area_shop_emb.npz")
    emb = pd.DataFrame(z["emb"].astype(np.float32), index=z["mgt_no"]).add_prefix("e")

    x = blog[["sent_mean", "neg_share", "ad_share"]].copy()
    x["log_posts"] = np.log1p(blog["n_posts_all"])
    x = x.join(emb, how="inner")
    cat = pd.get_dummies(shops["category"].reindex(x.index).fillna("기타"), prefix="cat", dtype=float)
    area = pd.get_dummies(shops["area_name"].reindex(x.index), prefix="area", dtype=float)
    x = x.join(cat).join(area).fillna(0)

    has = goog["rating"].notna() & (goog["rating_count"] >= MIN_REVIEWS)
    train_ids = x.index.intersection(goog.index[has])
    log(f"블로그 분석된 가게 {len(x)}곳, 학습용(구글 리뷰 {MIN_REVIEWS}개 이상) {len(train_ids)}곳, 지역 평균 평점 {m:.2f}")
    y = goog.loc[train_ids, "bayes"].values
    # 후보 모델을 같은 교차검증으로 비교해 순위상관이 가장 높은 것을 쓴다 (학습 가게가 적으면 임베딩 384차원은 과적합)
    small = ["sent_mean", "neg_share", "ad_share", "log_posts"] + [c for c in x.columns if c.startswith(("cat_", "area_"))]
    candidates = {"감성 평균만": ["sent_mean"], "감성·글 수·업태·지역": small, "임베딩 포함 전체": list(x.columns)}
    cv = KFold(5, shuffle=True, random_state=42)
    log(f"[교차검증] 예측 vs 실제 구글 베이지안 평점 ({len(train_ids)}곳, 5겹)")
    best = None
    for label, cols in candidates.items():
        m_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 20)))
        pred_cv = cross_val_predict(m_, x.loc[train_ids, cols], y, cv=cv)
        r_, p_ = spearmanr(pred_cv, y)
        log(f"  {label}: 순위상관 {r_:.2f} (p={p_:.1e}), 평균 절대오차 {np.abs(pred_cv - y).mean():.2f}점")
        if best is None or r_ > best[1]:
            best = (label, r_, cols, m_, p_)
    label, rho, cols, model, p_best = best
    log(f"  비교: 모두 평균으로 예측하면 오차 {np.abs(y - y.mean()).mean():.2f}점 → '{label}' 모델 사용")
    model.fit(x.loc[train_ids, cols], y)

    out = pd.DataFrame(index=shops.index)
    out["taste_source"] = None
    out["taste_rating"] = np.nan
    g_ok = goog.index[goog["rating"].notna() & (goog["rating_count"] > 0)].intersection(out.index)
    out.loc[g_ok, "taste_source"] = "google"
    out.loc[g_ok, "taste_rating"] = goog.loc[g_ok, "bayes"]
    # 블로그 글이 충분한 가게만 추정한다 (근거 부족 가게는 비워 둠)
    enough = blog.index[~blog["low_evidence"].astype(bool)]
    est = x.index.intersection(enough).difference(g_ok)
    if not (rho > 0 and p_best < 0.05):
        log(f"  검증에서 의미 있는 예측력이 없어(p={p_best:.2f}) 블로그 추정 맛 점수는 쓰지 않음")
        est = est[:0]
    if len(est):
        out.loc[est, "taste_source"] = "blog_model"
        out.loc[est, "taste_rating"] = model.predict(x.loc[est, cols])
    out["taste_cv_rho"] = round(rho, 3)
    log(f"맛 점수: 구글 {len(g_ok)}곳, 블로그 추정 {len(est)}곳, 정보 없음 {out['taste_source'].isna().sum()}곳")
    out.reset_index().round(3).to_csv(DATA / "area_taste.csv", index=False, encoding="utf-8-sig")
    log("→ data/area_taste.csv")
    (OUT / "taste_model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
