"""① 폐업 예측: 서울 일반음식점 이산시간 생존모델 (LightGBM) + Cox 기준모델 + SHAP.

가게를 '개업 후 k년차' 구간으로 쪼갠 가게-연도 데이터를 만들고,
"그 해 초에 살아 있던 가게가 1년 안에 폐업하는가"를 학습한다.
이미 오래 운영한 가게의 위험을 따로 계산할 수 있고, 연도별 경기(코로나 등)도 반영된다.

검증: 2021년까지 끝난 구간으로 학습 → 2022~2024년에 시작한 구간으로 평가 (시간 분할)
입력:  data/seoul_restaurants.csv, data/area_restaurants.csv
출력:  analysis/output/model_summary.txt, model_*.png
       data/seoul_survival.csv     서울 영업 중 가게 전체의 3년 폐업 위험과 주요 요인(SHAP)
실행:  python survival_model.py
"""
import matplotlib
matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import CoxPHFitter
from sklearn.metrics import roc_auc_score

from prep import DATA, OUT, ROOT, load_clean

CKPT = ROOT / "analysis" / "checkpoints"   # 중간 결과 저장 (중단 후 이어서 실행)
CKPT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CUTOFF = pd.Timestamp("2025-12-31")   # 이 날까지 관찰이 끝난 구간만 학습·평가에 사용
TODAY = pd.Timestamp("2026-10-14")    # 데이터 기준일 (지도 표시용 예측 시점)
LAST_OBS_YEAR = CUTOFF.year - 1       # 1년 구간이 CUTOFF 안에 끝나는 마지막 시작 연도
TRAIN_END = pd.Timestamp("2022-01-01")
TEST_START, TEST_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2025-01-01")
MIN_OPEN_YEAR = 1990                  # 폐업 기록이 1990년경부터라 그 전 개업은 좌측 절단
CELL_DEG = 0.003                      # 동네 격자 크기 (약 270m × 330m)
TOP_CATS = ["한식", "분식", "경양식", "기타", "호프/통닭", "일식", "중국식", "정종/대포집/소주방"]
FEATURES = ["category", "gu", "log_area", "lat", "lng", "open_month", "name_len",
            "is_branch", "is_corp", "has_latin",
            "age", "cal_year", "cell_active", "cell_same_cat", "cell_closed_prev"]
CAT_FEATURES = ["category", "gu"]
LABELS = {
    "category": "업태", "gu": "자치구", "log_area": "면적", "lat": "위치(위도)", "lng": "위치(경도)",
    "open_month": "개업 월", "name_len": "상호 길이", "is_branch": "지점(○○점)",
    "is_corp": "법인", "has_latin": "영문 상호", "age": "영업 연수", "cal_year": "연도",
    "cell_active": "주변 식당 수", "cell_same_cat": "주변 같은 업태 수",
    "cell_closed_prev": "작년 주변 폐업 수",
}

lines = []


def log(s=""):
    print(s, flush=True)  # 파일로 출력을 돌려도 진행 상황이 바로 보이게
    lines.append(str(s))


# ---------------------------------------------------------------- 특성 만들기
def static_features(d):
    d["category"] = d["category"].where(d["category"].isin(TOP_CATS), "그외")
    d["log_area"] = np.log1p(d["area"].clip(upper=2000))
    d["open_month"] = d["open_date"].dt.month
    name = d["name"].fillna("")
    d["name_len"] = name.str.replace(r"\(.*?\)|\s", "", regex=True).str.len()
    d["is_branch"] = name.str.contains(r"\S{2,}점\)?$|\S+점\s*\(", regex=True).astype(int)
    d["is_corp"] = name.str.contains(r"\(주\)|주식회사|㈜", regex=True).astype(int)
    d["has_latin"] = name.str.contains(r"[A-Za-z]", regex=True).astype(int)
    d["cell"] = ((d["lat"] // CELL_DEG).astype("Int64").astype(str) + "_"
                 + (d["lng"] // CELL_DEG).astype("Int64").astype(str))
    d.loc[d["lat"].isna(), "cell"] = pd.NA
    return d


def cell_year_table(d, years):
    """격자×연도별: 연초 영업 중 식당 수, 같은 업태 수, 전년도 폐업 수 (그 시점 정보만 사용)."""
    g = d.dropna(subset=["cell"])
    oy = g["open_year"].clip(lower=years[0]).astype(int)
    cy = g["close_year"].fillna(9999).astype(int)
    rows = []
    for y in years:
        alive = (oy < y) & (cy >= y)        # y년 1월 1일 기준 영업 중
        closed_prev = cy == y - 1
        a = g[alive].groupby("cell").size().rename("cell_active")
        ac = g[alive].groupby(["cell", "category"]).size().rename("cell_same_cat")
        cp = g[closed_prev].groupby("cell").size().rename("cell_closed_prev")
        t = pd.concat([a, cp], axis=1).fillna(0).reset_index().rename(columns={"index": "cell"})
        t["cal_year"] = y
        ac = ac.reset_index()
        ac["cal_year"] = y
        rows.append((t, ac))
    return (pd.concat([r[0] for r in rows], ignore_index=True),
            pd.concat([r[1] for r in rows], ignore_index=True))


def person_years(d, end):
    """가게별 개업 후 k년차 구간 [시작, 끝)을 펼친다. end 이후까지 이어지는 미완 구간은 버린다."""
    max_k = int((end - d["open_date"].min()).days / 365.25) + 1
    out = []
    for k in range(max_k):
        start = d["open_date"] + pd.DateOffset(years=k)
        stop = d["open_date"] + pd.DateOffset(years=k + 1)
        close = d["close_date"].where(d["closed"] == 1)
        alive = start <= end
        alive &= close.isna() | (close >= start)
        if not alive.any():
            break
        event = close.notna() & (close < stop)
        # 끝까지 관찰된 구간만 쓴다. "폐업했으면 포함"으로 하면 마지막 해에 폐업한 가게만 남아 폐업률이 100%가 된다
        keep = alive & (stop <= end)
        s = d.loc[keep, ["mgt_no"]].copy()
        s["age"] = k
        s["start"] = start[keep]
        s["stop"] = stop[keep]
        s["event"] = event[keep].astype(int)
        out.append(s)
    py = pd.concat(out, ignore_index=True)
    py["cal_year"] = py["start"].dt.year
    return py


def attach(py, shops, cell_tab, cat_tab):
    x = py.merge(shops, on="mgt_no", how="left")
    x = x.merge(cell_tab, on=["cell", "cal_year"], how="left")
    x = x.merge(cat_tab, on=["cell", "category", "cal_year"], how="left")
    for c in ["cell_active", "cell_same_cat", "cell_closed_prev"]:
        x.loc[x["cell"].notna(), c] = x.loc[x["cell"].notna(), c].fillna(0)
    for c in CAT_FEATURES:
        x[c] = x[c].astype("category")
    return x


# ---------------------------------------------------------------- 모델
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=42)


def fit_lgb(x, rounds=None, valid=None):
    dtrain = lgb.Dataset(x[FEATURES], x["event"], categorical_feature=CAT_FEATURES)
    if valid is not None:
        dvalid = lgb.Dataset(valid[FEATURES], valid["event"], reference=dtrain)
        return lgb.train(LGB_PARAMS, dtrain, 2000, valid_sets=[dvalid],
                         callbacks=[lgb.early_stopping(50, verbose=False)])
    return lgb.train(LGB_PARAMS, dtrain, rounds)


def checkpoint_lgb(name, train_fn):
    """학습한 모델을 analysis/checkpoints/ 에 저장하고, 있으면 다시 학습하지 않고 불러온다.
    처음부터 다시 하려면 checkpoints 폴더를 지운다."""
    path = CKPT / f"{name}.txt"
    if path.exists():
        log(f"저장된 모델 사용: {path.name}")
        return lgb.Booster(model_str=path.read_text("utf-8"))
    log(f"{name} 학습 중...")
    model = train_fn()
    # LightGBM의 save_model은 Windows 한글 경로에 못 쓰므로 문자열로 받아 직접 저장
    best = model.best_iteration if model.best_iteration > 0 else None
    path.write_text(model.model_to_string(num_iteration=best), "utf-8")
    return model


def cox_design(x, columns=None):
    """Cox용 수치 행렬: 업태 더미 + 면적 + 개업연도 + 개업 당시 주변 식당 수."""
    m = pd.get_dummies(x["category"].astype(str), prefix="cat", drop_first=False, dtype=float)
    m = m.drop(columns=["cat_한식"], errors="ignore")
    m["log_area"] = x["log_area"].fillna(x["log_area"].median())
    m["open_year"] = x["open_year"]
    m["log_cell_active"] = np.log1p(x["cell_active"].fillna(0))
    if columns is not None:
        m = m.reindex(columns=columns, fill_value=0.0)
    return m


def calibration_plot(y, p, fname):
    bins = pd.qcut(p, 10, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby("bin", observed=True).mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(t["p"], t["y"], "o-", label="LightGBM")
    lim = max(t["p"].max(), t["y"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="완벽한 보정")
    ax.set(title="보정 곡선 (2022~2024 검증)", xlabel="예측 1년 폐업 확률", ylabel="실제 폐업 비율")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)


def main():
    d = load_clean(log)
    d = d[d["open_year"] >= MIN_OPEN_YEAR].reset_index(drop=True)
    log(f"{MIN_OPEN_YEAR}년 이후 개업만 사용: {len(d):,}곳")
    d = static_features(d)

    years = list(range(MIN_OPEN_YEAR, TODAY.year + 1))
    cell_tab, cat_tab = cell_year_table(d, years)
    shops = d[["mgt_no", "category", "gu", "log_area", "lat", "lng", "open_month", "name_len",
               "is_branch", "is_corp", "has_latin", "cell", "open_year"]]

    py_path = CKPT / "person_years.pkl"
    if py_path.exists():
        py = pd.read_pickle(py_path)
        log("저장된 가게-연도 데이터 사용")
    else:
        py = attach(person_years(d, CUTOFF), shops, cell_tab, cat_tab)
        py.to_pickle(py_path)
    log(f"가게-연도 구간 {len(py):,}개, 1년 폐업률 {py['event'].mean():.1%}")

    train = py[py["stop"] <= TRAIN_END]
    test = py[(py["start"] >= TEST_START) & (py["start"] < TEST_END)]
    log(f"학습 구간 {len(train):,}개 (~2021) / 검증 구간 {len(test):,}개 (2022~2024 시작)")

    # LightGBM: 학습 기간 마지막 2년(2020~2021 종료 구간)으로 조기 종료 라운드 결정
    inner_valid = train["stop"] > pd.Timestamp("2020-01-01")
    tuned = checkpoint_lgb("lgb_tuned", lambda: fit_lgb(train[~inner_valid], valid=train[inner_valid]))
    # 파일에서 불러온 모델은 best_iteration이 -1이라 저장된 트리 수를 쓴다
    rounds = tuned.best_iteration if tuned.best_iteration > 0 else tuned.current_iteration()
    log(f"조기 종료 라운드 {rounds}")
    booster = checkpoint_lgb("lgb_train", lambda: fit_lgb(train, rounds))
    p_lgb = booster.predict(test[FEATURES])
    auc_lgb = roc_auc_score(test["event"], p_lgb)

    # Cox 기준모델: 개업 시점 정보만으로 생존곡선 → 구간별 1년 폐업확률
    first = train[train["age"] == 0]
    shop_dur = d.set_index("mgt_no")
    cox_df = first[["mgt_no"]].copy()
    stop_date = shop_dur.loc[cox_df["mgt_no"], "close_date"].where(
        shop_dur.loc[cox_df["mgt_no"], "closed"] == 1).values
    stop_date = pd.to_datetime(stop_date)
    end = pd.Series(stop_date).where(pd.Series(stop_date) < TRAIN_END, TRAIN_END)
    open_d = shop_dur.loc[cox_df["mgt_no"], "open_date"].values
    cox_df["T"] = ((end.values - open_d) / np.timedelta64(1, "D") / 365.25).clip(1 / 365.25)
    cox_df["E"] = (pd.Series(stop_date).notna() & (pd.Series(stop_date) < TRAIN_END)).astype(int).values
    X = cox_design(first)
    cols = X.columns
    log("Cox 기준모델 학습 중... (3만 곳 표본, lifelines는 단일 코어라 15만 곳은 15분 넘게 걸렸음)")
    sample = np.random.default_rng(42).choice(len(X), size=min(30_000, len(X)), replace=False)
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(pd.concat([X.iloc[sample].reset_index(drop=True),
                       cox_df[["T", "E"]].iloc[sample].reset_index(drop=True)], axis=1), "T", "E")
    H0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
    Xt = cox_design(test, cols)
    xb = np.exp(cph.predict_log_partial_hazard(Xt).values)
    h = lambda t: np.interp(t, H0.index.values, H0.values)
    age = test["age"].values
    p_cox = 1 - np.exp(-(h(age + 1) - h(age)) * xb)
    auc_cox = roc_auc_score(test["event"], p_cox)

    # 비교 기준: 업태×영업연수 평균 폐업률 (단순 규칙)
    base = (train.assign(age20=np.minimum(train["age"], 20))
            .groupby(["category", "age20"], observed=True)["event"].mean().rename("p_base").reset_index())
    p_base = (test.assign(age20=np.minimum(test["age"], 20))[["category", "age20"]]
              .merge(base, on=["category", "age20"], how="left")["p_base"]
              .fillna(train["event"].mean()).values)  # 행마다 하면 294만 행 평균을 38만 번 계산해 1시간 걸림
    auc_base = roc_auc_score(test["event"], p_base)

    log("\n[검증 AUC: 2022~2024년 1년 내 폐업 예측]")
    log(f"  단순 규칙(업태×영업연수 평균)  {auc_base:.3f}")
    log(f"  Cox 비례위험 (개업 시점 정보)  {auc_cox:.3f}  (C-index 학습 {cph.concordance_index_:.3f})")
    log(f"  LightGBM 이산시간 생존모델     {auc_lgb:.3f}  (라운드 {rounds})")
    log(f"  검증 실제 폐업률 {test['event'].mean():.1%}, LightGBM 평균 예측 {p_lgb.mean():.1%}")
    calibration_plot(test["event"].values, p_lgb, "model_calibration.png")

    log("\n[Cox 위험비 (한식 대비, 1보다 크면 빨리 폐업)]")
    log(cph.summary[["exp(coef)", "p"]].round(3).to_string())

    # 최종 모델: 2025년까지 전체로 재학습
    final = checkpoint_lgb("lgb_final", lambda: fit_lgb(py, int(rounds * 1.2)))  # 데이터가 늘어난 만큼 라운드를 조금 늘린다
    imp = pd.Series(final.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    log("\n[LightGBM 특성 중요도 (gain 비율)]")
    log((imp / imp.sum()).map("{:.1%}".format).rename(index=LABELS).to_string())

    log("\n서울 영업 중 가게 3년 폐업 위험 계산 중...")
    # 서울 영업 중 가게 전체의 향후 3년 폐업 위험과 서울 안 상대 순위
    active = d[d["closed"] == 0].copy()
    age_now = ((TODAY - active["open_date"]).dt.days // 365.25).astype(int)
    surv = np.ones(len(active))
    first_year_x = None
    for j in range(3):
        cur = pd.DataFrame({"mgt_no": active["mgt_no"].values, "age": (age_now + j).values,
                            "cal_year": LAST_OBS_YEAR})  # 학습에 있는 마지막 연도 효과를 그대로 쓴다
        # 미래 연도의 주변 정보는 알 수 없으므로 현재 값을 그대로 쓴다
        cx = attach(cur.assign(cal_year=TODAY.year), shops, cell_tab, cat_tab)
        assert len(cx) == len(cur), "merge로 행 수가 바뀜 (키 중복)"
        cx["cal_year"] = cur["cal_year"].values
        if j == 0:
            first_year_x = cx
        surv *= 1 - final.predict(cx[FEATURES])
    active["risk_3y"] = 1 - surv
    active["risk_pct"] = active["risk_3y"].rank(pct=True)
    log("서울 영업 중 3년 위험 분위수(10/25/50/75/90%): "
        + " / ".join(f"{q:.0%}" for q in active["risk_3y"].quantile([.1, .25, .5, .75, .9])))
    age_band = pd.cut(age_now, [-1, 2, 5, 10, 20, 100], labels=["0-2년", "3-5년", "6-10년", "11-20년", "21년+"])
    log("영업 연수별 3년 위험 중앙값: " + ", ".join(
        f"{k} {v:.0%}" for k, v in active.groupby(age_band.values, observed=True)["risk_3y"].median().items()))

    area = pd.read_csv(DATA / "area_restaurants.csv", dtype={"mgt_no": str})
    area = area[area["closed"] == 0]
    is_area = active["mgt_no"].isin(area["mgt_no"]).values
    log(f"\nAI 분석 지역 영업 중 {len(area)}곳 중 모델 대상 {is_area.sum()}곳 "
        f"(1990년 이전 개업 등 제외 {len(area) - is_area.sum()}곳)")

    # SHAP: 올해 폐업 위험을 높이거나 낮추는 요인 상위 3개. 서울 영업 중 가게 전부 (10만 곳 넘어서 체크포인트에 저장)
    shap_path = CKPT / "shap_active.npy"
    if shap_path.exists() and np.load(shap_path, mmap_mode="r").shape[0] == len(first_year_x):
        sv = np.load(shap_path)
        log("저장된 SHAP 값 사용")
    else:
        log(f"SHAP 요인 계산 중... ({len(first_year_x):,}곳, 5만 곳씩)")
        explainer = shap.TreeExplainer(final)
        parts = []
        for i in range(0, len(first_year_x), 50_000):
            v = explainer.shap_values(first_year_x[FEATURES].iloc[i:i + 50_000])
            parts.append(v[1] if isinstance(v, list) else v)
            log(f"  {min(i + 50_000, len(first_year_x)):,}/{len(first_year_x):,}")
        sv = np.concatenate(parts).astype(np.float32)
        np.save(shap_path, sv)
    # 연도는 예측할 때 모두 같은 값이라 가게별 요인으로 보여주지 않는다
    shown_idx = [i for i, f in enumerate(FEATURES) if f != "cal_year"]
    up, down = [], []
    for row in sv:
        order = [i for i in np.argsort(row) if i in shown_idx]
        up.append(", ".join(LABELS[FEATURES[i]] for i in order[::-1][:3] if row[i] > 0))
        down.append(", ".join(LABELS[FEATURES[i]] for i in order[:3] if row[i] < 0))
    shown = first_year_x.loc[is_area, FEATURES].copy()
    for c in CAT_FEATURES:
        shown[c] = shown[c].cat.codes  # 색 표시용 숫자 변환
    shap.summary_plot(sv[is_area], shown.rename(columns=LABELS), show=False, max_display=12)
    plt.title("AI 분석 지역 가게의 올해 폐업 위험 요인 (SHAP)")
    plt.tight_layout()
    plt.savefig(OUT / "model_shap_area.png", dpi=120)
    plt.close()

    res = active[["mgt_no", "name", "category", "gu", "open_date"]].copy()
    res["age_years"] = age_now.values
    res["risk_3y"] = active["risk_3y"].round(3).values
    res["risk_pct_seoul"] = active["risk_pct"].round(3).values  # 서울 영업 중 가게 중 위험 순위
    res["survival_score"] = (100 * (1 - res["risk_pct_seoul"])).round().astype(int)
    res["risk_up"] = up
    res["risk_down"] = down
    res.to_csv(DATA / "seoul_survival.csv", index=False, encoding="utf-8-sig")
    log(f"서울 영업 중 가게 3년 폐업 위험 중앙값 {active['risk_3y'].median():.0%}, "
        f"AI 분석 지역 중앙값 {res.loc[is_area, 'risk_3y'].median():.0%}")
    gu_med = res.groupby("gu")["risk_3y"].median().sort_values()
    log("자치구별 3년 위험 중앙값: 낮은 곳 " + ", ".join(f"{g} {v:.0%}" for g, v in gu_med.head(3).items())
        + " / 높은 곳 " + ", ".join(f"{g} {v:.0%}" for g, v in gu_med.tail(3).items()))
    log(f"→ data/seoul_survival.csv ({len(res):,}곳)")

    (OUT / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
