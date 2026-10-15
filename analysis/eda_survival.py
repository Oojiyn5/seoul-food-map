"""① 폐업 예측 EDA: 서울 일반음식점의 생존 기간을 탐색한다.

입력:  data/seoul_restaurants.csv (collect/step1 결과)
출력:  analysis/output/eda_*.png, analysis/output/eda_summary.txt
실행:  python eda_survival.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from prep import OUT, load_clean

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def km_plot(d, group, title, fname, top=None):
    """그룹별 카플란-마이어 생존곡선과 중앙 생존기간."""
    groups = d[group].value_counts()
    if top:
        groups = groups.head(top)
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    log(f"\n[{title}] 그룹별 중앙 생존기간(년)")
    for g in groups.index:
        s = d[d[group] == g]
        kmf.fit(s["years"], s["closed"], label=f"{g} (n={len(s):,})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
        log(f"  {g:<12} n={len(s):>7,}  중앙 {kmf.median_survival_time_:5.1f}년  "
            f"3년 생존 {kmf.predict(3):.0%}  10년 생존 {kmf.predict(10):.0%}")
    sub = d[d[group].isin(groups.index)]
    p = multivariate_logrank_test(sub["years"], sub[group], sub["closed"]).p_value
    log(f"  로그순위 검정 p = {p:.2g}")
    ax.set(title=title, xlabel="영업 기간(년)", ylabel="생존 확률", xlim=(0, 30))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=120)
    plt.close(fig)


def main():
    d = load_clean(log)

    # 폐업 연도 분포: 오래된 폐업 기록이 빠져 있으면(좌측 절단) 생존기간이 부풀려진다
    log("\n폐업 연도별 건수 (기록이 시작되는 시점 확인)")
    cy = d.loc[d["closed"] == 1, "close_year"].value_counts().sort_index()
    log(cy[cy.index >= 1990].to_string())
    log("\n개업 연도별 건수 (10년 단위)")
    log((d["open_year"] // 10 * 10).value_counts().sort_index().to_string())

    # 연간 폐업률 추이
    fig, ax = plt.subplots(figsize=(8, 4))
    cy[(cy.index >= 2000) & (cy.index <= 2025)].plot(ax=ax, marker="o")
    ax.set(title="연도별 폐업 건수 (서울 일반음식점)", xlabel="폐업 연도", ylabel="건수")
    fig.tight_layout()
    fig.savefig(OUT / "eda_close_by_year.png", dpi=120)
    plt.close(fig)

    # 전체 생존곡선
    kmf = KaplanMeierFitter().fit(d["years"], d["closed"])
    log(f"\n서울 전체 중앙 생존기간 {kmf.median_survival_time_:.1f}년, "
        f"1년 {kmf.predict(1):.0%} / 3년 {kmf.predict(3):.0%} / 5년 {kmf.predict(5):.0%} 생존")

    km_plot(d, "category", "업태별 생존곡선 (상위 8개)", "eda_km_category.png", top=8)

    # 개업 시기(코호트): 최근 개업한 가게일수록 빨리 망하는지
    d["cohort"] = pd.cut(d["open_year"], [1900, 1999, 2009, 2014, 2019, 2030],
                         labels=["~1999", "2000-09", "2010-14", "2015-19", "2020~"])
    km_plot(d, "cohort", "개업 시기별 생존곡선", "eda_km_cohort.png")

    # 면적
    d["area_bin"] = pd.cut(d["area"], [-1, 0, 33, 66, 132, 1e9],
                           labels=["0/결측", "~33㎡", "33-66㎡", "66-132㎡", "132㎡~"])
    km_plot(d, "area_bin", "가게 면적별 생존곡선", "eda_km_area.png")

    km_plot(d, "gu", "자치구별 생존곡선 (가게 수 상위 6개)", "eda_km_gu.png", top=6)

    # 연희동 vs 서울
    d["region"] = "서울(연희동 제외)"
    d.loc[(d["gu"] == "서대문구") & (d["dong"] == "연희동"), "region"] = "연희동"
    km_plot(d, "region", "연희동 vs 서울", "eda_km_yeonhui.png")

    (OUT / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
