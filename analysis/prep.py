"""분석 공통: 서울 일반음식점 데이터 불러오기·정리."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)


def load_clean(verbose=print):
    """영업기간 오류와 축제·행사 임시 부스를 뺀 서울 일반음식점."""
    d = pd.read_csv(DATA / "seoul_restaurants.csv", low_memory=False,
                    usecols=["mgt_no", "name", "category", "open_date", "close_date",
                             "addr_jibun", "addr_road", "area", "closed", "duration_days",
                             "lat", "lng"])
    verbose(f"전체 {len(d):,}곳, 폐업 {d['closed'].sum():,}곳 ({d['closed'].mean():.0%})")
    d["open_date"] = pd.to_datetime(d["open_date"], errors="coerce")
    d["close_date"] = pd.to_datetime(d["close_date"], errors="coerce")
    addr = d["addr_jibun"].fillna(d["addr_road"])
    d["gu"] = addr.str.extract(r"서울특별시\s+(\S+구)")[0]
    d["dong"] = d["addr_jibun"].str.extract(r"\S+구\s+(\S+동)")[0]
    d["years"] = d["duration_days"] / 365.25
    d["open_year"] = d["open_date"].dt.year
    d["close_year"] = d["close_date"].dt.year

    bad = d["duration_days"].isna() | (d["duration_days"] < 0) | d["open_date"].isna()
    verbose(f"영업기간·개업일 오류: {bad.sum():,}곳 → 제외")
    d = d[~bad].copy()
    d.loc[d["years"] == 0, "years"] = 1 / 365.25  # 개업 당일 폐업은 하루로

    # 축제·행사 임시 부스: "(한시적)" 상호이거나, 10㎡ 미만인데 60일 안에 폐업
    popup = d["name"].str.contains("한시적", na=False) | (
        (d["area"] < 10) & (d["closed"] == 1) & (d["duration_days"] <= 60))
    verbose(f"임시 부스(한시적·초소형 단기): {popup.sum():,}곳 → 제외")
    d = d[~popup].reset_index(drop=True)
    verbose(f"분석 대상 {len(d):,}곳, 폐업 {d['closed'].mean():.0%}")
    return d
