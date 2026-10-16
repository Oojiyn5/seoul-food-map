"""1단계: 일반음식점 인허가 데이터 → 서울 전체(모델 학습·서울 지도용) + AI 분석 지역(연희·연남·신촌·홍대·합정·상수).

사전 준비:
  공공데이터포털 '식품_일반음식점_서울특별시' CSV (LOCALDATA 이관 후 버전, 2026-10 확인)
  를 내려받아 data/raw/ 폴더에 넣는다. (파일명은 상관없음, 가장 최근 csv를 사용)

출력:
  data/seoul_restaurants.csv   서울 전체, 폐업 포함 (생존분석 라벨)
  data/area_restaurants.csv    AI 분석 지역, 폐업 포함 (dong: 법정동, area_name: 상권 이름)

서울 전체 파일이 이미 있으면 원본 CSV를 다시 읽지 않고 지역만 다시 고른다 (원본부터 다시: --refresh)
"""
import sys

import pandas as pd
from pyproj import Transformer

from common import AREA_ORDER, DATA, RAW, TARGET_AREAS

# 원본 컬럼명은 배포처/시기마다 조금씩 달라서 후보를 여러 개 둔다.
COLUMN_ALIASES = {
    "mgt_no": ["관리번호"],
    "name": ["사업장명", "업소명"],
    "category": ["업태구분명", "위생업태명"],
    "status": ["영업상태명"],
    "status_detail": ["상세영업상태명"],
    "open_date": ["인허가일자"],
    "close_date": ["폐업일자"],
    "addr_jibun": ["소재지전체주소", "소재지지번주소", "지번주소"],
    "addr_road": ["도로명전체주소", "소재지도로명주소", "도로명주소"],
    "area": ["소재지면적", "시설총규모"],
    "x": ["좌표정보(x)", "좌표정보x(epsg5174)", "좌표정보(X)", "x"],
    "y": ["좌표정보(y)", "좌표정보y(epsg5174)", "좌표정보(Y)", "y"],
    "main_food": ["주된음식"],
}


def find_csv():
    files = sorted(RAW.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        sys.exit(f"data/raw 에 CSV가 없습니다. 안내에 따라 일반음식점 CSV를 받아 넣어주세요: {RAW}")
    return files[0]


def detect_encoding(path):
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                f.read(100_000)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError("인코딩을 판별하지 못했습니다.")


def build_rename(columns):
    normalized = {c.strip().lower().replace(" ", ""): c for c in columns}
    rename = {}
    for std, candidates in COLUMN_ALIASES.items():
        for cand in candidates:
            key = cand.lower().replace(" ", "")
            if key in normalized:
                rename[normalized[key]] = std
                break
    missing = {"mgt_no", "name", "status", "open_date", "addr_jibun"} - set(rename.values())
    if missing:
        sys.exit(f"필수 컬럼을 찾지 못했습니다: {missing}\n실제 컬럼: {list(columns)}")
    return rename


def build_seoul():
    path = find_csv()
    enc = detect_encoding(path)
    print(f"읽는 중: {path.name} ({enc})")

    header = pd.read_csv(path, encoding=enc, nrows=0).columns
    rename = build_rename(header)

    chunks = []
    for chunk in pd.read_csv(path, encoding=enc, usecols=list(rename), dtype=str, chunksize=200_000):
        chunk = chunk.rename(columns=rename).reindex(columns=list(COLUMN_ALIASES))
        addr = chunk["addr_jibun"].fillna("") + " " + chunk["addr_road"].fillna("")
        chunks.append(chunk[addr.str.contains("서울특별시", regex=False)])
    df = pd.concat(chunks, ignore_index=True)
    print(f"서울 일반음식점: {len(df):,}건")

    # 날짜/숫자 정리
    for col in ("open_date", "close_date"):
        df[col] = pd.to_datetime(df[col].str.replace("-", "").str[:8], format="%Y%m%d", errors="coerce")
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["closed"] = df["status"].str.contains("폐업", na=False).astype(int)

    # 생존분석용: 영업 기간(일). 영업 중이면 오늘까지 (중도절단)
    end = df["close_date"].where(df["closed"] == 1, pd.Timestamp.today().normalize())
    df["duration_days"] = (end - df["open_date"]).dt.days

    # 좌표 변환: 중부원점 TM(EPSG:5174) → WGS84
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    has_xy = df["x"].notna() & df["y"].notna()
    tf = Transformer.from_crs("EPSG:5174", "EPSG:4326", always_xy=True)
    lng, lat = tf.transform(df.loc[has_xy, "x"].values, df.loc[has_xy, "y"].values)
    df.loc[has_xy, "lng"], df.loc[has_xy, "lat"] = lng, lat

    df = df.drop_duplicates("mgt_no")
    df.to_csv(DATA / "seoul_restaurants.csv", index=False, encoding="utf-8-sig")
    print(f"좌표 없는 행 비율: {(~has_xy).mean():.1%}")
    return df


def main():
    seoul_path = DATA / "seoul_restaurants.csv"
    if seoul_path.exists() and "--refresh" not in sys.argv:
        df = pd.read_csv(seoul_path, dtype={"mgt_no": str}, low_memory=False)
        print(f"저장된 서울 파일 사용: {len(df):,}건 (원본부터 다시 하려면 --refresh)")
    else:
        df = build_seoul()

    # 지번 주소의 구·법정동으로 지역을 고른다 (도로명 주소 괄호 안 동 이름은 행정동이 섞여 있어 보조로만)
    gu = df["addr_jibun"].str.extract(r"서울특별시\s+(\S+구)")[0].fillna(df["addr_road"].str.extract(r"서울특별시\s+(\S+구)")[0])
    dong = df["addr_jibun"].str.extract(r"\S+구\s+(\S+동)")[0].fillna(df["addr_road"].str.extract(r"\((\S+동)[,)]")[0])
    df["dong"] = dong
    df["area_name"] = [TARGET_AREAS.get((g, d)) for g, d in zip(gu, dong)]
    target = df[df["area_name"].notna()]
    target.to_csv(DATA / "area_restaurants.csv", index=False, encoding="utf-8-sig")

    open_now = target[target["closed"] == 0]
    print(f"AI 분석 지역: 전체 {len(target):,}건 / 영업 중 {len(open_now):,}건 / 폐업 {target['closed'].sum():,}건")
    print("지역별 영업 중:")
    print(open_now["area_name"].value_counts().reindex(AREA_ORDER).to_string())
    print("영업 중 업태 분포:")
    print(open_now["category"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
