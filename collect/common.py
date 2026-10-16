"""수집 스크립트 공통 설정: 경로, API 키, 캐시, 재시도 요청."""
import json
import time
from pathlib import Path

import requests
import toml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
for d in (DATA, RAW, CACHE):
    d.mkdir(parents=True, exist_ok=True)

SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"

# AI 전체 분석 지역 (구, 법정동) → 상권 이름. 블로그·네이버 지도 검색어에는 사람들이 실제로 쓰는 상권 이름을 붙인다.
# 서울 전체는 생존력·위생만 계산한다. 모델 학습은 서울 전체를 사용한다.
TARGET_AREAS = {
    ("서대문구", "연희동"): "연희동",
    ("마포구", "연남동"): "연남동",
    ("서대문구", "창천동"): "신촌",
    ("마포구", "노고산동"): "신촌",
    ("마포구", "서교동"): "홍대",
    ("마포구", "동교동"): "홍대",
    ("마포구", "합정동"): "합정",
    ("마포구", "상수동"): "상수",
}
AREA_ORDER = ["연희동", "연남동", "신촌", "홍대", "합정", "상수"]


def secret(name: str) -> str:
    secrets = toml.load(SECRETS_PATH)
    if name not in secrets:
        raise KeyError(f"{SECRETS_PATH} 에 '{name}' 키를 추가하세요.")
    return secrets[name]


def get_json(url, *, method="GET", retries=3, backoff=2.0, **kwargs):
    """실패 시 지수 백오프로 재시도하는 JSON 요청."""
    for attempt in range(retries):
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            if resp.status_code == 429 and "per day" in resp.text:
                # 일일 쿼터 소진: 재시도해도 소용없으므로 바로 중단
                raise RuntimeError(f"429 일일 쿼터 소진: {resp.text[:300]}")
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")
            if 400 <= resp.status_code < 500:
                # 키·권한 문제 등은 재시도해도 같으므로 응답 본문과 함께 바로 중단
                raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            print(f"  ! 요청 실패({e}), {wait:.0f}초 후 재시도")
            time.sleep(wait)


class JsonCache:
    """키별 응답을 파일에 저장해 중단 후 재실행해도 이어서 수집되게 한다."""

    def __init__(self, name: str):
        self.path = CACHE / f"{name}.json"
        self.data = json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), "utf-8")
        tmp.replace(self.path)


NL = "\n"


class JsonlCache(JsonCache):
    """큰 캐시용: 한 줄에 키 하나씩 덧붙여 쓴다. JsonCache처럼 매번 전체 파일을 다시 쓰지 않아서
    수천 곳을 수집해도 느려지지 않는다. 예전 .json 캐시가 있으면 처음 한 번 옮겨 담는다."""

    def __init__(self, name: str):
        self.path = CACHE / f"{name}.jsonl"
        self.data = {}
        old = CACHE / f"{name}.json"
        if not self.path.exists() and old.exists():
            with self.path.open("w", encoding="utf-8") as f:
                for k, v in json.loads(old.read_text("utf-8")).items():
                    f.write(json.dumps([k, v], ensure_ascii=False) + NL)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        k, v = json.loads(line)
                    except ValueError:
                        continue  # 쓰다 끊긴 마지막 줄
                    self.data[k] = v
        ends_clean = not self.path.exists() or self.path.stat().st_size == 0 or self.path.read_bytes()[-1:] == NL.encode()
        self._f = self.path.open("a", encoding="utf-8")
        if not ends_clean:
            self._f.write(NL)  # 끊긴 줄 뒤에 바로 이어 쓰지 않게

    def __setitem__(self, key, value):
        self.data[key] = value
        self._f.write(json.dumps([key, value], ensure_ascii=False) + NL)

    def save(self):
        self._f.flush()
