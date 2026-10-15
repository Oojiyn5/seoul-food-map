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

# 대상 지역 (지도에 표시할 범위). 모델 학습은 서울 전체를 사용한다.
TARGET_GU = "서대문구"
TARGET_DONG = "연희동"


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
