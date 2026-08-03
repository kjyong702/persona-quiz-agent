# 빌드와 실행을 나눈다. uv와 빌드 도구가 최종 이미지에 남지 않게 하려는 것이고,
# 의존성 설치 층을 소스와 분리해 코드만 고쳤을 때 재설치를 건너뛰려는 것이기도 하다
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
# 잠금 파일과 프로젝트 정의만 먼저 복사한다. 소스를 같이 넣으면 코드 한 줄만
# 고쳐도 이 층이 무효가 되어 의존성을 매번 다시 받는다
COPY pyproject.toml uv.lock ./
# --frozen: 잠금 파일을 갱신하지 않는다. 빌드가 조용히 다른 버전을 쓰면
#           로컬에서 통과한 평가가 이미지에서는 다른 결과를 낸다
# --no-dev: 테스트 의존성은 런타임에 필요 없다
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim

# 루트로 돌리지 않는다. 컨테이너 안에서 무언가 실행되더라도 권한을 줄인다
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --chown=app:app app ./app
COPY --chown=app:app prompts ./prompts
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app seed ./seed

# **데이터 디렉터리를 미리 만들고 소유권을 넘긴다.**
# 이걸 안 하면 컨테이너가 뜨자마자 죽는다. 볼륨을 붙이면 /data가 root 소유로
# 생기는데 프로세스는 uid 1000이라 SQLite 파일을 못 만든다.
# ("unable to open database file"만 나오고 권한 문제라는 말은 안 나온다)
#
# Docker는 **비어 있는 named volume을 마운트할 때 이미지의 해당 경로 내용과
# 소유권을 그대로 복사한다.** 그래서 여기서 chown해두면 볼륨도 app 소유가 된다.
# bind mount는 이 규칙이 적용되지 않으므로 호스트 쪽에서 맞춰야 한다
RUN mkdir -p /data && chown app:app /data
VOLUME ["/data"]

# **시크릿은 이미지에 넣지 않는다.** .env는 .dockerignore로 막고
# 실행 시 환경변수나 시크릿 마운트로 주입한다
USER app

EXPOSE 8000

# 헬스체크는 /healthz다. /readyz를 쓰면 앵커가 없을 때 컨테이너가 재시작되고,
# 재시작해도 앵커는 안 생기므로 루프에 빠진다.
# **재시작으로 고쳐지지 않는 문제는 liveness가 아니다**
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
