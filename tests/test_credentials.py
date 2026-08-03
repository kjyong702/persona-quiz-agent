"""자격증명 갱신.

여기서 확인하는 것은 **키가 바뀌었을 때 프로세스가 알아채는가**다.
키 회전 자체는 제공사와 시크릿 매니저의 일이고, 우리 몫은 회전된 값을
언제 읽느냐뿐이다.

목으로 클라이언트를 만든다. 진짜 `AsyncOpenAI`를 만들면 키가 유효한지
확인하지 않으므로 테스트로서 얻는 것이 없고, 대신 무엇으로 만들어졌는지를
봐야 하기 때문이다.
"""

import pytest

from app.core import credentials


class FakeClient:
    def __init__(self, key: str) -> None:
        self.key = key


@pytest.fixture
def refreshable(monkeypatch: pytest.MonkeyPatch) -> credentials.RefreshableClient:
    monkeypatch.setenv(credentials.ENV_VAR, "key-one")
    return credentials.RefreshableClient[FakeClient](
        build=FakeClient, name="test", ttl_seconds=0.0
    )


def _error(message: str) -> Exception:
    return RuntimeError(message)


def test_키가_같으면_같은_객체를_돌려준다(
    refreshable: credentials.RefreshableClient,
) -> None:
    """**커넥션 풀 때문이다.**

    TTL이 지날 때마다 새로 만들면 httpx 커넥션이 매번 끊기고 다음 호출들이
    TLS 핸드셰이크를 다시 한다. 값이 그대로면 재사용해야 한다.
    """
    first = refreshable.get(_error)
    second = refreshable.get(_error)
    assert first is second


def test_키가_바뀌면_새로_만든다(
    refreshable: credentials.RefreshableClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = refreshable.get(_error)
    monkeypatch.setenv(credentials.ENV_VAR, "key-two")
    second = refreshable.get(_error)

    assert first is not second
    assert second.key == "key-two"


def test_TTL_안에는_환경을_다시_읽지_않는다(monkeypatch: pytest.MonkeyPatch) -> None:
    """TTL이 있는 이유. 매 호출마다 환경을 뒤지지 않는다."""
    monkeypatch.setenv(credentials.ENV_VAR, "key-one")
    client = credentials.RefreshableClient[FakeClient](
        build=FakeClient, name="test", ttl_seconds=3600.0
    )
    first = client.get(_error)
    monkeypatch.setenv(credentials.ENV_VAR, "key-two")

    assert client.get(_error) is first  # TTL이 안 지났으니 옛 키 그대로


def test_키가_없으면_준_예외를_던진다(monkeypatch: pytest.MonkeyPatch) -> None:
    """호출자마다 다른 예외를 쓴다. 임베딩과 판정의 실패가 섞이면 안 된다."""
    monkeypatch.delenv(credentials.ENV_VAR, raising=False)
    monkeypatch.setattr(credentials.settings, "openai_api_key", None)
    client = credentials.RefreshableClient[FakeClient](build=FakeClient, name="test")

    with pytest.raises(RuntimeError, match=credentials.ENV_VAR):
        client.get(_error)


def test_환경변수가_설정값보다_우선한다(monkeypatch: pytest.MonkeyPatch) -> None:
    """컨테이너에서 시크릿을 갈아끼우면 보통 환경변수로 온다.

    `settings`는 부팅 시 한 번 읽은 값이라 회전을 못 따라간다.
    """
    monkeypatch.setattr(credentials.settings, "openai_api_key", "from-settings")
    monkeypatch.setenv(credentials.ENV_VAR, "from-env")
    assert credentials.current_api_key() == "from-env"


def test_설정값으로_물러난다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(credentials.ENV_VAR, raising=False)
    monkeypatch.setattr(credentials.settings, "openai_api_key", "from-settings")
    assert credentials.current_api_key() == "from-settings"


def test_지문에_키가_남지_않는다() -> None:
    """**로그와 헬스체크에 나가는 값이다.** 원본이 복원되면 안 된다."""
    secret = "sk-verylongsecretvalue-1234567890"
    printed = credentials.fingerprint(secret)

    assert secret not in printed
    assert "sk-" not in printed
    assert len(printed) == 8
    assert credentials.fingerprint(secret) == printed  # 같은 키는 같은 지문
    assert credentials.fingerprint("sk-other") != printed
    assert credentials.fingerprint(None) == "none"


def test_상태에_키가_안_들어간다(
    refreshable: credentials.RefreshableClient,
) -> None:
    """헬스체크가 이걸 그대로 내보낸다."""
    refreshable.get(_error)
    status = refreshable.status()

    assert "key-one" not in str(status)
    assert status["key_fingerprint"] == credentials.fingerprint("key-one")
    assert status["loaded_at"] is not None


def test_두_모듈이_각자_클라이언트를_들고_있다() -> None:
    """**한쪽만 회전되면 절반만 살아 있는 상태가 된다.**

    임베딩과 판정이 같은 키를 쓰지만 클라이언트는 따로다. 둘 다 같은 방식으로
    갱신되는지 확인한다. 한쪽을 고치고 다른 쪽을 잊는 것이 실제로 하기 쉬운 실수다.
    """
    from app.core import embedding, llm

    assert isinstance(embedding._clients, credentials.RefreshableClient)
    assert isinstance(llm._clients, credentials.RefreshableClient)
    assert embedding._clients is not llm._clients
