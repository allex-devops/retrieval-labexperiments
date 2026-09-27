import httpx
import pytest

from retrieval_lab.llm import LLM, LLMError


def make(handler, **kw):
    slept = []
    llm = LLM("http://model.test/v1", "k", "chat-m", "embed-m", transport=httpx.MockTransport(handler), sleep=slept.append, **kw)
    return llm, slept


def test_missing_settings_fail_loudly(monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_MODEL", "EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMError, match="LLM_BASE_URL"):
        LLM()
    llm = LLM("http://model.test/v1")
    with pytest.raises(LLMError, match="LLM_MODEL"):
        llm.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(LLMError, match="EMBED_MODEL"):
        llm.embed(["x"])


def test_retries_transient_errors_honouring_retry_after():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "7"})
        if len(calls) == 2:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm, slept = make(handler)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert len(calls) == 3 and slept[0] == 7.0
    assert calls[0].headers["authorization"] == "Bearer k"


def test_client_errors_are_not_retried():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(401, text="bad key")

    llm, _ = make(handler)
    with pytest.raises(LLMError, match="401"):
        llm.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1


def test_gives_up_after_the_retry_budget():
    llm, slept = make(lambda req: httpx.Response(503), retries=2)
    with pytest.raises(LLMError, match="gave up"):
        llm.embed(["x"])
    assert len(slept) == 2


def test_embeddings_come_back_in_input_order():
    def handler(req):
        return httpx.Response(200, json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})

    llm, _ = make(handler)
    assert llm.embed(["a", "b"]) == [[1.0], [2.0]]
