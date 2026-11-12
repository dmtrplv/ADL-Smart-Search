import json

import pytest

from adl_smart_search.config import AppConfig
from adl_smart_search.llm import (
    CorporateAsyncConfig,
    CorporateAsyncProvider,
    GigaChatConfig,
    GigaChatProvider,
    LLMError,
    LLMProvider,
    OllamaConfig,
    OllamaProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    _corp_length_error_hint,
    _extract_corp_async_text,
    _find_finish_reason,
    _format_http_error_for_user,
    build_provider,
    chat_with_continuation,
)
from adl_smart_search.prompts import PromptConfigError
from adl_smart_search.ui.llm_provider import merged_corp_model_kwargs_json


def test_extract_corp_async_text_from_attributes_message_content():
    resp = {
        "data": {
            "attributes": {
                "status": "done",
                "message": {"content": "ok"},
            }
        }
    }
    done, text = _extract_corp_async_text(resp)
    assert done is True
    assert text == "ok"


def test_extract_corp_async_text_pending_without_text():
    resp = {"data": {"attributes": {"status": "running"}}}
    done, text = _extract_corp_async_text(resp)
    assert done is False
    assert text == ""


def test_extract_corp_async_text_error_is_terminal():
    resp = {"data": {"attributes": {"status": "done", "error": {"code": "X"}}}}
    done, text = _extract_corp_async_text(resp)
    assert done is True
    assert "Ошибка провайдера" in text


def test_extract_corp_async_text_waits_for_known_status():
    # Частичный content при незавершённом статусе нельзя показывать как готовый ответ.
    for status in ("generating", "inprogress", "streaming", "unknown"):
        resp = {"data": {"attributes": {"status": status, "content": "partial"}}}
        done, text = _extract_corp_async_text(resp)
        assert done is False
        assert text == ""


def test_extract_corp_async_text_without_status_returns_text():
    resp = {"data": {"attributes": {"content": "full"}}}
    done, text = _extract_corp_async_text(resp)
    assert done is True
    assert text == "full"


def test_find_finish_reason_nested():
    resp = {
        "meta": {
            "model_response": {
                "choices": [{"finish_reason": "length", "message": {"content": "cut"}}],
            }
        }
    }
    assert _find_finish_reason(resp) == "length"


class _SequencedProvider(LLMProvider):
    def __init__(self, chunks: list[str], reasons: list[str]):
        self.chunks = list(chunks)
        self.reasons = list(reasons)
        self.calls: list[list[dict[str, str]]] = []
        self.last_finish_reason = ""

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        self.last_finish_reason = self.reasons.pop(0)
        return self.chunks.pop(0)


def test_chat_with_continuation_appends_truncated_parts():
    provider = _SequencedProvider(["Первая часть", "вторая часть"], ["length", "stop"])

    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert answer == "Первая часть\nвторая часть"
    assert len(provider.calls) == 2
    assert provider.calls[1][-1]["role"] == "user"
    assert "Продолжи ответ" in provider.calls[1][-1]["content"]
    assert provider.calls[1][-2] == {"role": "assistant", "content": "Первая часть"}


def test_chat_with_continuation_requires_prompt_on_truncation():
    provider = _SequencedProvider(["часть"], ["length"])

    with pytest.raises(PromptConfigError):
        chat_with_continuation(provider, [{"role": "user", "content": "q"}])


def test_chat_with_continuation_deduplicates_overlap():
    provider = _SequencedProvider(["Ответ: привет", "привет, мир"], ["length", "stop"])

    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert answer == "Ответ: привет, мир"


def test_chat_with_continuation_adds_note_when_still_truncated():
    provider = _SequencedProvider(["a", "b", "c"], ["length", "length", "length"])

    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert len(provider.calls) == 3
    assert "Ответ оборван лимитом модели" in answer


def test_chat_with_continuation_keeps_partial_answer_when_continuation_fails():
    class FailingProvider(_SequencedProvider):
        def chat(self, messages):
            if self.calls:
                raise RuntimeError("boom")
            self.calls.append(list(messages))
            self.last_finish_reason = "length"
            return "частичный ответ"

    provider = FailingProvider([], [])
    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert "частичный ответ" in answer
    assert "Ответ оборван лимитом модели" in answer


def test_chat_with_continuation_does_nothing_on_stop():
    provider = _SequencedProvider(["готовый ответ"], ["stop"])

    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert answer == "готовый ответ"
    assert len(provider.calls) == 1


class _FakeOllamaClient:
    def __init__(self):
        self.kwargs: dict = {}

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return {"message": {"content": "ok"}, "done_reason": "stop"}


def test_ollama_does_not_cap_num_predict_by_default():
    client = _FakeOllamaClient()
    provider = OllamaProvider(OllamaConfig(model="m"))
    provider._client = lambda: client  # type: ignore[method-assign]

    assert provider.chat([{"role": "user", "content": "q"}]) == "ok"
    assert "num_predict" not in client.kwargs["options"]
    assert provider.last_finish_reason == "stop"


def test_ollama_applies_user_num_predict():
    client = _FakeOllamaClient()
    provider = OllamaProvider(OllamaConfig(model="m", options={"num_predict": 128}))
    provider._client = lambda: client  # type: ignore[method-assign]

    provider.chat([{"role": "user", "content": "q"}])

    assert client.kwargs["options"]["num_predict"] == 128


def test_openai_provider_exposes_finish_reason(monkeypatch):
    import requests

    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Resp())
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(base_url="https://example.test/v1", api_key="k", model="m")
    )

    assert provider.chat([{"role": "user", "content": "q"}]) == "cut"
    assert provider.last_finish_reason == "length"


class _FakeGigaPart:
    def __init__(self, text):
        self.text = text


class _FakeGigaMessage:
    def __init__(self, text, role="assistant", finish_reason="stop"):
        self.role = role
        self.content = [_FakeGigaPart(text)]
        self.finish_reason = finish_reason


class _FakeGigaResponse:
    def __init__(self, text, finish_reason="stop", usage=None):
        self.messages = [_FakeGigaMessage(text, finish_reason=finish_reason)]
        self.finish_reason = finish_reason
        self.usage = usage


class _FakeUsageDetails:
    def __init__(self, cached_tokens):
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens, cached_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.input_tokens_details = _FakeUsageDetails(cached_tokens)


class _FakeGigaChatAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


class _FakeGigaClient:
    def __init__(self, responses):
        self.chat = _FakeGigaChatAPI(responses)


def _gigachat_provider(responses):
    provider = GigaChatProvider(GigaChatConfig(credentials="key", model="GigaChat-2"))
    client = _FakeGigaClient(responses)
    provider._client_or_create = lambda: client
    provider._fake_client = client
    return provider


def test_gigachat_provider_maps_messages_and_text():
    provider = _gigachat_provider([_FakeGigaResponse("ответ")])

    out = provider.chat(
        [
            {"role": "system", "content": "система"},
            {"role": "user", "content": "вопрос"},
        ]
    )

    assert out == "ответ"
    assert provider.last_finish_reason == "stop"
    request = provider._fake_client.chat.requests[0]
    assert request.model == "GigaChat-2"
    assert [m.role for m in request.messages] == ["system", "user"]
    assert [m.content[0].text for m in request.messages] == ["система", "вопрос"]


def test_gigachat_continuation_uses_finish_reason():
    provider = _gigachat_provider(
        [
            _FakeGigaResponse("часть 1", finish_reason="length"),
            _FakeGigaResponse("часть 2", finish_reason="stop"),
        ]
    )

    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert "часть 1" in answer
    assert "часть 2" in answer
    assert len(provider._fake_client.chat.requests) == 2


def test_gigachat_ssl_error_is_user_friendly():
    class Broken:
        class chat:
            @staticmethod
            def create(request):
                raise RuntimeError("certificate verify failed")

    provider = GigaChatProvider(GigaChatConfig(credentials="key", model="GigaChat-2"))
    provider._client_or_create = lambda: Broken()

    try:
        provider.chat([{"role": "user", "content": "q"}])
    except Exception as e:
        assert "SSL" in str(e) or "сертификат" in str(e)
    else:
        raise AssertionError("LLMError expected")


def test_gigachat_usage_is_captured():
    provider = _gigachat_provider(
        [_FakeGigaResponse("ok", usage=_FakeUsage(input_tokens=1200, output_tokens=150, cached_tokens=800))]
    )

    provider.chat([{"role": "user", "content": "q"}])

    assert provider.last_usage == {
        "input_tokens": 1200,
        "output_tokens": 150,
        "total_tokens": 1350,
        "cached_tokens": 800,
    }
    assert provider.usage_total["input_tokens"] == 1200


def test_gigachat_usage_accumulates_over_continuations():
    first = _FakeGigaResponse("часть 1", finish_reason="length", usage=_FakeUsage(100, 10))
    second = _FakeGigaResponse("часть 2", finish_reason="stop", usage=_FakeUsage(200, 20))
    provider = _gigachat_provider([first, second])

    chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    # chat_with_continuation сбрасывает счётчик в начале ответа и суммирует вызовы.
    assert provider.usage_total["input_tokens"] == 300
    assert provider.usage_total["output_tokens"] == 30


def test_gigachat_usage_absent_does_not_break():
    provider = _gigachat_provider([_FakeGigaResponse("ok")])

    assert provider.chat([{"role": "user", "content": "q"}]) == "ok"
    assert provider.last_usage == {}


def test_build_provider_openai_uses_separate_settings():
    provider = build_provider(
        mode="api",
        ollama_model="",
        api_kind="openai_compatible",
        api_base_url="https://mesh.example/async",
        api_key="mesh-key",
        api_model="mesh-model",
        openai_base_url="https://openai.example/v1",
        openai_api_key="openai-key",
        openai_model="openai-model",
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.cfg.base_url == "https://openai.example/v1"
    assert provider.cfg.api_key == "openai-key"
    assert provider.cfg.model == "openai-model"


def test_build_provider_gigachat():
    provider = build_provider(
        mode="api",
        ollama_model="",
        api_kind="gigachat",
        gigachat_credentials="key",
        gigachat_model="GigaChat-2",
        gigachat_scope="GIGACHAT_API_PERS",
    )

    assert isinstance(provider, GigaChatProvider)
    assert provider.cfg.model == "GigaChat-2"


def test_corp_sends_default_max_tokens():
    cfg = AppConfig(adl_dir=".")

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert data["max_tokens"] == 8192


def test_corp_explicit_max_tokens_wins_when_checkbox_off():
    cfg = AppConfig(adl_dir=".")
    cfg.corp_max_tokens = 8192
    cfg.corp_model_kwargs_json = json.dumps({"max_tokens": 2048})

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert data["max_tokens"] == 2048


def test_corp_checkbox_overrides_json_max_tokens():
    cfg = AppConfig(adl_dir=".")
    cfg.corp_use_max_tokens = True
    cfg.corp_max_tokens = 3000
    cfg.corp_model_kwargs_json = json.dumps({"max_tokens": 2048})

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert data["max_tokens"] == 3000


def test_corp_temperature_goes_into_chat_template_kwargs():
    cfg = AppConfig(adl_dir=".")
    cfg.corp_use_temperature = True
    cfg.corp_temperature = 0.7
    cfg.corp_model_kwargs_json = json.dumps(
        {"cluster": "http://cluster", "service_url": "qwen"}
    )

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert "temperature" not in data
    assert data["chat_template_kwargs"]["temperature"] == 0.7
    assert data["cluster"] == "http://cluster"
    assert data["service_url"] == "qwen"


def test_corp_enable_thinking_false_goes_into_chat_template_kwargs():
    cfg = AppConfig(adl_dir=".")
    cfg.corp_use_enable_thinking = True
    cfg.corp_enable_thinking = False

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert data["chat_template_kwargs"]["enable_thinking"] is False


def test_corp_merges_existing_chat_template_kwargs():
    cfg = AppConfig(adl_dir=".")
    cfg.corp_use_temperature = True
    cfg.corp_temperature = 0.4
    cfg.corp_model_kwargs_json = json.dumps(
        {"chat_template_kwargs": {"custom": "keep"}}
    )

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert data["chat_template_kwargs"] == {"custom": "keep", "temperature": 0.4}


def test_corp_sampling_params_are_top_level():
    cfg = AppConfig(adl_dir=".")
    cfg.corp_use_top_p = True
    cfg.corp_top_p = 0.8
    cfg.corp_use_top_k = True
    cfg.corp_top_k = 20
    cfg.corp_use_presence_penalty = True
    cfg.corp_presence_penalty = 0.5
    cfg.corp_use_repetition_penalty = True
    cfg.corp_repetition_penalty = 1.15

    data = json.loads(merged_corp_model_kwargs_json(cfg))

    assert data["top_p"] == 0.8
    assert data["top_k"] == 20
    assert data["presence_penalty"] == 0.5
    assert data["repetition_penalty"] == 1.15


def test_default_api_kind_is_corp_async():
    from adl_smart_search.ui.llm_provider import DEFAULT_API_KIND

    assert DEFAULT_API_KIND == "corp_async"
    assert AppConfig(adl_dir=".").api_kind == "corp_async"


def test_corp_async_continuation_merges_parts(monkeypatch):
    import requests

    make_calls = {"post": 0, "get": 0}
    payloads = [
        {
            "data": {
                "attributes": {
                    "status": "done",
                    "message": {"content": "Part1"},
                    "model_response": {"choices": [{"finish_reason": "length"}]},
                }
            }
        },
        {
            "data": {
                "attributes": {
                    "status": "done",
                    "message": {"content": "Part2"},
                    "model_response": {"choices": [{"finish_reason": "stop"}]},
                }
            }
        },
    ]

    class Resp:
        status_code = 200
        text = ""

        def __init__(self, payload=None):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_post(*args, **kwargs):
        make_calls["post"] += 1
        return Resp()

    def fake_get(*args, **kwargs):
        payload = payloads[make_calls["get"]]
        make_calls["get"] += 1
        return Resp(payload)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    provider = CorporateAsyncProvider(
        CorporateAsyncConfig(
            base_url="https://example.test/async/v1",
            api_key="k",
            provider="p",
            model="m",
            poll_timeout_s=5,
            poll_interval_s=0.2,
        )
    )

    answer = chat_with_continuation(
        provider,
        [{"role": "user", "content": "q"}],
        continuation_prompt="Продолжи ответ.",
    )

    assert answer == "Part1\nPart2"
    assert make_calls == {"post": 2, "get": 2}


def test_corp_uses_resource_id_from_post_response(monkeypatch):
    import requests

    seen_urls: list[str] = []

    class Resp:
        def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
            self.status_code = status_code
            self.payload = payload
            self.text = text

        def json(self):
            return self.payload

    def fake_post(*args, **kwargs):
        return Resp(201, {"data": {"id": "server-resource-id"}})

    def fake_get(url, **kwargs):
        seen_urls.append(url)
        return Resp(
            200,
            {
                "data": {
                    "attributes": {
                        "status": "done",
                        "message": {"content": "ok"},
                    }
                }
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    provider = CorporateAsyncProvider(
        CorporateAsyncConfig(
            base_url="https://example.test/async/v1",
            api_key="k",
            provider="p",
            model="m",
            poll_timeout_s=5,
            poll_interval_s=0.2,
        )
    )

    assert provider.chat([{"role": "user", "content": "q"}]) == "ok"
    assert seen_urls[-1].endswith("/server-resource-id")


def test_corp_retries_once_when_resource_stays_404(monkeypatch):
    import requests

    calls = {"post": 0, "get": 0}

    class Resp:
        def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
            self.status_code = status_code
            self.payload = payload
            self.text = text

        def json(self):
            return self.payload

    def fake_post(*args, **kwargs):
        calls["post"] += 1
        return Resp(201, {"data": {"id": f"id-{calls['post']}"}})

    def fake_get(*args, **kwargs):
        calls["get"] += 1
        body = '{"errors":[{"status":"404","detail":"not found"}]}'
        return Resp(404, {"errors": [{"status": "404", "detail": "not found"}]}, text=body)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    provider = CorporateAsyncProvider(
        CorporateAsyncConfig(
            base_url="https://example.test/async/v1",
            api_key="k",
            provider="p",
            model="m",
            poll_timeout_s=1,
            poll_interval_s=0.2,
        )
    )

    with pytest.raises(LLMError) as exc:
        provider.chat([{"role": "user", "content": "q"}])

    assert calls["post"] == 2
    assert "не опубликовал результат" in exc.value.user_message


def test_format_http_error_uses_jsonapi_detail():
    body = json.dumps(
        {
            "errors": [
                {
                    "status": "400",
                    "detail": "Ответ оборван по лимиту токенов: увеличьте max_tokens.",
                }
            ]
        }
    )

    message = _format_http_error_for_user(400, body)

    assert "увеличьте max_tokens" in message
    assert "HTTP 400" in message


def test_corp_length_error_hint_for_reasoning():
    body = json.dumps(
        {
            "errors": [
                {
                    "status": "400",
                    "title": "Ошибка от AieduChatV2",
                    "meta": {
                        "model_response": {
                            "choices": [
                                {
                                    "finish_reason": "length",
                                    "message": {
                                        "reasoning": "We need answer in Russian...",
                                        "content": "",
                                    },
                                }
                            ]
                        }
                    },
                }
            ]
        }
    )

    hint = _corp_length_error_hint(body)

    assert "reasoning" in hint
    assert "enable_thinking" in hint
    assert "max_tokens" in hint

