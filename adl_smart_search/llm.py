from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .prompts import PromptConfigError

logger = logging.getLogger(__name__)


class LLMProvider:
    # Заполняется провайдером: "stop" | "length" | ... (если API это сообщает).
    last_finish_reason: str = ""
    # Статистика токенов последнего запроса (если провайдер её отдаёт).
    # Например: {"input_tokens": 1234, "output_tokens": 456, "cached_tokens": 0}.
    last_usage: dict[str, int] = {}
    # Сумма по всем вызовам внутри одного ответа (автопродолжение может вызвать
    # модель несколько раз). Сбрасывается в chat_with_continuation.
    usage_total: dict[str, int] = {}

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        raise NotImplementedError

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)


class LLMError(RuntimeError):
    """
    Ошибка вызова LLM, пригодная для показа пользователю в UI.
    Не должна содержать секретов (ключи, токены).
    """

    def __init__(
        self,
        user_message: str,
        *,
        status_code: Optional[int] = None,
        error_type: str = "llm_error",
        details: str = "",
    ):
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details


# Сколько раз пробуем дописать ответ, если модель упёрлась в лимит токенов.
_MAX_ANSWER_CONTINUATIONS = 2
_TRUNCATION_NOTE = (
    "\n\n_… Ответ оборван лимитом модели. "
    "Если это повторяется, увеличьте лимит ответа (max_tokens / num_predict) в настройках LLM._"
)
_TRUNCATION_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit", "token-limit"}


def _is_length_finish(reason: Any) -> bool:
    return str(reason or "").strip().lower() in _TRUNCATION_REASONS


def _find_finish_reason(payload: Any, depth: int = 0) -> str:
    """Ищет finish_reason/truncated в ответе любого формата (recursive, с ограничением глубины)."""
    if depth > 6:
        return ""
    if isinstance(payload, dict):
        for key in ("finish_reason", "stop_reason", "done_reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if payload.get("truncated") is True or payload.get("incomplete") is True:
            return "length"
        for value in payload.values():
            found = _find_finish_reason(value, depth + 1)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_finish_reason(item, depth + 1)
            if found:
                return found
    return ""


def _merge_answer_parts(parts: list[str]) -> str:
    """Склеивает части ответа, убирая возможный повтор на стыке (модель начала с уже сказанного)."""
    out = ""
    for raw in parts:
        part = (raw or "").strip()
        if not part:
            continue
        if not out:
            out = part
            continue
        overlap = 0
        max_overlap = min(len(out), len(part), 300)
        for size in range(max_overlap, 0, -1):
            if out.endswith(part[:size]):
                overlap = size
                break
        rest = part[overlap:].lstrip()
        if rest:
            out = out + ("" if overlap else "\n") + rest
    return out


def chat_with_continuation(
    provider: LLMProvider,
    messages: list[dict[str, str]],
    *,
    max_continuations: int = _MAX_ANSWER_CONTINUATIONS,
    continuation_prompt: str = "",
) -> str:
    """Вызывает LLM и дописывает ответ, если провайдер сообщил об обрыве по лимиту токенов.

    Провайдер должен выставлять ``last_finish_reason``. Ограничение в 2 продолжения
    защищает от бесконечного цикла и лишних расходов. Текст автопродолжения
    берётся из ``config/prompts.json`` и передаётся вызывающим кодом.
    """
    convo = list(messages)
    parts: list[str] = []
    truncated = False
    attempts = max(0, int(max_continuations)) + 1
    # Новый ответ — новая статистика (автопродолжение суммирует вызовы).
    try:
        provider.usage_total = {}
    except Exception:
        pass
    for attempt in range(attempts):
        try:
            text = provider.chat(convo) or ""
        except Exception as e:
            # Первый вызов должен пробросить ошибку как раньше. Если уже есть
            # частичный ответ, лучше показать его с пометкой, чем потерять.
            if not parts:
                raise
            logger.warning("chat_with_continuation: continuation failed: %s", e)
            truncated = True
            break
        parts.append(text)
        truncated = _is_length_finish(getattr(provider, "last_finish_reason", ""))
        if not truncated or attempt == attempts - 1:
            break
        partial = _merge_answer_parts(parts)
        if not partial:
            break
        prompt = (continuation_prompt or "").strip()
        if not prompt:
            raise PromptConfigError(
                "Не задан промпт автопродолжения ответа: "
                "проверьте ключ continuation_prompt в config/prompts.json."
            )
        convo = convo + [
            {"role": "assistant", "content": partial},
            {"role": "user", "content": prompt},
        ]
    answer = _merge_answer_parts(parts)
    if truncated:
        answer = (answer + _TRUNCATION_NOTE).strip() if answer else _TRUNCATION_NOTE.strip()
    return answer



@dataclass
class OllamaConfig:
    model: str
    base_url: str = ""  # e.g. http://127.0.0.1:11434 or http://LAN_IP:11434
    options: Optional[dict[str, Any]] = None


class OllamaProvider(LLMProvider):
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def _client(self):
        import ollama

        if self.cfg.base_url:
            return ollama.Client(host=self.cfg.base_url)
        return ollama.Client()

    def chat(self, messages: list[dict[str, str]]) -> str:
        default_opts: dict[str, Any] = {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            # num_predict не ограничиваем: 900 токенов обрезали длинные ответы.
            # Лимит добавляется, только если пользователь включил его в настройках.
        }
        opts = dict(default_opts)
        if isinstance(self.cfg.options, dict):
            # подмешиваем только валидные значения (None игнорируем)
            for k, v in self.cfg.options.items():
                if v is None:
                    continue
                opts[k] = v

        self.last_finish_reason = ""
        client = self._client()
        resp = client.chat(
            model=self.cfg.model,
            messages=messages,
            # Настройки против "зацикливания" и мусорного вывода на малых моделях.
            options=opts,
        )
        try:
            if isinstance(resp, dict):
                reason = resp.get("done_reason") or resp.get("finish_reason") or ""
            else:
                reason = getattr(resp, "done_reason", "") or getattr(resp, "finish_reason", "")
            self.last_finish_reason = str(reason or "")
            return resp["message"]["content"] or ""
        except Exception:
            # фоллбек на старый формат
            # Склеим историю в один prompt (на случай старого интерфейса).
            logger.debug("OllamaProvider: chat response without 'message.content', fallback to generate")
            joined = "\n\n".join(
                [
                    f"{m.get('role','')}: {m.get('content','')}".strip()
                    for m in (messages or [])
                    if m.get("content")
                ]
            )
            resp2 = client.generate(model=self.cfg.model, prompt=joined)
            return (resp2 or {}).get("response", "") or ""


@dataclass
class OpenAICompatibleConfig:
    """
    Универсальный провайдер под корпоративный API в стиле OpenAI:
    POST {base_url}/chat/completions
    """

    base_url: str
    api_key: str
    model: str
    timeout_s: int = 120


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, cfg: OpenAICompatibleConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]]) -> str:
        import requests

        self.last_finish_reason = ""
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.2,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
            if r.status_code >= 400:
                raise LLMError(
                    _format_http_error_for_user(r.status_code, r.text),
                    status_code=r.status_code,
                    error_type="http_error",
                    details=_shorten_text(r.text),
                )
            data = r.json()
            try:
                choice = data["choices"][0]
                self.last_finish_reason = str(choice.get("finish_reason") or "")
                return (choice["message"]["content"] or "").strip()
            except Exception:
                logger.warning(
                    "OpenAICompatibleProvider: response missing choices[0].message.content: %s",
                    _shorten_text(r.text),
                )
                return ""
        except LLMError:
            raise
        except requests.exceptions.SSLError as e:
            raise LLMError(
                "Ошибка SSL при подключении к LLM API (сертификат/прокси).",
                error_type="ssl_error",
                details=str(e),
            )
        except requests.exceptions.ProxyError as e:
            raise LLMError(
                "Ошибка прокси при подключении к LLM API.",
                error_type="proxy_error",
                details=str(e),
            )
        except requests.exceptions.ConnectTimeout as e:
            raise LLMError(
                "Таймаут соединения с LLM API.",
                error_type="connect_timeout",
                details=str(e),
            )
        except requests.exceptions.ReadTimeout as e:
            raise LLMError(
                "Таймаут ожидания ответа от LLM API.",
                error_type="read_timeout",
                details=str(e),
            )
        except requests.exceptions.Timeout as e:
            raise LLMError(
                "Таймаут при обращении к LLM API.",
                error_type="timeout",
                details=str(e),
            )
        except requests.exceptions.RequestException as e:
            raise LLMError(
                "Сетевая ошибка при обращении к LLM API.",
                error_type="network_error",
                details=str(e),
            )


@dataclass
class CorporateAsyncConfig:
    """
    Корпоративный async API:
      POST {base_url}   (пример: https://.../message/async/v1)
      GET  {base_url}/{id}

    Авторизация:
      X-Api-Key: <key>
      accept: application/vnd.api+json
      Content-Type: application/json
    """

    base_url: str
    api_key: str
    provider: str
    model: str
    model_kwargs_json: str = ""  # UI-friendly JSON text
    timeout_s: int = 120
    poll_interval_s: float = 1.0
    poll_timeout_s: int = 120
    max_retries: int = 2


class CorporateAsyncProvider(LLMProvider):
    def __init__(self, cfg: CorporateAsyncConfig):
        self.cfg = cfg

    def chat(self, messages: list[dict[str, str]]) -> str:
        import json
        import time
        import uuid

        base = (self.cfg.base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("Corporate async API требует base_url")
        if not (self.cfg.api_key or "").strip():
            raise ValueError("Corporate async API требует api_key (X-Api-Key)")

        self.last_finish_reason = ""

        model_kwargs: dict[str, Any] = {}
        if (self.cfg.model_kwargs_json or "").strip():
            try:
                model_kwargs = json.loads(self.cfg.model_kwargs_json)
                if not isinstance(model_kwargs, dict):
                    model_kwargs = {}
            except Exception as e:
                raise LLMError(
                    "Некорректный JSON в model_kwargs (ожидается объект).",
                    error_type="bad_model_kwargs",
                    details=str(e),
                )

        headers = {
            "accept": "application/vnd.api+json",
            "Content-Type": "application/json",
            "X-Api-Key": self.cfg.api_key,
        }
        post_url = base
        poll_interval = max(0.2, float(self.cfg.poll_interval_s or 1.0))
        poll_timeout = max(1, int(self.cfg.poll_timeout_s))
        # Если API принял запрос, но ресурс так и не появился (только 404),
        # имеет смысл один раз повторить запрос с новым id: у async API бывает
        # несколько реплик и результат может «потеряться» на одной из них.
        attempts = 2

        for attempt in range(attempts):
            req_id = str(uuid.uuid4())
            payload = {
                "data": {
                    "id": req_id,
                    "type": "message",
                    "attributes": {
                        "provider": self.cfg.provider,
                        "model": self.cfg.model,
                        "model_kwargs": model_kwargs,
                        "messages": messages,
                    },
                }
            }

            # 1) POST
            post_resp = _requests_post_with_retries(
                post_url,
                headers=headers,
                json=payload,
                timeout_s=self.cfg.timeout_s,
                max_retries=self.cfg.max_retries,
            )
            resource_id = _extract_response_resource_id(post_resp) or req_id
            logger.info(
                "corp_async POST: status=%s req_id=%s resource_id=%s attempt=%s",
                getattr(post_resp, "status_code", "?"),
                req_id,
                resource_id,
                attempt + 1,
            )

            # 2) Poll GET until ready/timeout
            get_url = f"{base}/{resource_id}"
            deadline = time.time() + poll_timeout
            started = time.time()
            last_body = ""
            last_status: int | None = None
            not_ready = 0
            seen_non_404 = False

            while time.time() < deadline:
                r = _requests_get_with_retries(
                    get_url,
                    headers={"accept": "application/vnd.api+json", "X-Api-Key": self.cfg.api_key},
                    timeout_s=self.cfg.timeout_s,
                    max_retries=self.cfg.max_retries,
                )
                last_body = r.text or ""
                last_status = r.status_code
                # Некоторые реализации async API возвращают 404 первые N секунд
                # после принятия POST (ресурс ещё не материализован).
                if r.status_code == 404:
                    not_ready += 1
                    time.sleep(poll_interval)
                    continue
                if not seen_non_404:
                    seen_non_404 = True
                    logger.info(
                        "corp_async GET: resource_id=%s status=%s after %.1fs",
                        resource_id,
                        r.status_code,
                        time.time() - started,
                    )
                if r.status_code >= 400:
                    hint = _corp_length_error_hint(r.text)
                    message = (
                        hint
                        if hint
                        else f"{_format_http_error_for_user(r.status_code, r.text)} (GET {get_url})"
                    )
                    raise LLMError(
                        message,
                        status_code=r.status_code,
                        error_type="http_error",
                        details=_shorten_text(r.text),
                    )
                try:
                    data = r.json()
                except Exception:
                    # иногда backend может отвечать не JSON при ошибках/прокси
                    raise LLMError(
                        "Не удалось распарсить ответ LLM API (ожидали JSON).",
                        error_type="bad_json",
                        details=_shorten_text(last_body),
                    )

                done, text = _extract_corp_async_text(data)
                if done:
                    # finish_reason может лежать в attributes, message или nested model_response.
                    if text and not text.startswith("Ошибка провайдера:"):
                        self.last_finish_reason = _find_finish_reason(data)
                    return (text or "").strip()

                time.sleep(poll_interval)

            elapsed = time.time() - started
            logger.warning(
                "corp_async polling timeout: resource_id=%s attempt=%s "
                "elapsed=%.1fs last_status=%s not_ready_404=%s",
                resource_id,
                attempt + 1,
                elapsed,
                last_status,
                not_ready,
            )
            if attempt + 1 < attempts and not seen_non_404:
                logger.warning(
                    "corp_async: resource stayed 404 for %.1fs, retrying with a new id",
                    elapsed,
                )
                time.sleep(poll_interval)
                continue

            if last_status == 404:
                message = (
                    "МЭШ.ИИ.Хаб не опубликовал результат за "
                    f"{poll_timeout} с (последний ответ 404). "
                    "Похоже на задержку или сбой на стороне API/модели."
                )
            else:
                message = "Истекло время ожидания результата LLM (async polling timeout)."
            raise LLMError(
                message,
                error_type="poll_timeout",
                details=_shorten_text(last_body),
            )


@dataclass
class GigaChatConfig:
    credentials: str
    model: str
    scope: str = "GIGACHAT_API_PERS"
    base_url: str = ""
    ca_bundle_file: str = ""
    verify_ssl_certs: bool = True
    timeout_s: float = 120.0
    temperature: float = 0.2
    max_tokens: int = 4096


def _default_gigachat_ca_bundle() -> str:
    """Ищет корневой сертификат Минцифры рядом с проектом/exe."""
    candidates = [
        Path.cwd() / "sertificates" / "Russian_Trusted_Root_CA.cer",
        Path(__file__).resolve().parent.parent / "sertificates" / "Russian_Trusted_Root_CA.cer",
        Path(sys.executable).resolve().parent / "sertificates" / "Russian_Trusted_Root_CA.cer",
    ]
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.append(Path(meipass) / "sertificates" / "Russian_Trusted_Root_CA.cer")
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


def _resolve_gigachat_ca_bundle(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_file():
        return str(p)
    if not p.is_absolute():
        for base in (
            Path.cwd(),
            Path(__file__).resolve().parent.parent,
            Path(sys.executable).resolve().parent,
        ):
            candidate = base / p
            if candidate.is_file():
                return str(candidate)
    return path


class GigaChatProvider(LLMProvider):
    def __init__(self, cfg: GigaChatConfig):
        self.cfg = cfg
        self._client = None

    def _client_or_create(self):
        if self._client is not None:
            return self._client
        from gigachat import GigaChat

        kwargs: dict[str, Any] = {
            "credentials": self.cfg.credentials,
            "scope": self.cfg.scope,
            "model": self.cfg.model,
            "timeout": float(self.cfg.timeout_s),
            "verify_ssl_certs": bool(self.cfg.verify_ssl_certs),
            "max_retries": 2,
            "retry_backoff_factor": 0.5,
        }
        if (self.cfg.base_url or "").strip():
            kwargs["base_url"] = self.cfg.base_url.strip()
        ca_bundle = (self.cfg.ca_bundle_file or "").strip() or _default_gigachat_ca_bundle()
        if ca_bundle:
            kwargs["ca_bundle_file"] = _resolve_gigachat_ca_bundle(ca_bundle)
        self._client = GigaChat(**kwargs)
        return self._client

    def chat(self, messages: list[dict[str, str]]) -> str:
        from gigachat.exceptions import (
            AuthenticationError,
            BadRequestError,
            ForbiddenError,
            GigaChatException,
            NotFoundError,
            RateLimitError,
            RequestEntityTooLargeError,
            ServerError,
            UnprocessableEntityError,
        )
        from gigachat.models import ChatCompletionRequest, ChatMessage, ChatModelOptions

        self.last_finish_reason = ""
        self.last_usage = {}
        request = ChatCompletionRequest(
            model=self.cfg.model,
            messages=[
                ChatMessage(role=str(m.get("role") or "user"), content=str(m.get("content") or ""))
                for m in (messages or [])
                if m.get("content")
            ],
            model_options=ChatModelOptions(
                temperature=float(self.cfg.temperature),
                max_tokens=int(self.cfg.max_tokens),
            ),
        )
        try:
            response = self._client_or_create().chat.create(request)
        except AuthenticationError as e:
            raise LLMError(
                "401: GigaChat не принял авторизационный ключ. Проверьте GIGACHAT_CREDENTIALS.",
                status_code=401,
                error_type="auth_error",
                details=_shorten_text(str(e)),
            ) from e
        except ForbiddenError as e:
            raise LLMError(
                "403: нет доступа к GigaChat. Проверьте scope (GIGACHAT_API_PERS/B2B/CORP) и модель.",
                status_code=403,
                error_type="forbidden",
                details=_shorten_text(str(e)),
            ) from e
        except RateLimitError as e:
            retry_after = getattr(e, "retry_after", None)
            hint = f" Повторите через {retry_after} с." if retry_after else ""
            raise LLMError(
                "429: превышен лимит запросов GigaChat." + hint,
                status_code=429,
                error_type="rate_limit",
                details=_shorten_text(str(e)),
            ) from e
        except RequestEntityTooLargeError as e:
            raise LLMError(
                "413: контекст слишком большой для GigaChat. Уменьшите число выбранных результатов.",
                status_code=413,
                error_type="too_large",
                details=_shorten_text(str(e)),
            ) from e
        except (BadRequestError, UnprocessableEntityError) as e:
            raise LLMError(
                f"400: GigaChat отклонил запрос. {_shorten_text(str(e), 400)}",
                status_code=400,
                error_type="bad_request",
                details=_shorten_text(str(e)),
            ) from e
        except NotFoundError as e:
            raise LLMError(
                "404: модель GigaChat не найдена. Проверьте GIGACHAT_MODEL.",
                status_code=404,
                error_type="not_found",
                details=_shorten_text(str(e)),
            ) from e
        except ServerError as e:
            raise LLMError(
                "5xx: ошибка на стороне GigaChat.",
                status_code=500,
                error_type="server_error",
                details=_shorten_text(str(e)),
            ) from e
        except GigaChatException as e:
            text = str(e)
            if "certificate" in text.lower() or "ssl" in text.lower():
                raise LLMError(
                    "Ошибка SSL при подключении к GigaChat. Укажите сертификат Минцифры "
                    "или включите отключение проверки SSL (только для отладки).",
                    error_type="ssl_error",
                    details=_shorten_text(text),
                ) from e
            raise LLMError(
                f"Ошибка GigaChat: {_shorten_text(text, 400)}",
                error_type="gigachat_error",
                details=_shorten_text(text),
            ) from e
        except Exception as e:
            text = str(e)
            if "certificate" in text.lower() or "ssl" in text.lower():
                raise LLMError(
                    "Ошибка SSL при подключении к GigaChat. Укажите сертификат Минцифры "
                    "или включите отключение проверки SSL (только для отладки).",
                    error_type="ssl_error",
                    details=_shorten_text(text),
                ) from e
            raise LLMError(
                f"Ошибка GigaChat: {_shorten_text(text, 400)}",
                error_type="gigachat_error",
                details=_shorten_text(text),
            ) from e

        parts: list[str] = []
        finish_reason = str(getattr(response, "finish_reason", "") or "")
        for message in getattr(response, "messages", None) or []:
            role = str(getattr(message, "role", "") or "")
            if role and role != "assistant":
                continue
            for part in getattr(message, "content", None) or []:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
            if not finish_reason:
                finish_reason = str(getattr(message, "finish_reason", "") or "")
        self.last_finish_reason = finish_reason
        self._capture_usage(response)
        return "\n".join(p for p in parts if p).strip()

    def _capture_usage(self, response: Any) -> None:
        """Сохраняет статистику токенов: в UI видно, сколько стоил запрос."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        out: dict[str, int] = {}
        for src, dst in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = getattr(usage, src, None)
            if isinstance(value, int):
                out[dst] = value
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if isinstance(cached, int):
            out["cached_tokens"] = cached
        if not out:
            return
        self.last_usage = out
        totals = dict(getattr(self, "usage_total", None) or {})
        for key, value in out.items():
            totals[key] = int(totals.get(key, 0)) + int(value)
        self.usage_total = totals
        logger.info(
            "GigaChat usage: input=%s output=%s cached=%s total=%s",
            out.get("input_tokens"),
            out.get("output_tokens"),
            out.get("cached_tokens"),
            out.get("total_tokens"),
        )


def _shorten_text(s: str, limit: int = 1200) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "…"


def _format_http_error_for_user(status_code: int, body_text: str = "") -> str:
    if status_code in (401,):
        return "401 Unauthorized: неверный ключ/авторизация для LLM API."
    if status_code in (403,):
        return "403 Forbidden: нет прав на доступ к модели/эндпоинту LLM API."
    if status_code in (404,):
        return "404 Not Found: неверный URL/эндпоинт LLM API."
    if status_code in (408,):
        return "408 Request Timeout: сервер не успел обработать запрос LLM API."
    if status_code in (413,):
        return "413 Payload Too Large: слишком большой контекст/запрос для LLM API."
    if status_code in (429,):
        return "429 Too Many Requests: лимит запросов/квота LLM API (rate limit)."
    if 500 <= status_code <= 599:
        return f"{status_code} Server Error: ошибка на стороне LLM API."
    detail = _jsonapi_error_detail(body_text)
    if status_code == 400 and detail:
        return f"HTTP 400: {detail}"
    hint = detail or _shorten_text(body_text, 300)
    return f"HTTP {status_code}: ошибка LLM API. {hint}".strip()


def _jsonapi_error_detail(body_text: str) -> str:
    """Достаёт человекочитаемый detail из JSON:API-ошибки."""
    import json

    try:
        data = json.loads(body_text or "")
    except Exception:
        return ""
    errors = data.get("errors") if isinstance(data, dict) else None
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        return str(errors[0].get("detail") or errors[0].get("title") or "").strip()
    return ""


def _extract_response_resource_id(response) -> str:
    """``data.id`` из ответа POST на async endpoint (если API его вернул)."""
    try:
        data = response.json()
    except Exception:
        return ""
    if isinstance(data, dict):
        item = data.get("data")
        if isinstance(item, dict):
            return str(item.get("id") or "").strip()
    return ""


def _corp_length_error_hint(body_text: str) -> str:
    """Понятная подсказка, если провайдер вернул finish_reason=length."""
    import json

    try:
        data = json.loads(body_text or "")
    except Exception:
        return ""
    errors = data.get("errors") if isinstance(data, dict) else None
    if not isinstance(errors, list) or not errors or not isinstance(errors[0], dict):
        return ""
    meta = errors[0].get("meta")
    model_response = meta.get("model_response") if isinstance(meta, dict) else None
    choices = model_response.get("choices") if isinstance(model_response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    if str(choice.get("finish_reason") or "").strip().lower() != "length":
        return ""
    message = choice.get("message")
    reasoning = ""
    content = ""
    if isinstance(message, dict):
        reasoning = str(message.get("reasoning") or "").strip()
        content = str(message.get("content") or "").strip()
    if reasoning and not content:
        return (
            "Модель израсходовала лимит токенов на reasoning и не выдала ответ. "
            "Отключите enable_thinking в corp_catalog.json или увеличьте max_tokens."
        )
    return (
        "Модель оборвала ответ по лимиту токенов (finish_reason=length). "
        "Увеличьте max_tokens в corp_catalog.json."
    )



def _sleep_backoff_s(attempt: int) -> float:
    # attempt: 0..N
    # 0.5, 1.0, 2.0 ...
    return min(6.0, 0.5 * (2**max(0, attempt)))


def _requests_post_with_retries(url: str, *, headers: dict, json: dict, timeout_s: int, max_retries: int):
    import time

    import requests

    last_exc: Optional[Exception] = None
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            r = requests.post(url, headers=headers, json=json, timeout=timeout_s)
            if r.status_code in (429,) or 500 <= r.status_code <= 599:
                if attempt < max_retries:
                    time.sleep(_sleep_backoff_s(attempt))
                    continue
            if r.status_code >= 400:
                raise LLMError(
                    f"{_format_http_error_for_user(r.status_code, r.text)} (POST {url})",
                    status_code=r.status_code,
                    error_type="http_error",
                    details=_shorten_text(r.text),
                )
            return r
        except LLMError:
            raise
        except requests.exceptions.SSLError as e:
            raise LLMError(
                "Ошибка SSL при подключении к LLM API (сертификат/прокси).",
                error_type="ssl_error",
                details=str(e),
            )
        except requests.exceptions.ProxyError as e:
            raise LLMError("Ошибка прокси при подключении к LLM API.", error_type="proxy_error", details=str(e))
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(_sleep_backoff_s(attempt))
                continue
            raise LLMError("Таймаут при обращении к LLM API.", error_type="timeout", details=str(e))
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(_sleep_backoff_s(attempt))
                continue
            raise LLMError("Сетевая ошибка при обращении к LLM API.", error_type="network_error", details=str(e))
    raise LLMError("Не удалось выполнить запрос к LLM API.", error_type="network_error", details=str(last_exc or ""))


def _requests_get_with_retries(url: str, *, headers: dict, timeout_s: int, max_retries: int):
    import time

    import requests

    last_exc: Optional[Exception] = None
    for attempt in range(max(0, int(max_retries)) + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout_s)
            if r.status_code in (429,) or 500 <= r.status_code <= 599:
                if attempt < max_retries:
                    time.sleep(_sleep_backoff_s(attempt))
                    continue
            return r
        except requests.exceptions.SSLError as e:
            raise LLMError(
                "Ошибка SSL при подключении к LLM API (сертификат/прокси).",
                error_type="ssl_error",
                details=str(e),
            )
        except requests.exceptions.ProxyError as e:
            raise LLMError("Ошибка прокси при подключении к LLM API.", error_type="proxy_error", details=str(e))
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(_sleep_backoff_s(attempt))
                continue
            raise LLMError("Таймаут при обращении к LLM API.", error_type="timeout", details=str(e))
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(_sleep_backoff_s(attempt))
                continue
            raise LLMError("Сетевая ошибка при обращении к LLM API.", error_type="network_error", details=str(e))
    raise LLMError("Не удалось выполнить запрос к LLM API.", error_type="network_error", details=str(last_exc or ""))


def _extract_corp_async_text(resp_json: dict) -> tuple[bool, str]:
    """
    Возвращает (done, text).
    Поддерживает несколько возможных форматов результата.
    """
    # Ожидаем JSON:API style: {"data": {"attributes": ...}}
    data = resp_json.get("data") if isinstance(resp_json, dict) else None
    attrs = (data or {}).get("attributes") if isinstance(data, dict) else None
    if not isinstance(attrs, dict):
        # возможно прямой формат
        attrs = resp_json if isinstance(resp_json, dict) else {}

    # 1) Статус/готовность (может лежать и в attributes, и в data)
    data_status = data if isinstance(data, dict) else {}
    status = str(
        attrs.get("status")
        or attrs.get("state")
        or data_status.get("status")
        or data_status.get("state")
        or ""
    ).lower()
    done = status in (
        "done",
        "completed",
        "complete",
        "success",
        "succeeded",
        "ready",
        "finished",
        "ok",
    )

    # 2) Пытаемся извлечь текст
    # возможные поля: "content", "message", "output_text", "result", "response"
    text = ""
    if isinstance(attrs.get("content"), str):
        text = attrs.get("content") or ""
    elif isinstance(attrs.get("message"), dict) and isinstance(attrs.get("message", {}).get("content"), str):
        # формат из примера Mesh AI: attributes.message.content
        text = attrs.get("message", {}).get("content") or ""
    elif isinstance(attrs.get("output_text"), str):
        text = attrs.get("output_text") or ""
    elif isinstance(attrs.get("result"), str):
        text = attrs.get("result") or ""
    elif isinstance(attrs.get("response"), str):
        text = attrs.get("response") or ""
    else:
        # иногда возвращают массив сообщений
        msgs = attrs.get("messages") or attrs.get("output") or attrs.get("choices")
        if isinstance(msgs, list) and msgs:
            # берем последний content, который строка
            for item in reversed(msgs):
                if isinstance(item, dict):
                    c = item.get("content")
                    if isinstance(c, str) and c.strip():
                        text = c
                        break
                    m = item.get("message")
                    if isinstance(m, dict) and isinstance(m.get("content"), str):
                        text = m.get("content") or ""
                        break

    # 3) Если явно есть ошибка
    err = attrs.get("error") or attrs.get("errors")
    if err:
        return True, f"Ошибка провайдера: {_shorten_text(str(err), 800)}"

    # 4) Если статус не пришёл, но текст уже есть — считаем готовым.
    if text and not status:
        return True, text
    if done and text:
        return True, text
    return False, ""


def build_provider(
    mode: str,
    ollama_model: str,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_model: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_model: Optional[str] = None,
    api_kind: Optional[str] = None,
    corp_provider: Optional[str] = None,
    corp_model_kwargs_json: Optional[str] = None,
    corp_poll_timeout_s: Optional[int] = None,
    corp_poll_interval_s: Optional[float] = None,
    gigachat_credentials: Optional[str] = None,
    gigachat_scope: Optional[str] = None,
    gigachat_model: Optional[str] = None,
    gigachat_base_url: Optional[str] = None,
    gigachat_ca_bundle_file: Optional[str] = None,
    gigachat_verify_ssl_certs: Optional[bool] = None,
    gigachat_timeout_s: Optional[float] = None,
    gigachat_temperature: Optional[float] = None,
    gigachat_max_tokens: Optional[int] = None,
    ollama_options: Optional[dict[str, Any]] = None,
    ollama_base_url: Optional[str] = None,
) -> LLMProvider:
    mode = (mode or "").strip().lower()
    if mode in ("ollama", "local", ""):
        return OllamaProvider(
            OllamaConfig(
                model=ollama_model,
                base_url=(ollama_base_url or "").strip(),
                options=ollama_options if isinstance(ollama_options, dict) else None,
            )
        )
    if mode in ("api", "openai"):
        kind = (api_kind or "gigachat").strip().lower()
        if kind in ("openai", "openai_compatible", "compatible"):
            if not openai_base_url or not openai_api_key or not openai_model:
                raise ValueError(
                    "OpenAI-совместимый API требует openai_base_url, openai_api_key и openai_model"
                )
            return OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    base_url=openai_base_url,
                    api_key=openai_api_key,
                    model=openai_model,
                )
            )
        if kind in ("corp_async", "corporate_async", "async"):
            if not api_base_url or not api_key or not api_model or not (corp_provider or "").strip():
                raise ValueError("Corporate async API требует api_base_url, api_key, api_model и provider")
            return CorporateAsyncProvider(
                CorporateAsyncConfig(
                    base_url=api_base_url,
                    api_key=api_key,
                    provider=corp_provider or "",
                    model=api_model,
                    model_kwargs_json=(corp_model_kwargs_json or ""),
                    poll_timeout_s=int(corp_poll_timeout_s or 120),
                    poll_interval_s=float(corp_poll_interval_s or 1.0),
                )
            )
        if kind in ("gigachat", "giga"):
            if not (gigachat_credentials or "").strip() or not (gigachat_model or "").strip():
                raise ValueError("GigaChat требует credentials и model")
            return GigaChatProvider(
                GigaChatConfig(
                    credentials=(gigachat_credentials or "").strip(),
                    model=(gigachat_model or "").strip(),
                    scope=(gigachat_scope or "GIGACHAT_API_PERS").strip(),
                    base_url=(gigachat_base_url or "").strip(),
                    ca_bundle_file=(gigachat_ca_bundle_file or "").strip(),
                    verify_ssl_certs=bool(gigachat_verify_ssl_certs is not False),
                    timeout_s=float(gigachat_timeout_s or 120),
                    temperature=float(gigachat_temperature if gigachat_temperature is not None else 0.2),
                    max_tokens=int(gigachat_max_tokens or 4096),
                )
            )
        raise ValueError(f"Неизвестный API kind: {kind}")
    raise ValueError(f"Неизвестный режим LLM: {mode}")

