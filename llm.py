"""Единый вызов LLM с фолбэком по провайдерам, ключам и моделям.

Порядок обхода: для каждого провайдера — свои модели; кончились (квота, отказ) —
следующий провайдер. Все ходы делаются с temperature=0 и strict JSON-схемой,
так что ответ любого провайдера разбирается одинаково.

Ключи читаются из .env:
    GEMINI_API_KEY, GEMINI_API_KEY_2..9   основной
    GROQ_API_KEY                          резерв, бесплатный, лимит 8000 ток/мин
    OPENROUTER_API_KEY                    резерв, платный по факту использования
"""

import json
import os
import time

QUOTA_MARKERS = ("PerDay", "RESOURCE_EXHAUSTED", "rate_limit", "429",
                 "NOT_FOUND", "404", "402", "413", "insufficient",
                 "credit balance", "no credits", "quota", "billing",
                 "exceeded", "too low")


def _gemini_keys() -> list[str]:
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 10)]
    return [os.environ[n] for n in names if os.environ.get(n)]


class Provider:
    """Один способ сходить в модель: провайдер + список его моделей."""

    def __init__(self, name: str, key: str, models: list[str]):
        self.name, self.key, self.models = name, key, models

    def call(self, model: str, prompt: str, schema: dict) -> str:
        raise NotImplementedError


class Gemini(Provider):
    """Прямой REST — SDK google-genai ломается при конфликте зависимостей
    («client has been closed») при полностью рабочих ключах."""

    ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
                "{model}:generateContent?key={key}")

    def call(self, model, prompt, schema):
        import urllib.error
        import urllib.request

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_schema(schema),
            },
        }
        req = urllib.request.Request(
            self.ENDPOINT.format(model=model, key=self.key),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}") from None
        return data["candidates"][0]["content"]["parts"][0]["text"]


def _gemini_schema(schema: dict, defs: dict | None = None):
    """JSON Schema от pydantic -> формат, который принимает Gemini REST:
    $ref/$defs разворачиваются, служебные ключи отбрасываются."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if isinstance(schema, list):
        return [_gemini_schema(v, defs) for v in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return _gemini_schema(defs.get(name, {}), defs)
    allowed = {"type", "properties", "items", "required", "enum", "description"}
    out = {}
    for key, value in schema.items():
        if key not in allowed:
            continue
        if key == "properties":  # это имена полей, а не вложенная схема
            out[key] = {name: _gemini_schema(sub, defs) for name, sub in value.items()}
        else:
            out[key] = _gemini_schema(value, defs)
    if out.get("type") == "object" and out.get("properties"):
        out["required"] = list(out["properties"])
    return out


def _strict(node):
    """strict-режим OpenAI/Groq требует additionalProperties:false и полный
    required на каждом объекте схемы; pydantic их не проставляет."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for v in node.values():
            _strict(v)
    elif isinstance(node, list):
        for v in node:
            _strict(v)
    return node


class OpenAICompatible(Provider):
    """Groq и OpenRouter говорят на одном протоколе."""

    base_url = ""

    def call(self, model, prompt, schema):
        import copy

        from openai import OpenAI
        client = OpenAI(api_key=self.key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "covenant_spec", "strict": True,
                "schema": _strict(copy.deepcopy(schema))}})
        return resp.choices[0].message.content


class Anthropic(Provider):
    """Structured output у Claude делается через forced tool use."""

    def call(self, model, prompt, schema):
        import anthropic
        client = anthropic.Anthropic(api_key=self.key)
        resp = client.messages.create(
            model=model, max_tokens=8192,  # temperature у новых моделей не задаётся
            tools=[{"name": "emit_spec",
                    "description": "Вернуть спецификацию ковенанта",
                    "input_schema": schema}],
            tool_choice={"type": "tool", "name": "emit_spec"},
            messages=[{"role": "user", "content": prompt}])
        for block in resp.content:
            if block.type == "tool_use":
                return json.dumps(block.input, ensure_ascii=False)
        raise RuntimeError("Claude не вернул tool_use")


class Groq(OpenAICompatible):
    base_url = "https://api.groq.com/openai/v1"


class OpenRouter(OpenAICompatible):
    base_url = "https://openrouter.ai/api/v1"


def build_chain(main_model: str | None = None) -> list[Provider]:
    """Провайдеры по убыванию приоритета; пустые ключи пропускаются."""
    chain: list[Provider] = []
    # Claude — платный, но лучший по качеству разбора: ставим первым, если
    # ключ задан; бесплатные провайдеры ниже подхватят при исчерпании средств
    if os.environ.get("ANTHROPIC_API_KEY"):
        chain.append(Anthropic("claude", os.environ["ANTHROPIC_API_KEY"],
                               [os.environ.get("CLAUDE_MODEL", "claude-sonnet-5"),
                                "claude-haiku-4-5-20251001"]))
    gemini_models = [main_model or os.environ.get("MODEL_MAIN", "gemini-3.6-flash"),
                     "gemini-3.5-flash", "gemini-flash-latest",
                     "gemini-3-flash-preview", "gemini-3.5-flash-lite",
                     "gemini-flash-lite-latest"]
    for i, key in enumerate(_gemini_keys(), 1):
        chain.append(Gemini(f"gemini#{i}", key, gemini_models))
    if os.environ.get("GROQ_API_KEY"):
        chain.append(Groq("groq", os.environ["GROQ_API_KEY"],
                          ["openai/gpt-oss-120b", "llama-3.3-70b-versatile",
                           "qwen/qwen3.6-27b", "openai/gpt-oss-20b"]))
    if os.environ.get("OPENROUTER_API_KEY"):
        chain.append(OpenRouter("openrouter", os.environ["OPENROUTER_API_KEY"],
                                ["deepseek/deepseek-chat-v3.1", "openai/gpt-4o-mini"]))
    return chain


class Router:
    """Держит текущего провайдера/модель и сдвигается при отказах."""

    def __init__(self, main_model: str | None = None):
        self.chain = build_chain(main_model)
        if not self.chain:
            raise SystemExit("Нет ни одного ключа в .env (GEMINI_API_KEY / "
                             "GROQ_API_KEY / OPENROUTER_API_KEY)")
        self.pi = self.mi = 0
        print(f"  провайдеры: {', '.join(p.name for p in self.chain)}")

    @property
    def current(self) -> str:
        p = self.chain[self.pi]
        return f"{p.name}/{p.models[self.mi]}"

    def _advance(self) -> bool:
        """Следующая модель, а если кончились — следующий провайдер."""
        p = self.chain[self.pi]
        if self.mi + 1 < len(p.models):
            self.mi += 1
        elif self.pi + 1 < len(self.chain):
            self.pi, self.mi = self.pi + 1, 0
        else:
            return False
        print(f"  .. переключаюсь на {self.current}")
        return True

    def complete(self, prompt: str, schema: dict, attempts: int = 8) -> str:
        last = None
        for _ in range(attempts):
            p = self.chain[self.pi]
            try:
                return p.call(p.models[self.mi], prompt, schema)
            except Exception as e:
                last = e
                if any(m in str(e) for m in QUOTA_MARKERS):
                    if self._advance():
                        continue
                    raise RuntimeError(f"Все провайдеры исчерпаны: {last}")
                time.sleep(5)
        raise RuntimeError(f"Не удалось получить ответ: {last}")
