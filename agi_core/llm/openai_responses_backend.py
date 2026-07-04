from __future__ import annotations

import importlib
from typing import Any

from .json_utils import coerce_json_dict


class OpenAIResponsesBackend:
    """Optional OpenAI Responses API backend for JSON-only specialist proposals."""

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.available = False
        self.last_error: str | None = None
        self._client: Any = None
        self._load()

    def generate_json(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any] | None:
        if not self.available or self._client is None:
            return None
        try:
            response = self._client.responses.create(
                model=self.model,
                input=self._build_input(prompt),
                max_output_tokens=max_tokens,
                temperature=temperature,
                text={"format": {"type": "json_object"}},
            )
            text = self._extract_text(response)
            payload = coerce_json_dict(text)
            if payload is None:
                self.last_error = "response did not contain a valid JSON object"
            return payload
        except Exception as exc:
            self.last_error = self._format_error(exc)
            return None

    def _load(self) -> None:
        if not self.api_key:
            self.last_error = "OPENAI_API_KEY missing"
            return
        try:
            module = importlib.import_module("openai")
        except Exception as exc:
            self.last_error = repr(exc)
            return
        client_cls = getattr(module, "OpenAI", None)
        if client_cls is None:
            self.last_error = "openai.OpenAI not found"
            return
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.timeout_seconds is not None:
            client_kwargs["timeout"] = self.timeout_seconds
        try:
            self._client = client_cls(**client_kwargs)
        except Exception as exc:
            self.last_error = repr(exc)
            return
        self.available = True

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        output = getattr(response, "output", None)
        if isinstance(output, list):
            collected: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        collected.append(text)
            if collected:
                return "\n".join(collected)
        raise ValueError("response did not include text output")

    def _build_input(self, prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "You are a JSON-only action proposal model. Return only a JSON object.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def _format_error(self, exc: Exception) -> str:
        parts: list[str] = [repr(exc)]
        message = getattr(exc, "message", None)
        if isinstance(message, str) and message and message not in parts:
            parts.append(message)
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                parts.append(f"status_code={status_code}")
            body = self._response_body_text(response)
            if body:
                parts.append(f"body={body}")
        return " | ".join(parts)

    def _response_body_text(self, response: Any) -> str | None:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                payload = json_method()
            except Exception:
                payload = None
            if payload is not None:
                return str(payload)
        body = getattr(response, "body", None)
        if body is not None:
            return str(body)
        return None
