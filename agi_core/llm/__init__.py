from __future__ import annotations

import os

from agi_core.config import RuntimeConfig

from .llama_cpp_backend import LlamaCppBackend
from .openai_responses_backend import OpenAIResponsesBackend


def build_llm_backend(config: RuntimeConfig):
    if config.llm_provider == "local":
        return LlamaCppBackend(config.effective_model_path())
    if config.llm_provider == "openai":
        return OpenAIResponsesBackend(
            model=config.openai_model,
            api_key=os.environ.get(config.openai_api_key_env),
            base_url=config.openai_base_url,
            timeout_seconds=config.openai_timeout_seconds,
        )
    raise ValueError(f"unsupported llm provider: {config.llm_provider}")


__all__ = ["LlamaCppBackend", "OpenAIResponsesBackend", "build_llm_backend"]
