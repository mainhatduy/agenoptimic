from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI

from app.configs.config import LLMModelConfig, Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load_from_json()


def get_llm(role: Optional[str] = None) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance configured for the given role.
    Falls back to the default LLM if the role is not found in models list.
    """
    settings = get_settings()

    model_cfg: Optional[LLMModelConfig] = None

    if role:
        for m in settings.llms.models:
            if m.role == role:
                model_cfg = m
                break

    if model_cfg is None:
        model_cfg = settings.llms.default

    provider_cfg = settings.llms.providers[model_cfg.provider]
    api_key = provider_cfg.api_key.get_secret_value() if provider_cfg.api_key else None

    return ChatOpenAI(
        model=model_cfg.model_name,
        base_url=provider_cfg.base_url,
        api_key=api_key,
        temperature=model_cfg.params.temperature,
        max_tokens=model_cfg.params.max_tokens,
        streaming=model_cfg.params.streaming,
    )
