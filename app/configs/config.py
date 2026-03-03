
import json
import os
from pathlib import Path
from typing import Dict, Literal, Optional, List

from pydantic import BaseModel, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[SecretStr] = None

    @model_validator(mode="after")
    def resolve_api_key(self) -> "ProviderConfig":
        if self.api_key_env and not self.api_key:
            value = os.environ.get(self.api_key_env)
            if value:
                self.api_key = SecretStr(value)
        return self


class LLMParams(BaseModel):
    temperature: float
    max_tokens: int
    streaming: bool


class LLMModelConfig(BaseModel):
    role: str
    provider: str
    model_name: str
    params: LLMParams


class LLMsConfig(BaseModel):
    default: LLMModelConfig
    providers: Dict[str, ProviderConfig]
    models: List[LLMModelConfig]


class Settings(BaseSettings):
    llms: LLMsConfig
    max_retries: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def load_from_json(cls, json_path: str = str(Path(__file__).parent / "config.json")) -> "Settings":
        """
        Đọc cấu hình từ file JSON, sau đó pydantic-settings tự động
        nạp biến môi trường từ .env để bổ sung các giá trị còn thiếu
        (ví dụ: api_key được resolve qua api_key_env trong từng model).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        return cls(**json_data)
