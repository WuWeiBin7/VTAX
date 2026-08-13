"""Runtime configuration for model providers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_RESEARCH_MODEL = "qwen3.7-plus"
DEFAULT_VISION_MODEL = "qwen3.7-plus"
DEFAULT_PROVIDER = "dashscope"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama"
DEFAULT_CONFIG_PATH = Path("config.json")
SUPPORTED_PROVIDERS = {"dashscope", "ollama"}


@dataclass(frozen=True)
class ModelConfig:
    """Configuration needed to call supported OpenAI-compatible providers."""

    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    research_model: str = DEFAULT_RESEARCH_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    provider: str = DEFAULT_PROVIDER
    ollama_base_url: str | None = None
    ollama_vision_model: str | None = None
    ollama_api_key: str = DEFAULT_OLLAMA_API_KEY

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: Path | None = DEFAULT_CONFIG_PATH,
        api_key: str | None = None,
        base_url: str | None = None,
        research_model: str | None = None,
        vision_model: str | None = None,
        provider: str | None = None,
        require_research_credentials: bool = True,
    ) -> "ModelConfig":
        """Build model configuration from CLI values, a config file, and env vars."""

        file_config = load_config_file(config_path)
        resolved_provider = _normalize_provider(
            provider
            or _clean_string(file_config.get("provider"))
            or os.getenv("ANNOTATION_PROVIDER")
            or DEFAULT_PROVIDER
        )
        resolved_api_key = (
            api_key
            or _clean_string(file_config.get("api_key"))
            or os.getenv("DASHSCOPE_API_KEY")
        )
        if require_research_credentials and not resolved_api_key:
            location = str(config_path or DEFAULT_CONFIG_PATH)
            msg = (
                "DashScope api_key is required for category research. "
                f"Set it in {location}, DASHSCOPE_API_KEY, or provide pre-generated guides."
            )
            raise RuntimeError(msg)

        resolved_ollama_base_url = (
            _clean_string(file_config.get("ollama_base_url"))
            or os.getenv("OLLAMA_BASE_URL")
        )
        resolved_ollama_vision_model = (
            _clean_string(file_config.get("ollama_vision_model"))
            or os.getenv("OLLAMA_VISION_MODEL")
        )
        resolved_ollama_api_key = (
            _clean_string(file_config.get("ollama_api_key"))
            or os.getenv("OLLAMA_API_KEY")
            or DEFAULT_OLLAMA_API_KEY
        )
        if resolved_provider == "ollama":
            if not resolved_ollama_base_url:
                msg = "ollama_base_url is required when provider is 'ollama'."
                raise RuntimeError(msg)
            if not resolved_ollama_vision_model:
                msg = "ollama_vision_model is required when provider is 'ollama'."
                raise RuntimeError(msg)

        return cls(
            provider=resolved_provider,
            api_key=resolved_api_key,
            base_url=(
                base_url
                or _clean_string(file_config.get("base_url"))
                or os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
            ),
            research_model=(
                research_model
                or _clean_string(file_config.get("research_model"))
                or DEFAULT_RESEARCH_MODEL
            ),
            vision_model=(
                vision_model
                or _clean_string(file_config.get("vision_model"))
                or DEFAULT_VISION_MODEL
            ),
            ollama_base_url=resolved_ollama_base_url,
            ollama_vision_model=resolved_ollama_vision_model,
            ollama_api_key=resolved_ollama_api_key,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "ModelConfig":
        """Backward-compatible alias for older callers."""

        return cls.from_sources(config_path=None, **kwargs)


def load_config_file(config_path: Path | None) -> dict[str, Any]:
    """Load a JSON configuration file if it exists."""

    if config_path is None or not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON config file: {config_path}"
        raise RuntimeError(msg) from exc

    if not isinstance(payload, dict):
        msg = f"Config file must contain a JSON object: {config_path}"
        raise RuntimeError(msg)
    return payload


def _clean_string(value: Any) -> str | None:
    """Return a stripped string or `None` for empty/non-string values."""

    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_provider(value: str) -> str:
    """Normalize and validate a provider name."""

    provider = value.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        msg = f"Unsupported provider: {value}. Expected one of: dashscope, ollama."
        raise RuntimeError(msg)
    return provider
