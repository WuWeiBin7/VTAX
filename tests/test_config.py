from pathlib import Path

import pytest

from data_annotation_agent.config import ModelConfig


def test_model_config_loads_from_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "api_key": "file-key",
          "base_url": "https://example.com/v1",
          "research_model": "research-model",
          "vision_model": "vision-model"
        }
        """,
        encoding="utf-8",
    )

    config = ModelConfig.from_sources(config_path=config_path)

    assert config.api_key == "file-key"
    assert config.base_url == "https://example.com/v1"
    assert config.research_model == "research-model"
    assert config.vision_model == "vision-model"
    assert config.provider == "dashscope"


def test_model_config_cli_values_override_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "api_key": "file-key",
          "base_url": "https://example.com/v1",
          "research_model": "research-model",
          "vision_model": "vision-model"
        }
        """,
        encoding="utf-8",
    )

    config = ModelConfig.from_sources(
        config_path=config_path,
        base_url="https://override.example/v1",
        vision_model="override-vision",
    )

    assert config.api_key == "file-key"
    assert config.base_url == "https://override.example/v1"
    assert config.research_model == "research-model"
    assert config.vision_model == "override-vision"


def test_model_config_requires_api_key_without_config_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="api_key is required"):
        ModelConfig.from_sources(config_path=None)


def test_model_config_supports_ollama_provider_without_research_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "provider": "ollama",
          "ollama_base_url": "http://127.0.0.1:11434/v1",
          "ollama_vision_model": "qwen3-vl:4b"
        }
        """,
        encoding="utf-8",
    )

    config = ModelConfig.from_sources(
        config_path=config_path,
        require_research_credentials=False,
    )

    assert config.provider == "ollama"
    assert config.api_key is None
    assert config.ollama_base_url == "http://127.0.0.1:11434/v1"
    assert config.ollama_vision_model == "qwen3-vl:4b"
    assert config.ollama_api_key == "ollama"


def test_model_config_provider_override_wins_over_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "provider": "dashscope",
          "api_key": "file-key",
          "ollama_base_url": "http://127.0.0.1:11434/v1",
          "ollama_vision_model": "qwen3-vl:4b"
        }
        """,
        encoding="utf-8",
    )

    config = ModelConfig.from_sources(config_path=config_path, provider="ollama")

    assert config.provider == "ollama"


def test_model_config_requires_ollama_model_for_ollama_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "provider": "ollama",
          "api_key": "file-key",
          "ollama_base_url": "http://127.0.0.1:11434/v1"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ollama_vision_model"):
        ModelConfig.from_sources(config_path=config_path)


def test_model_config_requires_ollama_base_url_for_ollama_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "provider": "ollama",
          "api_key": "file-key",
          "ollama_vision_model": "qwen3-vl:4b"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ollama_base_url"):
        ModelConfig.from_sources(config_path=config_path)
