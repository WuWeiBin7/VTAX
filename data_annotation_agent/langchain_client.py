"""LangChain-backed Qwen client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from data_annotation_agent.config import ModelConfig
from data_annotation_agent.images import PreparedImage
from data_annotation_agent.json_utils import content_to_text, extract_json_object
from data_annotation_agent.prompts import (
    SYSTEM_PROMPT,
    build_detection_context_prompt,
    build_detection_request_prompt,
    build_research_prompt,
)
from data_annotation_agent.schemas import (
    CategoryGuide,
    CategoryGuides,
    DetectionResult,
    TokenUsage,
    normalize_guides,
)


class CompositeAnnotationClient:
    """Delegate research and detection to provider-specific clients."""

    def __init__(self, research_client: Any, detection_client: Any) -> None:
        self._research_client = research_client
        self._detection_client = detection_client

    def research_categories(self, labels: list[str]) -> list[CategoryGuide]:
        """Return category guides from the configured research client."""

        return self._research_client.research_categories(labels)

    def detect_objects(
        self,
        *,
        image_path: Path,
        prepared_image: PreparedImage,
        labels: list[str],
        guides: list[CategoryGuide],
        coordinate_system: str,
    ) -> DetectionResult:
        """Detect objects with the configured vision client."""

        return self._detection_client.detect_objects(
            image_path=image_path,
            prepared_image=prepared_image,
            labels=labels,
            guides=guides,
            coordinate_system=coordinate_system,
        )


class QwenLangChainClient:
    """Call DashScope Qwen models through OpenAI-compatible chat integrations."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def research_categories(self, labels: list[str]) -> list[CategoryGuide]:
        """Research visual and OCR cues for the requested labels."""

        model = self._build_model(
            self._config.research_model,
            extra_body={"enable_search": True},
        )
        agent = self._build_agent(model)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": build_research_prompt(labels)}]}
        )
        payload = self._last_message_json(response)
        return normalize_guides(CategoryGuides.model_validate(payload), labels)

    def detect_objects(
        self,
        *,
        image_path: Path,
        prepared_image: PreparedImage,
        labels: list[str],
        guides: list[CategoryGuide],
        coordinate_system: str,
    ) -> DetectionResult:
        """Detect requested objects in one prepared image."""

        client = self._build_openai_client()
        completion = client.chat.completions.create(
            model=self._config.vision_model,
            messages=self._build_detection_messages(
                image_path=image_path,
                prepared_image=prepared_image,
                labels=labels,
                guides=guides,
                coordinate_system=coordinate_system,
                cache_system_context=True,
            ),
            temperature=0,
        )
        result = DetectionResult.model_validate(
            extract_json_object(content_to_text(self._completion_message_content(completion)))
        )
        result.token_usage = self._extract_token_usage(completion)
        return result

    def _build_model(self, model_name: str, *, extra_body: dict[str, Any] | None = None) -> Any:
        """Create a LangChain ChatOpenAI model for DashScope."""

        self._require_dashscope_api_key()
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            msg = "Install langchain-openai to use QwenLangChainClient."
            raise RuntimeError(msg) from exc

        return ChatOpenAI(
            model=model_name,
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            temperature=0,
            extra_body=extra_body,
        )

    def _build_openai_client(self) -> Any:
        """Create a raw OpenAI-compatible client for requests that need cache-control blocks."""

        self._require_dashscope_api_key()
        return _build_openai_client(
            api_key=self._config.api_key or "",
            base_url=self._config.base_url,
            missing_message="Install openai to use QwenLangChainClient detection requests.",
        )

    def _require_dashscope_api_key(self) -> None:
        """Ensure DashScope credentials are present before a DashScope call."""

        if self._config.api_key:
            return
        msg = (
            "DashScope api_key is required for category research or DashScope detection. "
            "Set it in config.json, DASHSCOPE_API_KEY, or provide pre-generated guides."
        )
        raise RuntimeError(msg)

    @staticmethod
    def _build_agent(model: Any) -> Any:
        """Create a simple LangChain agent graph around a chat model."""

        try:
            from langchain.agents import create_agent
        except ImportError as exc:
            msg = "Install langchain to use QwenLangChainClient."
            raise RuntimeError(msg) from exc

        return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)

    @staticmethod
    def _last_message_json(response: dict[str, Any]) -> dict[str, Any]:
        """Parse JSON from the final LangChain agent message."""

        messages = response.get("messages", [])
        if not messages:
            msg = "LangChain agent returned no messages."
            raise ValueError(msg)
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            content = last_message.content
        else:
            content = last_message.get("content")
        return extract_json_object(content_to_text(content))

    @staticmethod
    def _completion_message_content(completion: Any) -> Any:
        """Return the content of the first chat completion message."""

        choices = getattr(completion, "choices", None)
        if not choices:
            msg = "Chat completion returned no choices."
            raise ValueError(msg)
        return getattr(choices[0].message, "content", None)

    @staticmethod
    def _extract_token_usage(response: Any) -> TokenUsage | None:
        """Extract token usage metadata from either raw OpenAI responses or LangChain responses."""

        return _extract_token_usage(response)

    @staticmethod
    def _build_detection_messages(
        *,
        image_path: Path,
        prepared_image: PreparedImage,
        labels: list[str],
        guides: list[CategoryGuide],
        coordinate_system: str,
        cache_system_context: bool = True,
    ) -> list[dict[str, Any]]:
        """Build detection messages with optional provider-specific cache metadata."""

        return build_detection_messages(
            image_path=image_path,
            prepared_image=prepared_image,
            labels=labels,
            guides=guides,
            coordinate_system=coordinate_system,
            cache_system_context=cache_system_context,
        )


class OllamaVisionClient:
    """Call Ollama's OpenAI-compatible chat completions endpoint for detection."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    def detect_objects(
        self,
        *,
        image_path: Path,
        prepared_image: PreparedImage,
        labels: list[str],
        guides: list[CategoryGuide],
        coordinate_system: str,
    ) -> DetectionResult:
        """Detect requested objects in one prepared image."""

        model = self._config.ollama_vision_model
        if not model:
            msg = "ollama_vision_model is required when provider is 'ollama'."
            raise RuntimeError(msg)
        base_url = self._config.ollama_base_url
        if not base_url:
            msg = "ollama_base_url is required when provider is 'ollama'."
            raise RuntimeError(msg)
        client = _build_openai_client(
            api_key=self._config.ollama_api_key,
            base_url=base_url,
            missing_message="Install openai to use OllamaVisionClient detection requests.",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=build_detection_messages(
                image_path=image_path,
                prepared_image=prepared_image,
                labels=labels,
                guides=guides,
                coordinate_system=coordinate_system,
                cache_system_context=False,
            ),
            temperature=0,
        )
        result = DetectionResult.model_validate(
            extract_json_object(content_to_text(_completion_message_content(completion)))
        )
        result.token_usage = _extract_token_usage(completion)
        return result


def build_annotation_client(config: ModelConfig) -> Any:
    """Build a unified annotation client from the configured provider."""

    dashscope_client = QwenLangChainClient(config)
    if config.provider == "dashscope":
        return dashscope_client
    if config.provider == "ollama":
        return CompositeAnnotationClient(
            research_client=dashscope_client,
            detection_client=OllamaVisionClient(config),
        )
    msg = f"Unsupported provider: {config.provider}"
    raise RuntimeError(msg)


def build_detection_messages(
    *,
    image_path: Path,
    prepared_image: PreparedImage,
    labels: list[str],
    guides: list[CategoryGuide],
    coordinate_system: str,
    cache_system_context: bool,
) -> list[dict[str, Any]]:
    """Build provider-compatible detection messages."""

    shared_context = "\n\n".join(
        [
            SYSTEM_PROMPT,
            build_detection_context_prompt(
                labels=labels,
                guides=guides,
                coordinate_system=coordinate_system,
            ),
        ]
    )
    system_text: dict[str, Any] = {
        "type": "text",
        "text": shared_context,
    }
    if cache_system_context:
        system_text["cache_control"] = {"type": "ephemeral"}
    return [
        {
            "role": "system",
            "content": [system_text],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_detection_request_prompt(
                        image_path=image_path.name,
                        image_width=prepared_image.width,
                        image_height=prepared_image.height,
                        coordinate_system=coordinate_system,
                        image_content_left=prepared_image.pad_left,
                        image_content_top=prepared_image.pad_top,
                        image_content_width=prepared_image.scaled_width,
                        image_content_height=prepared_image.scaled_height,
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": prepared_image.data_url},
                },
            ],
        },
    ]


def _build_openai_client(*, api_key: str, base_url: str, missing_message: str) -> Any:
    """Create a raw OpenAI-compatible client."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(missing_message) from exc

    return OpenAI(api_key=api_key, base_url=base_url)


def _completion_message_content(completion: Any) -> Any:
    """Return the content of the first chat completion message."""

    choices = getattr(completion, "choices", None)
    if not choices:
        msg = "Chat completion returned no choices."
        raise ValueError(msg)
    return getattr(choices[0].message, "content", None)


def _extract_token_usage(response: Any) -> TokenUsage | None:
    """Extract token usage metadata from either raw OpenAI responses or LangChain responses."""

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        return TokenUsage(
            input_tokens=_first_int(
                getattr(usage, "prompt_tokens", None),
                getattr(usage, "input_tokens", None),
            ),
            output_tokens=_first_int(
                getattr(usage, "completion_tokens", None),
                getattr(usage, "output_tokens", None),
            ),
            total_tokens=_first_int(
                getattr(usage, "total_tokens", None),
                getattr(usage, "tokens", None),
            ),
            cached_tokens=_first_int(_usage_detail(prompt_tokens_details, "cached_tokens")),
            cache_creation_input_tokens=_first_int(
                _usage_detail(prompt_tokens_details, "cache_creation_input_tokens"),
                getattr(usage, "cache_creation_input_tokens", None),
            )
        )

    if not isinstance(response, dict):
        return None

    messages = response.get("messages", [])
    if not messages:
        return None

    last_message = messages[-1]
    usage_metadata = getattr(last_message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return TokenUsage(
            input_tokens=_first_int(
                usage_metadata.get("input_tokens"),
                usage_metadata.get("prompt_tokens"),
            ),
            output_tokens=_first_int(
                usage_metadata.get("output_tokens"),
                usage_metadata.get("completion_tokens"),
            ),
            total_tokens=_first_int(
                usage_metadata.get("total_tokens"),
                usage_metadata.get("tokens"),
            ),
            cached_tokens=_first_int(
                usage_metadata.get("cached_tokens"),
                usage_metadata.get("cache_read_input_tokens"),
            ),
            cache_creation_input_tokens=_first_int(
                usage_metadata.get("cache_creation_input_tokens"),
            ),
        )

    response_metadata = getattr(last_message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            return TokenUsage(
                input_tokens=_first_int(
                    token_usage.get("input_tokens"),
                    token_usage.get("prompt_tokens"),
                ),
                output_tokens=_first_int(
                    token_usage.get("output_tokens"),
                    token_usage.get("completion_tokens"),
                ),
                total_tokens=_first_int(
                    token_usage.get("total_tokens"),
                    token_usage.get("tokens"),
                ),
                cached_tokens=_first_int(
                    token_usage.get("cached_tokens"),
                    token_usage.get("cache_read_input_tokens"),
                ),
                cache_creation_input_tokens=_first_int(
                    token_usage.get("cache_creation_input_tokens"),
                ),
            )
    return None


def _first_int(*values: Any) -> int | None:
    """Return the first value that can be converted to an integer."""

    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _usage_detail(usage_details: Any, key: str) -> Any:
    """Read a usage detail value from either an object or a dictionary."""

    if usage_details is None:
        return None
    if isinstance(usage_details, dict):
        return usage_details.get(key)
    return getattr(usage_details, key, None)
