from pathlib import Path
from types import SimpleNamespace

from data_annotation_agent.images import PreparedImage
from data_annotation_agent.langchain_client import QwenLangChainClient, build_detection_messages
from data_annotation_agent.schemas import CategoryGuide


class Message:
    content = '{"shapes": []}'
    usage_metadata = None
    response_metadata = None


class UsageMessage:
    content = '{"shapes": []}'
    usage_metadata = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    response_metadata = None


class ResponseMetadataMessage:
    content = '{"shapes": []}'
    usage_metadata = None
    response_metadata = {
        "token_usage": {
            "prompt_tokens": 13,
            "completion_tokens": 5,
            "total_tokens": 18,
            "cached_tokens": 9,
            "cache_creation_input_tokens": 4,
        }
    }


def test_last_message_json_supports_langchain_message_objects() -> None:
    payload = QwenLangChainClient._last_message_json({"messages": [Message()]})

    assert payload == {"shapes": []}


def test_last_message_json_supports_dict_messages() -> None:
    payload = QwenLangChainClient._last_message_json({"messages": [{"content": '{"guides": []}'}]})

    assert payload == {"guides": []}


def test_extract_token_usage_supports_usage_metadata() -> None:
    usage = QwenLangChainClient._extract_token_usage({"messages": [UsageMessage()]})

    assert usage is not None
    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.total_tokens == 18


def test_extract_token_usage_supports_response_metadata() -> None:
    usage = QwenLangChainClient._extract_token_usage({"messages": [ResponseMetadataMessage()]})

    assert usage is not None
    assert usage.input_tokens == 13
    assert usage.output_tokens == 5
    assert usage.total_tokens == 18
    assert usage.cached_tokens == 9
    assert usage.cache_creation_input_tokens == 4


def test_extract_token_usage_supports_openai_usage_objects() -> None:
    usage = QwenLangChainClient._extract_token_usage(
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=21,
                completion_tokens=8,
                total_tokens=29,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=14,
                    cache_creation_input_tokens=7,
                ),
            )
        )
    )

    assert usage is not None
    assert usage.input_tokens == 21
    assert usage.output_tokens == 8
    assert usage.total_tokens == 29
    assert usage.cached_tokens == 14
    assert usage.cache_creation_input_tokens == 7


def test_build_detection_messages_uses_cached_system_context() -> None:
    prepared_image = PreparedImage(
        data_url="data:image/jpeg;base64,abc",
        width=1024,
        height=1024,
        original_width=640,
        original_height=480,
        canvas_width=1024,
        canvas_height=1024,
        scaled_width=1024,
        scaled_height=768,
        scale=1.6,
        pad_left=0,
        pad_top=128,
    )

    messages = QwenLangChainClient._build_detection_messages(
        image_path=Path("sample.jpg"),
        prepared_image=prepared_image,
        labels=["poppy"],
        guides=[CategoryGuide(label="poppy", visual_cues=["red petals"])],
        coordinate_system="pixel",
    )

    assert messages[0]["role"] == "system"
    system_content = messages[0]["content"][0]
    assert system_content["type"] == "text"
    assert system_content["cache_control"] == {"type": "ephemeral"}
    assert "poppy" in system_content["text"]

    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "text"
    assert "sample.jpg" in messages[1]["content"][0]["text"]
    assert "left=0" in messages[1]["content"][0]["text"]
    assert "top=128" in messages[1]["content"][0]["text"]
    assert "right=1024" in messages[1]["content"][0]["text"]
    assert "bottom=896" in messages[1]["content"][0]["text"]
    assert messages[1]["content"][1]["type"] == "image_url"
    assert messages[1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,abc"


def test_build_detection_messages_can_disable_cached_system_context() -> None:
    prepared_image = PreparedImage(
        data_url="data:image/jpeg;base64,abc",
        width=1024,
        height=1024,
        original_width=640,
        original_height=480,
        canvas_width=1024,
        canvas_height=1024,
        scaled_width=1024,
        scaled_height=768,
        scale=1.6,
        pad_left=0,
        pad_top=128,
    )

    messages = build_detection_messages(
        image_path=Path("sample.jpg"),
        prepared_image=prepared_image,
        labels=["poppy"],
        guides=[CategoryGuide(label="poppy", visual_cues=["red petals"])],
        coordinate_system="pixel",
        cache_system_context=False,
    )

    assert "cache_control" not in messages[0]["content"][0]


def test_extract_token_usage_returns_none_without_usage() -> None:
    usage = QwenLangChainClient._extract_token_usage(SimpleNamespace())

    assert usage is None
