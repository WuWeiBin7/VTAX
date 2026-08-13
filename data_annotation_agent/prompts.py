"""Prompts used by the annotation agent."""

from __future__ import annotations

import json

from data_annotation_agent.schemas import CategoryGuide

SYSTEM_PROMPT = """你是一个图像识别准确、类别知识扎实的数据标注 agent，专门为目标检测数据集生成矩形框标注 JSON。
你的任务边界：
- 只识别用户指定的类别，不新增类别，不推测无关目标。
- 只输出二维矩形框标注，不输出分割、多边形、关键点或额外解释文本。
- 若目标不可见、证据不足、被严重遮挡到无法可靠定位，返回空 shapes。
- 对敏感类别仅做图像识别与数据标注，不提供获取、制造、使用、交易或规避监管相关建议。

输出 JSON 结构：
{
  "shapes": [
    {
      "label": "<用户类别>",
      "score": <0到1之间的置信度>,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
"""


def build_research_prompt(labels: list[str]) -> str:
    """Build the category research prompt."""

    return f"""请联网研究这些目标检测类别，并为每个类别生成视觉识别指南：{json.dumps(labels, ensure_ascii=False)}

要求：
- 必须为每个输入类别生成一条指南，label 必须与输入类别完全一致。
- 所有说明内容请优先使用简体中文；只有专有名词、英文标识、OCR 关键词可以保留英文。
- visual_cues 描述可见外观、形态、颜色、部位、典型场景、可用于画框定位的边界特征。
- ocr_cues 描述图像中可能出现的文字、包装、标识、招牌、告示、上下文文本或英文关键词。
- negative_cues 描述容易误检的相似对象、非目标情况和排除规则。
- ambiguity_notes 描述标注边界、遮挡、截断、低置信度和不确定情况的处理方式。
- 每个字段必须是字符串数组，数组元素必须是可读文本。
- 只输出 JSON，不要输出 Markdown 或解释。

输出格式：
{{
  "guides": [
    {{
      "label": "<输入类别>",
      "visual_cues": ["..."],
      "ocr_cues": ["..."],
      "negative_cues": ["..."],
      "ambiguity_notes": ["..."]
    }}
  ]
}}
"""


def build_detection_context_prompt(
    *,
    labels: list[str],
    guides: list[CategoryGuide],
    coordinate_system: str,
) -> str:
    """Build the shared detection instructions cached across images."""

    guide_payload = [guide.model_dump(mode="json") for guide in guides]
    return f"""请对后续输入的图像进行目标检测矩形框标注。
允许的类别：{json.dumps(labels, ensure_ascii=False)}
类别识别指南：{json.dumps(guide_payload, ensure_ascii=False, indent=2)}

标注要求：
- 只标注允许类别中的可见实例。
- {_coordinate_system_instruction(coordinate_system)}
- bbox 格式为 [x1, y1, x2, y2]，代表目标外接矩形左上角与右下角。
- 矩形框应尽量紧贴目标主体，不要包含大量背景。
- 若某类目标不存在或无法可靠定位，不要为该类别输出框。
- score 是 0 到 1 之间的小数。
- 只输出 JSON，不要输出 Markdown、注释或解释。

输出格式：
{{
  "shapes": [
    {{"label": "<允许类别之一>", "score": 0.0, "bbox": [0, 0, 10, 10]}}
  ]
}}
"""


def build_detection_request_prompt(
    *,
    image_path: str,
    image_width: int,
    image_height: int,
    coordinate_system: str,
    image_content_left: int = 0,
    image_content_top: int = 0,
    image_content_width: int | None = None,
    image_content_height: int | None = None,
) -> str:
    """Build the per-image request appended after the cached context."""

    lines = [
        f"当前图片文件名：{image_path}",
        f"当前输入图像尺寸：width={image_width}, height={image_height}",
    ]
    if (
        coordinate_system in {"pixel", "normalized-1000"}
        and image_content_width is not None
        and image_content_height is not None
    ):
        right = image_content_left + image_content_width
        bottom = image_content_top + image_content_height
        lines.extend(
            [
                "这不是原图，而是我们预处理后的固定方形画布。",
                f"画布尺寸固定为：{image_width} x {image_height}",
                (
                    "原图实际内容区域为："
                    f"left={image_content_left}, top={image_content_top}, "
                    f"right={right}, bottom={bottom}"
                ),
                "其余区域是补边 padding，不能把 padding 当作目标主体区域。",
            ]
        )
    lines.append(_coordinate_system_request_hint(coordinate_system))
    lines.append("请基于以上规则，只返回本张图的 JSON 检测结果。")
    return "\n".join(lines)


def _coordinate_system_instruction(coordinate_system: str) -> str:
    """Return the shared coordinate-system instructions for detection."""

    if coordinate_system == "normalized-1000":
        return (
            "坐标使用 0-1000 坐标系，不要输出原始像素坐标。"
            "0 表示图像最左/最上边界，1000 表示图像最右/最下边界。"
        )
    if coordinate_system == "pixel":
        return (
            "坐标必须使用当前输入方形画布的像素坐标，不要输出归一化坐标、0-1000 坐标，"
            "也不要输出原图坐标。"
        )
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)


def _coordinate_system_request_hint(coordinate_system: str) -> str:
    """Return the per-image coordinate hint that depends on current image size."""

    if coordinate_system == "normalized-1000":
        return "请注意：当前图像尺寸仅供定位参考，输出坐标仍需使用 0-1000 坐标系。"
    if coordinate_system == "pixel":
        return "请注意：输出 bbox 必须严格落在当前方形画布像素范围内。"
    msg = f"Unsupported coordinate system: {coordinate_system}"
    raise ValueError(msg)
