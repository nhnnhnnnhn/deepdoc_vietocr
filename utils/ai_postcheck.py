import base64
import io
import json
import logging
import os
from typing import Any

import requests

from utils.runtime_env import env_bool


def get_gemini_api_key(args) -> str | None:
    api_key = args.gemini_api_key or os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    return None


def load_gemini_system_prompt(prompt_file: str | None, default_prompt_file: str) -> str:
    prompt_file = prompt_file or default_prompt_file
    if not os.path.isabs(prompt_file):
        prompt_file = os.path.join(os.getcwd(), prompt_file)

    with open(prompt_file, encoding="utf-8") as handle:
        return handle.read().strip()


def compact_metadata_for_gemini(metadata: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "input": metadata.get("input"),
        "output_markdown": metadata.get("output_markdown"),
        "page_count": metadata.get("page_count"),
        "region_count": metadata.get("region_count"),
        "asset_count": metadata.get("asset_count"),
        "ocr_confidence": metadata.get("ocr_confidence"),
        "layout_confidence": metadata.get("layout_confidence"),
        "gemini_postcheck_decision": metadata.get("gemini_postcheck_decision"),
        "pages": [],
    }

    for page in metadata.get("pages", []):
        compact_page = {
            "page_number": page.get("page_number"),
            "width": page.get("width"),
            "height": page.get("height"),
            "ocr_confidence": page.get("ocr_confidence"),
            "layout_confidence": page.get("layout_confidence"),
            "vlm_risks": page.get("vlm_risks", []),
            "regions": [],
        }
        for region in page.get("regions", []):
            compact_region = {
                "content_type": region.get("content_type"),
                "layout_type": region.get("layout_type"),
                "bbox": region.get("bbox"),
                "ocr_confidence": region.get("ocr_confidence"),
            }
            if region.get("asset_path"):
                compact_region["asset_path"] = region.get("asset_path")
            compact_page["regions"].append(compact_region)
        compact["pages"].append(compact_page)

    return compact


def gemini_postcheck_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "checked_markdown": {
                "type": "string",
                "description": "The corrected final Markdown. Preserve useful image links and table structure.",
            },
            "summary": {
                "type": "string",
                "description": "Short Vietnamese summary of the post-check result.",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "page_number": {"type": ["integer", "null"]},
                        "issue": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                    "required": ["severity", "page_number", "issue", "suggestion"],
                },
            },
            "image_notes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_number": {"type": ["integer", "null"]},
                        "asset_path": {"type": ["string", "null"]},
                        "description": {"type": "string"},
                    },
                    "required": ["page_number", "asset_path", "description"],
                },
            },
            "vlm_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "risk_id": {"type": ["string", "null"]},
                        "page_number": {"type": ["integer", "null"]},
                        "category": {"type": "string"},
                        "finding": {"type": "string"},
                        "correction": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                    },
                    "required": ["risk_id", "page_number", "category", "finding", "correction", "confidence"],
                },
            },
            "confidence": {
                "type": "number",
                "description": "Confidence from 0 to 1 that the checked Markdown is faithful to the document.",
            },
        },
        "required": ["checked_markdown", "summary", "issues", "image_notes", "vlm_findings", "confidence"],
    }


def format_token_usage(token_usage: dict[str, Any] | None) -> str:
    if not token_usage:
        return "unavailable"

    fields = [
        ("prompt", "prompt_token_count"),
        ("cached", "cached_content_token_count"),
        ("output", "candidates_token_count"),
        ("thinking", "thoughts_token_count"),
        ("tool", "tool_use_prompt_token_count"),
        ("total", "total_token_count"),
    ]
    return ", ".join(f"{label}={token_usage[key]}" for label, key in fields if key in token_usage) or "unavailable"


def parse_ai_json_response(content) -> dict[str, Any]:
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in (None, "text"))
    else:
        text = str(content or "")

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first : last + 1]
    return json.loads(text)


def openai_compatible_base_url(args) -> str:
    base_url = (
        getattr(args, "ai_base_url", None)
        or os.environ.get("AI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    return base_url.rstrip("/")


def openai_compatible_chat_url(args) -> str:
    base_url = openai_compatible_base_url(args)
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def openai_compatible_token_usage(response_json: dict[str, Any]) -> dict[str, Any]:
    usage = response_json.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    token_usage = {
        "prompt_token_count": usage.get("prompt_tokens"),
        "candidates_token_count": usage.get("completion_tokens"),
        "thoughts_token_count": completion_details.get("reasoning_tokens"),
        "total_token_count": usage.get("total_tokens"),
    }
    return {key: value for key, value in token_usage.items() if value not in (None, [], {})}


def ai_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def ai_env_int(name: str, default=None):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def image_to_openai_data_url(image, max_side: int) -> str:
    img = image.copy()
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def openai_response_format(args):
    mode = (
        getattr(args, "ai_response_format", None)
        or os.environ.get("AI_RESPONSE_FORMAT")
        or os.environ.get("OPENAI_RESPONSE_FORMAT")
        or "json_object"
    ).strip().lower()

    if mode in ("", "none", "disabled", "off"):
        return None
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ocr_markdown_postcheck",
                "schema": gemini_postcheck_schema(),
                "strict": False,
            },
        }
    return {"type": "json_object"}


def run_gemini_postcheck(markdown_text, metadata, page_images, args, api_key: str, default_prompt_file: str):
    system_prompt = load_gemini_system_prompt(args.gemini_system_prompt, default_prompt_file)
    schema = gemini_postcheck_schema()
    compact_metadata = compact_metadata_for_gemini(metadata)
    max_input_chars = ai_env_int("AI_MAX_INPUT_CHARS", 40000)
    if max_input_chars and len(markdown_text) > max_input_chars:
        logging.warning(
            f"Markdown input truncated from {len(markdown_text)} to {max_input_chars} chars (set AI_MAX_INPUT_CHARS to change)"
        )
        markdown_text = markdown_text[:max_input_chars] + "\n\n[... truncated ...]"
    user_text = (
        "Review and correct the OCR Markdown. Return JSON only, following this JSON schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Source Markdown:\n"
        f"{markdown_text}\n\n"
        "Compact metadata:\n"
        f"{json.dumps(compact_metadata, ensure_ascii=False)}\n\n"
        "Detected VLM risks:\n"
        f"{json.dumps(metadata.get('vlm_risks', []), ensure_ascii=False)}"
    )

    content = [{"type": "text", "text": user_text}]
    include_images = env_bool("AI_INCLUDE_IMAGES", True)
    if include_images:
        max_pages = int(getattr(args, "gemini_max_pages", 6) or 6)
        max_side = int(getattr(args, "gemini_image_max_side", 1600) or 1600)
        for image in page_images[:max_pages]:
            content.append({"type": "image_url", "image_url": {"url": image_to_openai_data_url(image, max_side)}})

    payload = {
        "model": args.gemini_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": ai_env_float("AI_TEMPERATURE", 0.0),
    }
    max_tokens = ai_env_int("AI_MAX_TOKENS", None)
    if max_tokens:
        payload["max_tokens"] = max_tokens

    response_format = openai_response_format(args)
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    chat_url = openai_compatible_chat_url(args)
    try:
        response = requests.post(
            chat_url,
            headers=headers,
            json=payload,
            timeout=float(args.gemini_timeout),
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"AI API request timed out: {chat_url}. Check AI_BASE_URL and whether the provider is reachable."
        ) from exc
    except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
        raise RuntimeError(
            f"Could not connect to AI API: {chat_url}. Check AI_BASE_URL and whether the provider is reachable."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        body = response.text[:1000] if response is not None else ""
        raise RuntimeError(f"AI API returned HTTP {response.status_code} from {chat_url}: {body}") from exc
    response_json = response.json()

    choices = response_json.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compatible response has no choices: {response_json}")
    message = choices[0].get("message") or {}
    content_text = message.get("content", "")
    try:
        result = parse_ai_json_response(content_text)
    except json.JSONDecodeError as exc:
        snippet = str(content_text)[:1000]
        raise RuntimeError(f"OpenAI-compatible response was not valid JSON: {snippet}") from exc

    if "checked_markdown" not in result:
        raise RuntimeError("OpenAI-compatible response JSON is missing checked_markdown")

    result["_token_usage"] = openai_compatible_token_usage(response_json)
    result["_raw_usage_metadata"] = response_json.get("usage", {})
    return result
