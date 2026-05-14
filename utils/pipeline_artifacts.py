import hashlib
import json
import os
from typing import Any


def _document_output_base(source_path: str) -> tuple[str, str]:
    output_folder = os.path.dirname(source_path)
    document_name = os.path.basename(output_folder)
    return output_folder, document_name


def build_document_artifact_path(source_path: str, suffix: str) -> str:
    output_folder, document_name = _document_output_base(source_path)
    return os.path.join(output_folder, f"{document_name}_{suffix}")


def merged_markdown_output_path(page_output_path: str) -> str:
    return build_document_artifact_path(page_output_path, "full.md")


def metadata_output_path(markdown_output_path: str) -> str:
    return build_document_artifact_path(markdown_output_path, "metadata.json")


def ocr_raw_output_path(markdown_output_path: str) -> str:
    return build_document_artifact_path(markdown_output_path, "ocr_raw.md")


def vlm_checked_output_path(markdown_output_path: str) -> str:
    return build_document_artifact_path(markdown_output_path, "vlm_checked.md")


def vlm_review_output_path(markdown_output_path: str) -> str:
    return build_document_artifact_path(markdown_output_path, "vlm_review.json")


def relative_output_path(path: str, base_dir: str) -> str:
    return os.path.relpath(path, base_dir).replace(os.sep, "/")


def default_correction_memory(version: int) -> dict[str, Any]:
    return {"version": version, "rules": [], "blocked": []}


def load_correction_memory(path: str, version: int) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return default_correction_memory(version)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default_correction_memory(version)
    if not isinstance(payload, dict):
        return default_correction_memory(version)
    payload.setdefault("version", version)
    payload.setdefault("rules", [])
    payload.setdefault("blocked", [])
    if not isinstance(payload["rules"], list):
        payload["rules"] = []
    if not isinstance(payload["blocked"], list):
        payload["blocked"] = []
    return payload


def save_correction_memory(path: str, memory: dict[str, Any]) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(memory, handle, ensure_ascii=False, indent=2)


def correction_rule_id(wrong: str, correct: str) -> str:
    digest = hashlib.sha256(f"{wrong}\0{correct}".encode("utf-8")).hexdigest()
    return digest[:16]
