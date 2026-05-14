from pathlib import Path
from typing import Any


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        return "upload"
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "._- ()").strip()
    return safe or "upload"


def read_text_preview(path: Path, limit: int) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        content = handle.read(limit)
    return content.decode("utf-8", errors="replace"), size > limit


def summarize_postcheck(metadata: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    postcheck = metadata.get("ai_postcheck") or metadata.get("gemini_postcheck")
    if isinstance(postcheck, dict):
        status = str(postcheck.get("status") or "unknown")
        token_usage = postcheck.get("token_usage")
        return status, token_usage if isinstance(token_usage, dict) and token_usage else None
    if metadata:
        return "disabled", None
    return "unavailable", None
