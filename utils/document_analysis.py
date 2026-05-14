import re
from typing import Any


SENSITIVE_TEXT_RE = re.compile(
    r"("
    r"\b\d{1,3}(?:[.,]\d{3})+(?:\s*(?:đồng|vnd|vnđ|usd))?\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{9,12}\b"
    r"|\b(?:số|so|mã|ma|mst|cccd|cmnd|hợp\s*đồng|hop\s*dong|đơn|don)\b"
    r")",
    re.IGNORECASE,
)
AMBIGUOUS_TOKEN_RE = re.compile(
    r"\b(?=[A-Za-z0-9?]{3,}\b)(?=[A-Za-z?]*[OolIlS?])(?=[A-Za-z0-9?]*\d)[A-Za-z0-9?/-]+\b"
)
CHECKBOX_HINT_RE = re.compile(r"(\[[ xX]\]|☐|☑|✓|□|■|checkbox|tick|chọn|chon)", re.IGNORECASE)
DOCUMENT_TITLE_RE = re.compile(
    r"^(quyết\s*định|thông\s*báo|hợp\s*đồng|tờ\s*trình|báo\s*cáo|biên\s*bản|công\s*văn)\b",
    re.IGNORECASE,
)
CHAPTER_RE = re.compile(r"^(chương|phần|mục)\s+([ivxlcdm]+|\d+)\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^điều\s+\d+[.,:]?\s+.+", re.IGNORECASE)
NUMBERED_SECTION_RE = re.compile(r"^(\d{1,2})\.\s+\S+")
NUMBERED_SUBSECTION_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.?\s*\S+")
DENSE_SUBSECTION_RE = re.compile(r"^(\d{2})\.?\s*\D+")
RECEIVER_RE = re.compile(r"^nơi\s+nhận\b", re.IGNORECASE)


def page_has_multicolumn_layout(page_metadata: dict[str, Any]) -> bool:
    width = page_metadata["width"]
    height = page_metadata["height"]
    content_regions = [
        region for region in page_metadata["layout_regions"] if region["type"] in {"text", "title", "table"} and region["score"] >= 0.15
    ]
    left = []
    right = []
    for region in content_regions:
        x0, y0, x1, y1 = region["bbox"]
        region_width = x1 - x0
        if region_width > width * 0.68 or (y1 - y0) < height * 0.04:
            continue
        center = (x0 + x1) / 2
        if center < width * 0.48:
            left.append(region)
        elif center > width * 0.52:
            right.append(region)

    if len(left) < 2 or len(right) < 2:
        return False

    left_span = max(r["bbox"][3] for r in left) - min(r["bbox"][1] for r in left)
    right_span = max(r["bbox"][3] for r in right) - min(r["bbox"][1] for r in right)
    return left_span > height * 0.25 and right_span > height * 0.25


def _crop_area(bbox: list[int] | tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def text_density_is_low(text: str, bbox: list[int] | tuple[int, int, int, int], threshold: float) -> bool:
    area = _crop_area(bbox)
    if area < 120000:
        return False
    density = len((text or "").strip()) / max(area, 1)
    return density < threshold


def add_vlm_risk(
    page_metadata: dict[str, Any],
    category: str,
    reason: str,
    bbox: list[int] | tuple[int, int, int, int] | None = None,
    text: str | None = None,
    asset_path: str | None = None,
    severity: str = "medium",
) -> dict[str, Any]:
    risk = {
        "risk_id": f"p{page_metadata['page_number']:03d}_r{len(page_metadata['vlm_risks']) + 1:03d}",
        "page_number": page_metadata["page_number"],
        "category": category,
        "severity": severity,
        "reason": reason,
    }
    if bbox is not None:
        risk["bbox"] = list(bbox)
    if text:
        risk["text"] = text[:1000]
    if asset_path:
        risk["asset_path"] = asset_path
    page_metadata["vlm_risks"].append(risk)
    return risk


def add_text_risks(
    page_metadata: dict[str, Any],
    text: str,
    bbox: list[int] | tuple[int, int, int, int],
    low_density_threshold: float,
) -> None:
    if not text:
        return

    if SENSITIVE_TEXT_RE.search(text):
        add_vlm_risk(
            page_metadata,
            "sensitive_value",
            "Line/block contains money, date, ID, contract/order code, or other exact-value fields.",
            bbox=bbox,
            text=text,
            severity="high",
        )

    if AMBIGUOUS_TOKEN_RE.search(text):
        add_vlm_risk(
            page_metadata,
            "ambiguous_characters",
            "Text contains OCR-ambiguous O/0, l/1, S/5-like tokens.",
            bbox=bbox,
            text=text,
            severity="high",
        )

    if CHECKBOX_HINT_RE.search(text):
        add_vlm_risk(
            page_metadata,
            "checkbox_or_selection",
            "Text contains checkbox/selection hints that need visual verification.",
            bbox=bbox,
            text=text,
            severity="medium",
        )

    if text_density_is_low(text, bbox, low_density_threshold):
        add_vlm_risk(
            page_metadata,
            "ocr_too_short_for_region",
            "OCR text is short compared with the visual area of this region.",
            bbox=bbox,
            text=text,
            severity="medium",
        )


def is_markdown_passthrough_line(text: str) -> bool:
    stripped = text.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith("![")
        or stripped.startswith("```")
        or stripped.startswith("- ")
        or stripped.startswith("+ ")
        or stripped.startswith("* ")
    )


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def is_short_heading_candidate(text: str, max_chars: int) -> bool:
    stripped = text.strip()
    return 3 <= len(stripped) <= max_chars and has_letters(stripped)


def normalize_dense_subsection(text: str, state: dict[str, Any]) -> str:
    parent_number = state.get("last_numbered_section")
    if not parent_number or not text.startswith(parent_number):
        return text

    child_index = len(parent_number)
    if child_index >= len(text) or not text[child_index].isdigit():
        return text

    child_number = text[child_index]
    rest = text[child_index + 1 :].lstrip(". ")
    if not rest:
        return text
    return f"{parent_number}.{child_number}. {rest}"


def classify_heading_line(text: str, state: dict[str, Any], max_chars: int):
    stripped = text.strip()
    if is_markdown_passthrough_line(stripped) or not is_short_heading_candidate(stripped, max_chars):
        return None

    if DOCUMENT_TITLE_RE.match(stripped):
        if not state.get("document_title_seen"):
            state["document_title_seen"] = True
            state["last_numbered_section"] = None
            return 1, "document_title", stripped
        state["last_numbered_section"] = None
        return 2, "document_section_title", stripped

    if ARTICLE_RE.match(stripped):
        state["last_numbered_section"] = None
        return 2, "article", stripped

    if RECEIVER_RE.match(stripped):
        state["last_numbered_section"] = None
        return 2, "receiver_section", stripped

    chapter_match = CHAPTER_RE.match(stripped)
    if chapter_match:
        keyword = chapter_match.group(1).lower()
        state["last_numbered_section"] = None
        return (3 if keyword == "mục" else 2), keyword, stripped

    subsection_match = NUMBERED_SUBSECTION_RE.match(stripped)
    if subsection_match:
        state["last_numbered_section"] = subsection_match.group(1)
        return 4, "numbered_subsection", stripped

    section_match = NUMBERED_SECTION_RE.match(stripped)
    if section_match:
        state["last_numbered_section"] = section_match.group(1)
        return 3, "numbered_section", stripped

    if DENSE_SUBSECTION_RE.match(stripped):
        normalized_text = normalize_dense_subsection(stripped, state)
        if normalized_text != stripped:
            return 4, "dense_numbered_subsection", normalized_text

    return None


def apply_heading_detection(
    markdown_text: str,
    page_number: int,
    state: dict[str, Any],
    enabled: bool = True,
    max_chars: int = 180,
) -> tuple[str, list[dict[str, Any]]]:
    if not enabled:
        return markdown_text, []

    enhanced_lines = []
    headings = []
    for line in markdown_text.splitlines():
        classification = classify_heading_line(line, state, max_chars)
        if not classification:
            enhanced_lines.append(line)
            continue

        level, reason, heading_text = classification
        enhanced_lines.append(f"{'#' * level} {heading_text}")
        headings.append(
            {
                "page_number": page_number,
                "level": level,
                "text": heading_text,
                "reason": reason,
            }
        )

    return "\n".join(enhanced_lines), headings
