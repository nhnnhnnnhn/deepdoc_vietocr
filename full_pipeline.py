import logging
import os
import sys
import argparse
import difflib
import json
import numpy as np
import re
import time

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../')))

from module.ocr import OCR, resolve_device
from module import LayoutRecognizer, TableStructureRecognizer, init_in_out
from utils.ai_postcheck import format_token_usage, get_gemini_api_key, run_gemini_postcheck
from utils.document_analysis import add_text_risks, apply_heading_detection, page_has_multicolumn_layout
from utils.pipeline_artifacts import (
    correction_rule_id,
    load_correction_memory as load_correction_memory_payload,
    merged_markdown_output_path,
    metadata_output_path,
    ocr_raw_output_path,
    relative_output_path,
    save_correction_memory as save_correction_memory_payload,
    vlm_checked_output_path,
    vlm_review_output_path,
)
from utils.runtime_env import configure_run_logging, env_bool, env_value, load_env_file

from datetime import datetime

VISUAL_REGION_TYPES = {"figure", "equation"}
DEFAULT_GEMINI_SYSTEM_PROMPT = os.path.join("conf", "ai_postcheck_system_prompt.txt")
DEFAULT_CORRECTION_MEMORY_PATH = os.path.join("conf", "ocr_correction_memory.json")
CORRECTION_MEMORY_VERSION = 1
CORRECTION_MEMORY_MAX_CHARS = 240
CORRECTION_MEMORY_MAX_LINES = 4


PIPELINE_LOG_HANDLE = configure_run_logging("log", "full_pipeline.log")

def extract_table_markdown(img, table_region, ocr, table_recognizer, low_score_threshold):
    # Use bbox if present
    if "bbox" in table_region:
        x0, y0, x1, y1 = map(int, table_region["bbox"])
    else:
        x0, y0, x1, y1 = map(int, [table_region["x0"], table_region["top"], table_region["x1"], table_region["bottom"]])
    table_img = img.crop((x0, y0, x1, y1))
    tb_cpns = table_recognizer([table_img])[0]
    ocr_results = normalize_ocr_results(ocr(np.array(table_img)))
    confidence = build_ocr_confidence_stats(ocr_results, low_score_threshold)
    boxes = LayoutRecognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "score": float(t[1]),
          "layout_type": "table",
          "page_number": 0} for b, t in ocr_results if t and t[0] and b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b, _ in ocr_results]) / 3 if ocr_results else 1
    )
    if not boxes:
        return "", confidence

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = LayoutRecognizer.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
        eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
        return LayoutRecognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted([r for r in tb_cpns if re.match(
        r"table column$", r["label"])], key=lambda x: x["x0"])
    clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    markdown = TableStructureRecognizer.construct_table(boxes, markdown=True)
    return markdown, confidence


def resolve_correction_memory_path(args):
    path = getattr(args, "correction_memory_path", None) or DEFAULT_CORRECTION_MEMORY_PATH
    return os.path.abspath(path)


def load_correction_memory(path):
    return load_correction_memory_payload(path, CORRECTION_MEMORY_VERSION)


def save_correction_memory(path, memory):
    save_correction_memory_payload(path, memory)


def correction_similarity(wrong, correct):
    return difflib.SequenceMatcher(None, wrong, correct, autojunk=False).ratio()


def is_memory_blocked(memory, wrong, correct):
    rule_id = correction_rule_id(wrong, correct)
    for rule in memory.get("rules", []):
        if not isinstance(rule, dict):
            continue
        if rule.get("id") == rule_id and rule.get("blocked"):
            return True
    for item in memory.get("blocked", []):
        if not isinstance(item, dict):
            continue
        if item.get("wrong") == wrong and item.get("correct") == correct:
            return True
    return False


def correction_candidate_reason(wrong, correct, min_similarity):
    wrong = wrong.strip()
    correct = correct.strip()
    if not wrong or not correct:
        return "empty"
    if wrong == correct:
        return "same"
    if len(wrong) > CORRECTION_MEMORY_MAX_CHARS or len(correct) > CORRECTION_MEMORY_MAX_CHARS:
        return "too_long"
    if wrong.count("\n") >= CORRECTION_MEMORY_MAX_LINES or correct.count("\n") >= CORRECTION_MEMORY_MAX_LINES:
        return "too_many_lines"
    if any(marker in wrong or marker in correct for marker in ("![", "<table", "</table", "assets/")):
        return "structured_or_asset"
    similarity = correction_similarity(wrong, correct)
    if similarity < min_similarity:
        return "low_similarity"
    return ""


def extract_correction_candidates(source_text, checked_text, issues=None, min_similarity=0.72):
    source_lines = source_text.splitlines(keepends=True)
    checked_lines = checked_text.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, source_lines, checked_lines, autojunk=False)
    issues = issues if isinstance(issues, list) else []
    candidates = []
    skipped = {}
    change_id = 0
    for tag, source_start, source_end, checked_start, checked_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        wrong = "".join(source_lines[source_start:source_end]).strip()
        correct = "".join(checked_lines[checked_start:checked_end]).strip()
        reason = correction_candidate_reason(wrong, correct, min_similarity)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            change_id += 1
            continue
        issue = issues[change_id] if change_id < len(issues) and isinstance(issues[change_id], dict) else {}
        candidates.append(
            {
                "id": correction_rule_id(wrong, correct),
                "wrong": wrong,
                "correct": correct,
                "kind": tag,
                "similarity": round(correction_similarity(wrong, correct), 4),
                "source_line": source_start + 1,
                "checked_line": checked_start + 1,
                "severity": issue.get("severity"),
                "page_number": issue.get("page_number"),
                "issue": issue.get("issue"),
                "suggestion": issue.get("suggestion"),
            }
        )
        change_id += 1
    return candidates, skipped


def apply_correction_memory(text, args):
    if not getattr(args, "correction_memory", True):
        return text, []
    memory_path = resolve_correction_memory_path(args)
    memory = load_correction_memory(memory_path)
    rules = [rule for rule in memory.get("rules", []) if isinstance(rule, dict) and not rule.get("blocked")]
    rules.sort(key=lambda rule: len(str(rule.get("wrong") or "")), reverse=True)
    applied = []
    now = datetime.now().isoformat(timespec="seconds")
    changed = False
    for rule in rules:
        wrong = rule.get("wrong")
        correct = rule.get("correct")
        if not isinstance(wrong, str) or not isinstance(correct, str) or not wrong or wrong == correct:
            continue
        count = text.count(wrong)
        if count <= 0:
            continue
        text = text.replace(wrong, correct)
        rule["applied_count"] = int(rule.get("applied_count") or 0) + count
        rule["last_applied_at"] = now
        applied.append(
            {
                "id": rule.get("id") or correction_rule_id(wrong, correct),
                "wrong": wrong,
                "correct": correct,
                "count": count,
            }
        )
        changed = True
    if changed:
        try:
            save_correction_memory(memory_path, memory)
        except OSError as exc:
            print(f"OCR correction memory update failed: {exc}")
    return text, applied


def learn_corrections_from_ai_review(source_text, checked_text, review_payload, args):
    if not getattr(args, "correction_memory", True):
        return {"enabled": False, "learned": 0, "updated": 0, "blocked": 0, "skipped": {}}
    memory_path = resolve_correction_memory_path(args)
    min_similarity = float(getattr(args, "correction_memory_min_similarity", 0.72))
    candidates, skipped = extract_correction_candidates(
        source_text,
        checked_text,
        review_payload.get("issues", []),
        min_similarity=min_similarity,
    )
    memory = load_correction_memory(memory_path)
    now = datetime.now().isoformat(timespec="seconds")
    by_id = {rule.get("id"): rule for rule in memory.get("rules", []) if isinstance(rule, dict)}
    learned = 0
    updated = 0
    blocked = 0

    for candidate in candidates:
        if is_memory_blocked(memory, candidate["wrong"], candidate["correct"]):
            blocked += 1
            continue
        rule = by_id.get(candidate["id"])
        if rule:
            rule["seen_count"] = int(rule.get("seen_count") or 0) + 1
            rule["last_seen_at"] = now
            rule.update({key: value for key, value in candidate.items() if value is not None})
            updated += 1
        else:
            rule = {
                **candidate,
                "created_at": now,
                "last_seen_at": now,
                "seen_count": 1,
                "applied_count": 0,
                "blocked": False,
            }
            memory["rules"].append(rule)
            by_id[candidate["id"]] = rule
            learned += 1

    if learned or updated:
        save_correction_memory(memory_path, memory)

    return {
        "enabled": True,
        "path": memory_path,
        "learned": learned,
        "updated": updated,
        "blocked": blocked,
        "skipped": skipped,
    }


def bbox_from_region(region, image_size):
    if "bbox" in region:
        x0, y0, x1, y1 = region["bbox"]
    else:
        x0, y0, x1, y1 = region.get("x0", 0), region.get("top", 0), region.get("x1", 0), region.get("bottom", 0)

    width, height = image_size
    x0 = max(0, min(width, int(round(x0))))
    y0 = max(0, min(height, int(round(y0))))
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    return x0, y0, x1, y1


def safe_asset_name(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "image"


def save_visual_region(img, bbox, assets_dir, page_number, region_number, label):
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return None

    os.makedirs(assets_dir, exist_ok=True)
    filename = f"page_{page_number:03d}_region_{region_number:03d}_{safe_asset_name(label)}.png"
    asset_path = os.path.join(assets_dir, filename)
    img.crop((x0, y0, x1, y1)).save(asset_path)
    return asset_path


def padded_bbox(bbox, image_size, padding=24):
    x0, y0, x1, y1 = bbox
    width, height = image_size
    return [
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    ]


def crop_area(bbox):
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def rounded_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def normalize_ocr_results(ocr_output):
    if not ocr_output:
        return []
    if isinstance(ocr_output, tuple) and len(ocr_output) == 3 and isinstance(ocr_output[2], dict):
        return []
    return list(ocr_output)


def build_ocr_confidence_stats(ocr_results, low_score_threshold):
    scores = []
    char_count = 0
    for _, rec_result in ocr_results or []:
        if not rec_result:
            continue

        text = rec_result[0] if len(rec_result) > 0 else ""
        score = rec_result[1] if len(rec_result) > 1 else None
        if not text or score is None:
            continue

        text = str(text).strip()
        if not text:
            continue

        scores.append(float(score))
        char_count += len(text)

    stats = {
        "line_count": len(scores),
        "char_count": char_count,
        "low_score_threshold": float(low_score_threshold),
    }
    if not scores:
        stats.update({
            "avg_score": None,
            "min_score": None,
            "median_score": None,
            "low_score_count": 0,
            "low_score_ratio": 0.0,
        })
        return stats

    low_score_count = sum(1 for score in scores if score < float(low_score_threshold))
    stats.update({
        "avg_score": rounded_or_none(sum(scores) / len(scores)),
        "min_score": rounded_or_none(min(scores)),
        "median_score": rounded_or_none(np.median(scores)),
        "low_score_count": low_score_count,
        "low_score_ratio": rounded_or_none(low_score_count / len(scores)),
    })
    return stats


def aggregate_ocr_confidence_stats(stats_list, low_score_threshold):
    total_lines = sum(int(stats.get("line_count") or 0) for stats in stats_list)
    total_chars = sum(int(stats.get("char_count") or 0) for stats in stats_list)
    low_score_count = sum(int(stats.get("low_score_count") or 0) for stats in stats_list)
    scored_stats = [
        stats for stats in stats_list
        if stats.get("avg_score") is not None and int(stats.get("line_count") or 0) > 0
    ]

    aggregated = {
        "line_count": total_lines,
        "char_count": total_chars,
        "low_score_threshold": float(low_score_threshold),
        "low_score_count": low_score_count,
        "low_score_ratio": rounded_or_none(low_score_count / total_lines) if total_lines else 0.0,
    }
    if not scored_stats:
        aggregated.update({
            "avg_score": None,
            "min_score": None,
            "median_score": None,
        })
        return aggregated

    weighted_score = sum(float(stats["avg_score"]) * int(stats["line_count"]) for stats in scored_stats)
    mins = [float(stats["min_score"]) for stats in scored_stats if stats.get("min_score") is not None]
    medians = [
        float(stats["median_score"])
        for stats in scored_stats
        for _ in range(max(1, int(stats.get("line_count") or 1)))
        if stats.get("median_score") is not None
    ]
    aggregated.update({
        "avg_score": rounded_or_none(weighted_score / total_lines) if total_lines else None,
        "min_score": rounded_or_none(min(mins)) if mins else None,
        "median_score": rounded_or_none(np.median(medians)) if medians else None,
    })
    return aggregated


def page_ocr_confidence(page_metadata, low_score_threshold):
    return aggregate_ocr_confidence_stats(
        [
            region["ocr_confidence"]
            for region in page_metadata.get("regions", [])
            if region.get("ocr_confidence")
        ],
        low_score_threshold,
    )


def document_ocr_confidence(metadata, low_score_threshold):
    return aggregate_ocr_confidence_stats(
        [
            page["ocr_confidence"]
            for page in metadata.get("pages", [])
            if page.get("ocr_confidence")
        ],
        low_score_threshold,
    )


def build_layout_confidence_stats(layout_regions, low_score_threshold):
    scores = [
        float(region.get("score"))
        for region in layout_regions
        if region.get("score") is not None
    ]
    stats = {
        "region_count": len(scores),
        "low_score_threshold": float(low_score_threshold),
    }
    if not scores:
        stats.update({
            "avg_score": None,
            "min_score": None,
            "median_score": None,
            "low_score_count": 0,
            "low_score_ratio": 0.0,
        })
        return stats

    low_score_count = sum(1 for score in scores if score < float(low_score_threshold))
    stats.update({
        "avg_score": rounded_or_none(sum(scores) / len(scores)),
        "min_score": rounded_or_none(min(scores)),
        "median_score": rounded_or_none(np.median(scores)),
        "low_score_count": low_score_count,
        "low_score_ratio": rounded_or_none(low_score_count / len(scores)),
    })
    return stats


def document_layout_confidence(metadata, low_score_threshold):
    scores = [
        float(region.get("score"))
        for page in metadata.get("pages", [])
        for region in page.get("regions", [])
        if region.get("score") is not None
    ]
    return build_layout_confidence_stats(
        [{"score": score} for score in scores],
        low_score_threshold,
    )


def page_layout_confidence(page_metadata, low_score_threshold):
    return build_layout_confidence_stats(
        [
            {"score": region.get("score")}
            for region in page_metadata.get("regions", [])
            if region.get("score") is not None
        ],
        low_score_threshold,
    )


def page_has_text_layout(page_metadata):
    return any(
        region.get("type") in {"text", "title", "table"}
        for region in page_metadata.get("layout_regions", [])
    )


def evaluate_low_confidence_gate(metadata, args):
    confidence = metadata.get("ocr_confidence") or {}
    layout_confidence = metadata.get("layout_confidence") or {}
    avg_threshold = float(args.vlm_ocr_avg_confidence_threshold)
    low_line_ratio_threshold = float(args.vlm_ocr_low_line_ratio)
    very_low_line_threshold = float(args.vlm_ocr_very_low_line_threshold)
    layout_avg_threshold = float(args.vlm_layout_avg_confidence_threshold)
    layout_low_region_ratio_threshold = float(args.vlm_layout_low_region_ratio)
    min_lines = int(args.vlm_ocr_min_lines)
    line_count = int(confidence.get("line_count") or 0)
    avg_score = confidence.get("avg_score")
    min_score = confidence.get("min_score")
    low_score_ratio = float(confidence.get("low_score_ratio") or 0.0)
    layout_avg_score = layout_confidence.get("avg_score")
    layout_low_score_ratio = float(layout_confidence.get("low_score_ratio") or 0.0)
    layout_region_count = int(layout_confidence.get("region_count") or 0)
    reasons = []

    if line_count < min_lines and any(page_has_text_layout(page) for page in metadata.get("pages", [])):
        reasons.append(
            f"recognized OCR lines {line_count} < minimum {min_lines} on a document with text/table layout"
        )
    if avg_score is not None and float(avg_score) < avg_threshold:
        reasons.append(f"average OCR confidence {avg_score:.4f} < threshold {avg_threshold:.4f}")
    if min_score is not None and float(min_score) < very_low_line_threshold:
        reasons.append(f"minimum OCR line confidence {min_score:.4f} < threshold {very_low_line_threshold:.4f}")
    if low_score_ratio >= low_line_ratio_threshold and line_count > 0:
        reasons.append(
            f"low-confidence OCR line ratio {low_score_ratio:.4f} >= threshold {low_line_ratio_threshold:.4f}"
        )
    if layout_avg_score is not None and float(layout_avg_score) < layout_avg_threshold:
        reasons.append(f"average layout confidence {layout_avg_score:.4f} < threshold {layout_avg_threshold:.4f}")
    if layout_region_count > 0 and layout_low_score_ratio >= layout_low_region_ratio_threshold:
        reasons.append(
            f"low-confidence layout region ratio {layout_low_score_ratio:.4f} >= "
            f"threshold {layout_low_region_ratio_threshold:.4f}"
        )

    low_confidence_categories = {"ocr_too_short_for_region", "ambiguous_characters"}
    low_confidence_risks = [
        risk for risk in metadata.get("vlm_risks", [])
        if risk.get("category") in low_confidence_categories
    ]
    if low_confidence_risks:
        reasons.append(
            f"{len(low_confidence_risks)} OCR uncertainty risk(s): "
            f"{', '.join(sorted({risk.get('category') for risk in low_confidence_risks}))}"
        )

    return {
        "low_confidence": bool(reasons),
        "reasons": reasons,
        "confidence": confidence,
        "layout_confidence": layout_confidence,
        "thresholds": {
            "avg_confidence": avg_threshold,
            "low_line_threshold": float(args.vlm_ocr_low_line_threshold),
            "low_line_ratio": low_line_ratio_threshold,
            "very_low_line_threshold": very_low_line_threshold,
            "min_lines": min_lines,
            "layout_avg_confidence": layout_avg_threshold,
            "layout_low_region_threshold": float(args.vlm_layout_low_region_threshold),
            "layout_low_region_ratio": layout_low_region_ratio_threshold,
        },
    }


def gemini_postcheck_decision(metadata, args):
    if not args.gemini_postcheck:
        return {
            "enabled": False,
            "run": False,
            "trigger": "disabled",
            "reason": "GEMINI_POSTCHECK=false",
        }

    trigger = str(args.vlm_postcheck_trigger or "low_confidence").strip().lower()
    low_confidence = evaluate_low_confidence_gate(metadata, args)
    if trigger == "always":
        return {
            "enabled": True,
            "run": True,
            "trigger": trigger,
            "reason": "VLM_POSTCHECK_TRIGGER=always",
            "low_confidence_gate": low_confidence,
        }
    if trigger == "never":
        return {
            "enabled": True,
            "run": False,
            "trigger": trigger,
            "reason": "VLM_POSTCHECK_TRIGGER=never",
            "low_confidence_gate": low_confidence,
        }
    if trigger == "risk_or_low_confidence":
        has_risks = bool(metadata.get("vlm_risks"))
        should_run = has_risks or low_confidence["low_confidence"]
        reason = "VLM risks present" if has_risks else "low confidence gate matched"
        if not should_run:
            reason = "no VLM risk and OCR confidence passed"
        return {
            "enabled": True,
            "run": should_run,
            "trigger": trigger,
            "reason": reason,
            "low_confidence_gate": low_confidence,
        }

    should_run = low_confidence["low_confidence"]
    return {
        "enabled": True,
        "run": should_run,
        "trigger": "low_confidence",
        "reason": "; ".join(low_confidence["reasons"]) if should_run else "OCR confidence passed; VLM post-check skipped",
        "low_confidence_gate": low_confidence,
    }


def save_vlm_risk_region(img, bbox, assets_dir, page_number, risk_number, category):
    bbox = padded_bbox(bbox, img.size)
    if crop_area(bbox) <= 0:
        return None

    os.makedirs(assets_dir, exist_ok=True)
    filename = f"page_{page_number:03d}_risk_{risk_number:03d}_{safe_asset_name(category)}.png"
    asset_path = os.path.join(assets_dir, filename)
    x0, y0, x1, y1 = bbox
    img.crop((x0, y0, x1, y1)).save(asset_path)
    return asset_path


def attach_vlm_risk_assets(page_metadata, img, risk_assets_dir, output_folder, max_crops):
    for risk in page_metadata["vlm_risks"][:max_crops]:
        if "asset_path" in risk or "bbox" not in risk:
            continue

        asset_path = save_vlm_risk_region(
            img,
            risk["bbox"],
            risk_assets_dir,
            page_metadata["page_number"],
            len([r for r in page_metadata["vlm_risks"] if r.get("asset_path")]) + 1,
            risk["category"],
        )
        if asset_path:
            risk["asset_path"] = relative_output_path(asset_path, output_folder)


def new_document_metadata(args, markdown_output_path):
    output_folder = os.path.dirname(markdown_output_path)
    return {
        "input": args.inputs,
        "output_markdown": os.path.basename(markdown_output_path),
        "assets_dir": relative_output_path(os.path.join(output_folder, "assets"), output_folder),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pages": [],
    }


def find_unmasked_bands(inv_mask, min_gap_rows=8):
    """
    Split an inverse-mask image into vertical bands separated by fully-masked
    (all-zero) horizontal strips of at least `min_gap_rows` pixels.

    Returns a list of (x0, y0, x1, y1) bounding boxes, one per band,
    ordered top-to-bottom.  Each box tightly wraps the unmasked pixels in
    that band so the OCR crop is as small as possible.
    """
    arr = np.array(inv_mask, dtype=np.uint8)   # shape: (H, W), values 0/1
    row_has_content = arr.any(axis=1)           # True for each row with ≥1 unmasked pixel

    bands = []
    in_band = False
    band_start = 0
    gap_count = 0

    for y, has in enumerate(row_has_content):
        if has:
            if not in_band:
                in_band = True
                band_start = y
            gap_count = 0
        else:
            if in_band:
                gap_count += 1
                if gap_count >= min_gap_rows:
                    bands.append((band_start, y - gap_count))
                    in_band = False
                    gap_count = 0

    if in_band:
        bands.append((band_start, len(row_has_content) - 1))

    result = []
    width = arr.shape[1]
    for y0, y1 in bands:
        band_arr = arr[y0 : y1 + 1, :]
        col_has_content = band_arr.any(axis=0)
        if not col_has_content.any():
            continue
        col_indices = np.where(col_has_content)[0]
        x0 = int(col_indices[0])
        x1 = int(col_indices[-1])
        result.append((x0, y0, min(x1 + 1, width - 1), y1))

    return result


def main(args):
    gemini_api_key = get_gemini_api_key(args)

    images, outputs = init_in_out(args)
    print(f"Loaded {len(images)} images")
    print(f"Output paths: {outputs}")
    runtime_device = resolve_device(args.device)
    image_threshold = float(args.image_threshold)
    heading_detection = not args.no_heading_detection
    heading_max_chars = int(args.heading_max_chars)
    heading_state = {"document_title_seen": False}
    vlm_risk_detection = not args.no_vlm_risk_detection
    vlm_max_risk_crops = int(args.vlm_max_risk_crops)
    vlm_low_text_density = float(args.vlm_low_text_density)
    vlm_ocr_low_line_threshold = float(args.vlm_ocr_low_line_threshold)
    vlm_layout_low_region_threshold = float(args.vlm_layout_low_region_threshold)

    print(f"Requested device: {args.device}")
    print(f"Resolved device: {runtime_device.describe()}")
    print(f"Image extraction threshold: {image_threshold}")
    print(f"Heading detection: {'enabled' if heading_detection else 'disabled'}")
    print(f"VLM risk detection: {'enabled' if vlm_risk_detection else 'disabled'}")
    print(f"AI post-check: {'enabled' if args.gemini_postcheck else 'disabled'}")
    print(
        "OCR correction memory: "
        f"{'enabled' if getattr(args, 'correction_memory', True) else 'disabled'} "
        f"({resolve_correction_memory_path(args)})"
    )
    if args.gemini_postcheck:
        print(f"VLM post-check trigger: {args.vlm_postcheck_trigger}")
        print(
            "OCR confidence gate: "
            f"avg<{float(args.vlm_ocr_avg_confidence_threshold):.2f}, "
            f"line<{vlm_ocr_low_line_threshold:.2f}, "
            f"very_low<{float(args.vlm_ocr_very_low_line_threshold):.2f}, "
            f"ratio>={float(args.vlm_ocr_low_line_ratio):.2f}, "
            f"layout_avg<{float(args.vlm_layout_avg_confidence_threshold):.2f}, "
            f"layout_region<{vlm_layout_low_region_threshold:.2f}"
        )

    layout_recognizer = LayoutRecognizer("layout", device=runtime_device)
    table_recognizer = TableStructureRecognizer(device=runtime_device)
    ocr = OCR(device=runtime_device)
    print(f"Layout ONNX providers: {layout_recognizer.ort_sess.get_providers()}")
    print(f"Table ONNX providers: {table_recognizer.ort_sess.get_providers()}")
    print(f"OCR detector ONNX providers: {ocr.text_detector[0].predictor.get_providers()}")
    print(f"VietOCR recognizer device: {ocr.text_recognizer[0].device}")

    markdown_pages_by_output = {}
    metadata_by_output = {}
    page_images_by_output = {}
    for idx, img in enumerate(images):
        print(f"Processing image {idx}: {outputs[idx]}")
        start_time = time.time()  # <-- Start timing

        layouts = layout_recognizer.forward([img], thr=float(args.threshold))[0]
        print(f"Detected {len(layouts)} layout regions")
        region_and_pos = []
        out_path = merged_markdown_output_path(outputs[idx])
        output_folder = os.path.dirname(out_path)
        assets_dir = os.path.join(output_folder, "assets")
        risk_assets_dir = os.path.join(assets_dir, "vlm")
        document_metadata = metadata_by_output.setdefault(out_path, new_document_metadata(args, out_path))
        page_number = len(document_metadata["pages"]) + 1
        page_metadata = {
            "page_number": page_number,
            "width": img.size[0],
            "height": img.size[1],
            "source_output_base": relative_output_path(outputs[idx], output_folder),
            "layout_regions": [],
            "regions": [],
            "headings": [],
            "vlm_risks": [],
        }

        from PIL import Image, ImageDraw

        # Create a mask for detected regions
        mask = Image.new("1", img.size, 0)
        draw = ImageDraw.Draw(mask)
        for region in layouts:
            x0, y0, x1, y1 = bbox_from_region(region, img.size)
            page_metadata["layout_regions"].append({
                "type": region.get("type", "").lower(),
                "score": float(region.get("score", 1.0)),
                "bbox": [x0, y0, x1, y1],
            })
            draw.rectangle([x0, y0, x1, y1], fill=1)
        page_metadata["raw_layout_confidence"] = build_layout_confidence_stats(
            page_metadata["layout_regions"],
            float(args.vlm_layout_low_region_threshold),
        )

        if vlm_risk_detection and page_has_multicolumn_layout(page_metadata):
            add_vlm_risk(
                page_metadata,
                "multi_column_page",
                "Page appears to contain multiple text columns; reading order should be verified visually.",
                bbox=[0, 0, img.size[0], img.size[1]],
                severity="high",
            )

        for region in layouts:
            label = region.get("type", "").lower()
            score = region.get("score", 1.0)
            bbox = bbox_from_region(region, img.size)
            y_pos = bbox[1]  # Use top y as position for ordering
            if label in ["table"] and score >= float(args.threshold):
                print(f"Extracting table markdown for region: {region}")
                markdown, ocr_confidence = extract_table_markdown(
                    img,
                    region,
                    ocr,
                    table_recognizer,
                    vlm_ocr_low_line_threshold,
                )
                region_and_pos.append((y_pos, markdown))
                page_metadata["regions"].append({
                    "content_type": "table",
                    "layout_type": label,
                    "score": float(score),
                    "bbox": list(bbox),
                    "markdown": markdown,
                    "ocr_confidence": ocr_confidence,
                })
                if vlm_risk_detection:
                    add_vlm_risk(
                        page_metadata,
                        "table_or_form_block",
                        "Detected table/form-like block; structure and cell values should be verified by VLM.",
                        bbox=bbox,
                        text=markdown,
                        severity="high",
                    )
                    add_text_risks(page_metadata, markdown, bbox, vlm_low_text_density)
            elif label in VISUAL_REGION_TYPES and score >= image_threshold:
                asset_path = save_visual_region(img, bbox, assets_dir, page_number, len(page_metadata["regions"]) + 1, label)
                if not asset_path:
                    continue

                rel_asset_path = relative_output_path(asset_path, output_folder)
                alt_text = f"{label} page {page_number}"
                region_and_pos.append((y_pos, f"![{alt_text}]({rel_asset_path})"))
                page_metadata["regions"].append({
                    "content_type": "image",
                    "layout_type": label,
                    "score": float(score),
                    "bbox": list(bbox),
                    "asset_path": rel_asset_path,
                })
                if vlm_risk_detection:
                    add_vlm_risk(
                        page_metadata,
                        "signature_stamp_qr_or_visual_mark",
                        "Detected visual region that may contain signature, stamp, QR code, screenshot, chart, or other non-text evidence.",
                        bbox=bbox,
                        asset_path=rel_asset_path,
                        severity="medium",
                    )

        # Now OCR any remaining undetected area (including non-table/figure).
        # Split the inverse mask into vertical bands so that header, body, and
        # footer sections are each OCR-ed independently, preserving reading order.
        inv_mask = mask.point(lambda p: 1 - p)
        for bx0, by0, bx1, by1 in find_unmasked_bands(inv_mask):
            region_img = img.crop((bx0, by0, bx1, by1))
            ocr_results = normalize_ocr_results(ocr(np.array(region_img)))
            ocr_confidence = build_ocr_confidence_stats(ocr_results, vlm_ocr_low_line_threshold)
            text = "\n".join([t[0] for _, t in ocr_results if t and t[0]])
            if text.strip():
                region_and_pos.append((by0, text))
                page_metadata["regions"].append({
                    "content_type": "text",
                    "layout_type": "remaining_area",
                    "bbox": [bx0, by0, bx1, by1],
                    "text": text,
                    "ocr_confidence": ocr_confidence,
                })
                if vlm_risk_detection:
                    add_text_risks(page_metadata, text, [bx0, by0, bx1, by1], vlm_low_text_density)

        if vlm_risk_detection:
            attach_vlm_risk_assets(page_metadata, img, risk_assets_dir, output_folder, vlm_max_risk_crops)
        page_metadata["ocr_confidence"] = page_ocr_confidence(page_metadata, vlm_ocr_low_line_threshold)
        page_metadata["layout_confidence"] = page_layout_confidence(
            page_metadata,
            float(args.vlm_layout_low_region_threshold),
        )

        # Sort by y position to preserve original order
        region_and_pos.sort(key=lambda x: x[0])
        markdown_concat = "\n\n".join([item[1] for item in region_and_pos])
        markdown_concat, headings = apply_heading_detection(
            markdown_concat,
            page_number,
            heading_state,
            enabled=heading_detection,
            max_chars=heading_max_chars,
        )
        page_metadata["headings"] = headings
        markdown_pages_by_output.setdefault(out_path, []).append(markdown_concat)
        document_metadata["pages"].append(page_metadata)
        page_images_by_output.setdefault(out_path, []).append(img)
        print(f"Queued page markdown for merged output: {out_path}")

        elapsed = time.time() - start_time  # <-- End timing
        print(f"Processing image {idx} done in {elapsed:.2f} seconds")  # <-- Print elapsed time

    for out_path, markdown_pages in markdown_pages_by_output.items():
        markdown_concat = "\n\n".join([page for page in markdown_pages if page])

        raw_path = ocr_raw_output_path(out_path)
        print(f"Writing OCR raw markdown to: {raw_path}")
        with open(raw_path, "w+", encoding="utf-8") as f:
            f.write(markdown_concat)
        logging.info(f"Saved OCR raw markdown to: {raw_path}")

        memory_applied = []
        try:
            markdown_concat, memory_applied = apply_correction_memory(markdown_concat, args)
        except Exception as exc:
            print(f"OCR correction memory apply failed, keeping current markdown: {exc}")
        print(f"Writing merged markdown to: {out_path}")
        with open(out_path, "w+", encoding='utf-8') as f:
            f.write(markdown_concat)
        logging.info(f"Saved merged markdown to: {out_path}")

        document_metadata = metadata_by_output[out_path]
        document_metadata["page_count"] = len(document_metadata["pages"])
        document_metadata["region_count"] = sum(len(page["regions"]) for page in document_metadata["pages"])
        document_metadata["heading_count"] = sum(len(page.get("headings", [])) for page in document_metadata["pages"])
        document_metadata["headings"] = [
            heading
            for page in document_metadata["pages"]
            for heading in page.get("headings", [])
        ]
        document_metadata["vlm_risk_count"] = sum(len(page.get("vlm_risks", [])) for page in document_metadata["pages"])
        document_metadata["vlm_risks"] = [
            risk
            for page in document_metadata["pages"]
            for risk in page.get("vlm_risks", [])
        ]
        document_metadata["ocr_confidence"] = document_ocr_confidence(
            document_metadata,
            float(args.vlm_ocr_low_line_threshold),
        )
        document_metadata["layout_confidence"] = document_layout_confidence(
            document_metadata,
            float(args.vlm_layout_low_region_threshold),
        )
        document_metadata["ocr_correction_memory"] = {
            "enabled": bool(getattr(args, "correction_memory", True)),
            "path": resolve_correction_memory_path(args),
            "applied_count": sum(item.get("count", 0) for item in memory_applied),
            "applied": memory_applied,
        }
        if memory_applied:
            print(
                "OCR correction memory applied: "
                f"{document_metadata['ocr_correction_memory']['applied_count']} replacement(s)"
            )
        postcheck_decision = gemini_postcheck_decision(document_metadata, args)
        document_metadata["gemini_postcheck_decision"] = postcheck_decision
        document_metadata["asset_count"] = sum(
            1
            for page in document_metadata["pages"]
            for region in page["regions"]
            if region.get("content_type") == "image"
        )
        meta_path = metadata_output_path(out_path)
        print(f"Writing metadata to: {meta_path}")
        with open(meta_path, "w+", encoding="utf-8") as f:
            json.dump(document_metadata, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved metadata to: {meta_path}")

        if args.gemini_postcheck and postcheck_decision["run"]:
            ai_postcheck_required = env_bool("AI_POSTCHECK_REQUIRED", False)
            if not gemini_api_key:
                error_message = (
                    "AI post-check was requested by the confidence gate but no API key was provided. "
                    "Set AI_API_KEY or OPENAI_API_KEY."
                )
                if ai_postcheck_required:
                    raise RuntimeError(error_message)
                print(f"AI post-check failed, keeping raw markdown: {error_message}")
                document_metadata["gemini_postcheck"] = {
                    "status": "failed",
                    "model": args.gemini_model,
                    "decision": postcheck_decision,
                    "error": error_message,
                    "token_usage": {},
                }
                with open(meta_path, "w+", encoding="utf-8") as f:
                    json.dump(document_metadata, f, ensure_ascii=False, indent=2)
            else:
                try:
                    print(f"Running AI post-check with model {args.gemini_model}: {out_path}")
                    print(f"VLM gate reason: {postcheck_decision['reason']}")
                    document_metadata["_output_folder"] = os.path.dirname(out_path)
                    gemini_result = run_gemini_postcheck(
                        markdown_concat,
                        document_metadata,
                        page_images_by_output.get(out_path, []),
                        args,
                        gemini_api_key,
                        DEFAULT_GEMINI_SYSTEM_PROMPT,
                    )
                    document_metadata.pop("_output_folder", None)
                    token_usage = gemini_result.get("_token_usage", {})
                    raw_usage_metadata = gemini_result.get("_raw_usage_metadata", {})
                    print(f"AI token usage: {format_token_usage(token_usage)}")

                    checked_path = vlm_checked_output_path(out_path)
                    checked_markdown = gemini_result["checked_markdown"]
                    with open(checked_path, "w+", encoding="utf-8") as f:
                        f.write(checked_markdown)

                    review_payload = {
                        "model": args.gemini_model,
                        "source_markdown": os.path.basename(out_path),
                        "checked_markdown": os.path.basename(checked_path),
                        "summary": gemini_result.get("summary", ""),
                        "issues": gemini_result.get("issues", []),
                        "image_notes": gemini_result.get("image_notes", []),
                        "vlm_findings": gemini_result.get("vlm_findings", []),
                        "confidence": gemini_result.get("confidence"),
                        "token_usage": token_usage,
                        "raw_usage_metadata": raw_usage_metadata,
                    }
                    try:
                        learning_summary = learn_corrections_from_ai_review(
                            markdown_concat,
                            checked_markdown,
                            review_payload,
                            args,
                        )
                    except Exception as exc:
                        learning_summary = {"enabled": True, "error": str(exc)}
                        print(f"OCR correction memory learning failed: {exc}")
                    review_payload["learned_corrections"] = learning_summary
                    if learning_summary.get("enabled"):
                        print(
                            "OCR correction memory learned: "
                            f"{learning_summary.get('learned', 0)} new, "
                            f"{learning_summary.get('updated', 0)} updated, "
                            f"{learning_summary.get('blocked', 0)} blocked"
                        )
                    review_path = vlm_review_output_path(out_path)
                    with open(review_path, "w+", encoding="utf-8") as f:
                        json.dump(review_payload, f, ensure_ascii=False, indent=2)
                    document_metadata["gemini_postcheck"] = {
                        "status": "completed",
                        "model": args.gemini_model,
                        "checked_markdown": os.path.basename(checked_path),
                        "review": os.path.basename(review_path),
                        "decision": postcheck_decision,
                        "token_usage": token_usage,
                        "raw_usage_metadata": raw_usage_metadata,
                        "learned_corrections": learning_summary,
                    }
                    with open(meta_path, "w+", encoding="utf-8") as f:
                        json.dump(document_metadata, f, ensure_ascii=False, indent=2)
                    logging.info(f"Saved AI checked markdown to: {checked_path}")
                    logging.info(f"Saved AI review to: {review_path}")
                except Exception as exc:
                    document_metadata.pop("_output_folder", None)
                    if ai_postcheck_required:
                        raise
                    error_message = str(exc)
                    print(f"AI post-check failed, keeping raw markdown: {error_message}")
                    logging.exception("AI post-check failed")
                    document_metadata["gemini_postcheck"] = {
                        "status": "failed",
                        "model": args.gemini_model,
                        "decision": postcheck_decision,
                        "error": error_message,
                        "token_usage": {},
                    }
                    with open(meta_path, "w+", encoding="utf-8") as f:
                        json.dump(document_metadata, f, ensure_ascii=False, indent=2)
        elif args.gemini_postcheck:
            print(f"Skipping AI/VLM post-check: {postcheck_decision['reason']}")
            document_metadata["gemini_postcheck"] = {
                "status": "skipped",
                "model": args.gemini_model,
                "decision": postcheck_decision,
                "token_usage": {},
            }
            with open(meta_path, "w+", encoding="utf-8") as f:
                json.dump(document_metadata, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    load_env_file()

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory or file path for images or PDFs",
                        required=True)
    parser.add_argument('--output_dir', help="Directory for output markdown files. Default: './table_markdown_outputs'",
                        default=env_value("OUTPUT_DIR", "./table_markdown_outputs"))
    parser.add_argument('--threshold',
                        help="Detection threshold. Default: 0.5",
                        default=env_value("THRESHOLD", 0.5))
    parser.add_argument('--device',
                        help="Inference device: auto, cpu, cuda, or cuda:<id>. Default: auto",
                        default=env_value("DEVICE", "auto"))
    parser.add_argument('--image_threshold',
                        help="Layout score threshold for extracting figure/equation image regions. Default: 0.2",
                        default=env_value("IMAGE_THRESHOLD", 0.2))
    parser.add_argument('--no_heading_detection',
                        help="Disable automatic Markdown heading detection.",
                        action="store_true",
                        default=env_bool("NO_HEADING_DETECTION", False))
    parser.add_argument('--heading_max_chars',
                        help="Maximum line length considered for automatic headings. Default: 180",
                        default=env_value("HEADING_MAX_CHARS", 180))
    parser.add_argument('--no_vlm_risk_detection',
                        help="Disable VLM risk detection and risk-region crop generation.",
                        action="store_true",
                        default=env_bool("NO_VLM_RISK_DETECTION", False))
    parser.add_argument('--vlm_max_risk_crops',
                        help="Maximum VLM risk crops saved/sent per document. Default: 12",
                        default=env_value("VLM_MAX_RISK_CROPS", 12))
    parser.add_argument('--vlm_low_text_density',
                        help="Chars-per-pixel threshold for OCR-too-short risk. Default: 0.00025",
                        default=env_value("VLM_LOW_TEXT_DENSITY", 0.00025))
    parser.add_argument('--ai_model',
                        dest="gemini_model",
                        metavar="AI_MODEL",
                        help="OpenAI-compatible model for markdown post-check. Default: gpt-4.1-mini",
                        default=env_value("AI_MODEL", env_value("OPENAI_MODEL", "gpt-4.1-mini")))
    parser.add_argument('--gemini_model', dest="gemini_model", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument('--ai_api_key',
                        dest="gemini_api_key",
                        metavar="AI_API_KEY",
                        help="OpenAI-compatible API key. Prefer AI_API_KEY or OPENAI_API_KEY environment variable.",
                        default=env_value("AI_API_KEY", env_value("OPENAI_API_KEY", None)))
    parser.add_argument('--gemini_api_key', dest="gemini_api_key", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument('--ai_base_url',
                        help="OpenAI-compatible base URL. Default: https://api.openai.com/v1",
                        default=env_value("AI_BASE_URL", env_value("OPENAI_BASE_URL", "https://api.openai.com/v1")))
    parser.add_argument('--ai_response_format',
                        help="Response format for OpenAI-compatible post-check: json_object, json_schema, or none.",
                        default=env_value("AI_RESPONSE_FORMAT", env_value("OPENAI_RESPONSE_FORMAT", "json_object")))
    parser.add_argument('--ai_system_prompt',
                        dest="gemini_system_prompt",
                        metavar="AI_SYSTEM_PROMPT",
                        help="Path to the AI post-check system prompt file.",
                        default=env_value("AI_SYSTEM_PROMPT", env_value("OPENAI_SYSTEM_PROMPT", DEFAULT_GEMINI_SYSTEM_PROMPT)))
    parser.add_argument('--gemini_system_prompt', dest="gemini_system_prompt", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument('--ai_max_pages',
                        dest="gemini_max_pages",
                        metavar="AI_MAX_PAGES",
                        help="Maximum rendered pages to send to the AI model for visual post-check. Default: 6",
                        default=env_value("AI_MAX_PAGES", env_value("OPENAI_MAX_PAGES", 6)))
    parser.add_argument('--gemini_max_pages', dest="gemini_max_pages", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument('--ai_image_max_side',
                        dest="gemini_image_max_side",
                        metavar="AI_IMAGE_MAX_SIDE",
                        help="Resize rendered page images before sending to the AI model. Default: 1600",
                        default=env_value("AI_IMAGE_MAX_SIDE", env_value("OPENAI_IMAGE_MAX_SIDE", 1600)))
    parser.add_argument('--gemini_image_max_side', dest="gemini_image_max_side", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument('--ai_timeout',
                        dest="gemini_timeout",
                        metavar="AI_TIMEOUT",
                        help="OpenAI-compatible API timeout in seconds. Default: 180",
                        default=env_value("AI_TIMEOUT", env_value("OPENAI_TIMEOUT", 180)))
    parser.add_argument('--gemini_timeout', dest="gemini_timeout", help=argparse.SUPPRESS, default=argparse.SUPPRESS)
    parser.add_argument('--no_correction_memory',
                        dest="correction_memory",
                        help="Disable OCR correction memory learned from AI reviews.",
                        action="store_false",
                        default=env_bool("OCR_CORRECTION_MEMORY", True))
    parser.add_argument('--correction_memory_path',
                        help="Path for OCR correction memory JSON. Default: conf/ocr_correction_memory.json",
                        default=env_value("OCR_CORRECTION_MEMORY_PATH", DEFAULT_CORRECTION_MEMORY_PATH))
    parser.add_argument('--correction_memory_min_similarity',
                        help="Minimum similarity for learned replacement rules. Default: 0.72",
                        default=env_value("OCR_CORRECTION_MEMORY_MIN_SIMILARITY", 0.72))
    args = parser.parse_args()
    args.gemini_postcheck = env_bool("AI_POSTCHECK", False)
    args.vlm_postcheck_trigger = env_value("VLM_POSTCHECK_TRIGGER", "low_confidence")
    args.vlm_ocr_avg_confidence_threshold = env_value("VLM_OCR_AVG_CONFIDENCE_THRESHOLD", 0.86)
    args.vlm_ocr_low_line_threshold = env_value("VLM_OCR_LOW_LINE_THRESHOLD", 0.78)
    args.vlm_ocr_low_line_ratio = env_value("VLM_OCR_LOW_LINE_RATIO", 0.20)
    args.vlm_ocr_very_low_line_threshold = env_value("VLM_OCR_VERY_LOW_LINE_THRESHOLD", 0.60)
    args.vlm_ocr_min_lines = env_value("VLM_OCR_MIN_LINES", 1)
    args.vlm_layout_avg_confidence_threshold = env_value("VLM_LAYOUT_AVG_CONFIDENCE_THRESHOLD", 0.60)
    args.vlm_layout_low_region_threshold = env_value("VLM_LAYOUT_LOW_REGION_THRESHOLD", 0.55)
    args.vlm_layout_low_region_ratio = env_value("VLM_LAYOUT_LOW_REGION_RATIO", 0.35)
    main(args)
