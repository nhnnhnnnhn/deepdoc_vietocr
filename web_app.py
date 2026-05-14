from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "web_runs"
STATIC_DIR = BASE_DIR / "web_static"
PIPELINE_PATH = BASE_DIR / "full_pipeline.py"
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LOG_READ_LIMIT = 512 * 1024
PREVIEW_READ_LIMIT = 2 * 1024 * 1024
CHANGE_SNIPPET_LIMIT = 1800
REVERSALS_FILENAME = "vlm_reversals.json"
CORRECTION_MEMORY_FILENAME = "ocr_correction_memory.json"
HISTORY_FILENAME = "history.json"
HISTORY_PATH = RUNS_DIR / HISTORY_FILENAME
HISTORY_VERSION = 1


@dataclass
class JobState:
    job_id: str
    status: str
    input_name: str
    input_path: str
    output_dir: str
    log_path: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None
    markdown_path: str | None = None
    metadata_path: str | None = None
    gemini_status: str | None = None
    token_usage: dict[str, Any] | None = None


app = FastAPI(title="DeepDoc VietOCR Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, JobState] = {}
jobs_lock = Lock()
history_lock = Lock()


def now_iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp))


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        return "upload"
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "._- ()").strip()
    return safe or "upload"


def require_job(job_id: str) -> JobState:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def update_job(job_id: str, **changes: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        for key, value in changes.items():
            setattr(job, key, value)


def path_in_job(job: JobState, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).resolve()
    job_root = Path(job.output_dir).resolve().parent
    try:
        path.relative_to(job_root)
    except ValueError:
        return None
    return path


def resolve_path_in_job(job: JobState, path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    job_root = Path(job.output_dir).resolve().parent
    try:
        resolved.relative_to(job_root)
    except ValueError:
        return None
    return resolved


def pick_latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime, str(path)))


def read_metadata(metadata_path: Path | None) -> dict[str, Any]:
    if not metadata_path or not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_text_preview(path: Path) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        content = handle.read(PREVIEW_READ_LIMIT)
    return content.decode("utf-8", errors="replace"), size > PREVIEW_READ_LIMIT


def discover_outputs(output_dir: Path) -> tuple[Path | None, Path | None, dict[str, Any]]:
    files = discover_output_files(output_dir)
    return files["markdown"], files["metadata"], files["metadata_payload"]


def discover_output_files(output_dir: Path) -> dict[str, Any]:
    checked = pick_latest(list(output_dir.rglob("*_vlm_checked.md")))
    legacy_checked = pick_latest(list(output_dir.rglob("*_gemini_checked.md")))
    raw = pick_latest(list(output_dir.rglob("*_full.md")))
    metadata = pick_latest(list(output_dir.rglob("*_metadata.json")))
    metadata_payload = read_metadata(metadata)
    review = None
    postcheck = metadata_payload.get("ai_postcheck") or metadata_payload.get("gemini_postcheck")
    if isinstance(postcheck, dict) and metadata:
        review_name = postcheck.get("review")
        if isinstance(review_name, str) and review_name:
            candidate = (metadata.parent / review_name).resolve()
            if candidate.exists() and candidate.suffix.lower() == ".json":
                review = candidate
    if review is None:
        review = pick_latest(list(output_dir.rglob("*_vlm_review.json")))
    return {
        "markdown": checked or legacy_checked or raw,
        "checked_markdown": checked or legacy_checked,
        "raw_markdown": raw,
        "metadata": metadata,
        "metadata_payload": metadata_payload,
        "review": review,
    }


def summarize_gemini(metadata: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    postcheck = metadata.get("ai_postcheck") or metadata.get("gemini_postcheck")
    if isinstance(postcheck, dict):
        status = str(postcheck.get("status") or "unknown")
        token_usage = postcheck.get("token_usage")
        return status, token_usage if isinstance(token_usage, dict) and token_usage else None
    if metadata:
        return "disabled", None
    return "unavailable", None


def _read_history_file() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, dict):
        records = payload.get("records", [])
    else:
        records = payload
    return [r for r in records if isinstance(r, dict)]


def _write_history_file(records: list[dict[str, Any]]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    payload = {"version": HISTORY_VERSION, "records": records}
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(HISTORY_PATH)


def build_history_record(job: JobState) -> dict[str, Any]:
    files = discover_output_files(Path(job.output_dir))
    metadata = files.get("metadata_payload") or {}
    ocr_conf = metadata.get("ocr_confidence") or {}
    duration = None
    if job.started_at and job.finished_at:
        duration = max(0.0, job.finished_at - job.started_at)
    markdown_path = files.get("markdown")
    return {
        "job_id": job.job_id,
        "input_name": job.input_name,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "duration_seconds": duration,
        "page_count": metadata.get("page_count"),
        "region_count": metadata.get("region_count"),
        "ai_check_status": job.gemini_status,
        "ocr_avg_score": ocr_conf.get("avg_score"),
        "markdown_name": markdown_path.name if markdown_path else None,
        "has_markdown": bool(markdown_path and markdown_path.exists()),
        "error": job.error,
    }


def upsert_history(job: JobState) -> None:
    record = build_history_record(job)
    with history_lock:
        records = _read_history_file()
        records = [r for r in records if r.get("job_id") != job.job_id]
        records.append(record)
        records.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        _write_history_file(records)


def _rehydrate_job_from_dir(job_dir: Path) -> JobState | None:
    output_dir = job_dir / "output"
    input_dir = job_dir / "input"
    log_path = job_dir / "job.log"
    if not output_dir.exists():
        return None
    inputs = list(input_dir.iterdir()) if input_dir.exists() else []
    input_path = inputs[0] if inputs else None
    files = discover_output_files(output_dir)
    metadata_payload = files.get("metadata_payload") or {}
    gemini_status, token_usage = summarize_gemini(metadata_payload)
    created_at = job_dir.stat().st_mtime
    if input_path and input_path.exists():
        created_at = min(created_at, input_path.stat().st_mtime)
    markdown = files.get("markdown")
    return JobState(
        job_id=job_dir.name,
        status="succeeded" if markdown else "failed",
        input_name=input_path.name if input_path else job_dir.name,
        input_path=str(input_path) if input_path else "",
        output_dir=str(output_dir),
        log_path=str(log_path),
        created_at=created_at,
        started_at=created_at,
        finished_at=log_path.stat().st_mtime if log_path.exists() else created_at,
        markdown_path=str(markdown) if markdown else None,
        metadata_path=str(files.get("metadata")) if files.get("metadata") else None,
        gemini_status=gemini_status,
        token_usage=token_usage,
    )


@app.on_event("startup")
def bootstrap_history() -> None:
    if not RUNS_DIR.exists():
        return
    with history_lock:
        records = _read_history_file()
        known_ids = {r.get("job_id") for r in records}
    rehydrated: list[JobState] = []
    for child in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
        if not child.is_dir():
            continue
        job = _rehydrate_job_from_dir(child)
        if job is None:
            continue
        with jobs_lock:
            jobs.setdefault(job.job_id, job)
        if job.job_id not in known_ids:
            rehydrated.append(job)
    for job in rehydrated:
        upsert_history(job)


def read_review_payload(review_path: Path | None) -> dict[str, Any]:
    if not review_path or not review_path.exists():
        return {}
    try:
        with review_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def reversals_path(job: JobState) -> Path:
    return Path(job.output_dir).resolve().parent / REVERSALS_FILENAME


def read_reversed_change_ids(job: JobState) -> set[int]:
    path = reversals_path(job)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    values = payload.get("reversed_change_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        return set()
    return {int(value) for value in values if isinstance(value, int) and value >= 0}


def write_reversed_change_ids(job: JobState, change_ids: set[int]) -> None:
    payload = {"reversed_change_ids": sorted(change_ids)}
    reversals_path(job).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def truncate_text(text: str, limit: int = CHANGE_SNIPPET_LIMIT) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]", True


def resolve_change_sources(job: JobState) -> tuple[Path, Path, Path, dict[str, Any]]:
    files = discover_output_files(Path(job.output_dir))
    review_path = resolve_path_in_job(job, files.get("review"))
    if not review_path or not review_path.exists():
        raise HTTPException(status_code=404, detail="AI change review not found")

    review = read_review_payload(review_path)
    source_name = review.get("source_markdown")
    checked_name = review.get("checked_markdown")

    source_path = review_path.parent / source_name if isinstance(source_name, str) and source_name else files.get("raw_markdown")
    checked_path = (
        review_path.parent / checked_name if isinstance(checked_name, str) and checked_name else files.get("checked_markdown")
    )
    source_path = resolve_path_in_job(job, source_path)
    checked_path = resolve_path_in_job(job, checked_path)

    if not source_path or not source_path.exists():
        raise HTTPException(status_code=404, detail="Source markdown for AI changes not found")
    if not checked_path or not checked_path.exists():
        raise HTTPException(status_code=404, detail="AI checked markdown not found")
    return source_path, checked_path, review_path, review


def user_markdown_path(checked_path: Path) -> Path:
    name = checked_path.name
    if name.endswith("_vlm_checked.md"):
        return checked_path.with_name(name[: -len("_vlm_checked.md")] + "_vlm_user.md")
    if name.endswith("_gemini_checked.md"):
        return checked_path.with_name(name[: -len("_gemini_checked.md")] + "_vlm_user.md")
    return checked_path.with_name(f"{checked_path.stem}_user.md")


def correction_memory_path() -> Path:
    configured = os.environ.get("OCR_CORRECTION_MEMORY_PATH")
    path = Path(configured) if configured else BASE_DIR / "conf" / CORRECTION_MEMORY_FILENAME
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def correction_rule_id(wrong: str, correct: str) -> str:
    return hashlib.sha256(f"{wrong}\0{correct}".encode("utf-8")).hexdigest()[:16]


def load_correction_memory() -> dict[str, Any]:
    path = correction_memory_path()
    if not path.exists():
        return {"version": 1, "rules": [], "blocked": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "rules": [], "blocked": []}
    if not isinstance(payload, dict):
        return {"version": 1, "rules": [], "blocked": []}
    payload.setdefault("version", 1)
    payload.setdefault("rules", [])
    payload.setdefault("blocked", [])
    if not isinstance(payload["rules"], list):
        payload["rules"] = []
    if not isinstance(payload["blocked"], list):
        payload["blocked"] = []
    return payload


def save_correction_memory(memory: dict[str, Any]) -> None:
    path = correction_memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def change_text_by_id(job: JobState, change_id: int) -> tuple[str, str] | None:
    source_path, checked_path, _, _ = resolve_change_sources(job)
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    checked_text = checked_path.read_text(encoding="utf-8", errors="replace")
    source_lines = source_text.splitlines(keepends=True)
    checked_lines = checked_text.splitlines(keepends=True)
    current_id = 0
    for tag, source_start, source_end, checked_start, checked_end in diff_opcodes(source_text, checked_text):
        if tag == "equal":
            continue
        if current_id == change_id:
            return (
                "".join(source_lines[source_start:source_end]).strip(),
                "".join(checked_lines[checked_start:checked_end]).strip(),
            )
        current_id += 1
    return None


def set_correction_memory_block(wrong: str, correct: str, blocked: bool, reason: str) -> None:
    if not wrong or not correct or wrong == correct:
        return
    memory = load_correction_memory()
    rule_id = correction_rule_id(wrong, correct)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    for rule in memory.get("rules", []):
        if isinstance(rule, dict) and rule.get("id") == rule_id:
            rule["blocked"] = blocked
            if blocked:
                rule["blocked_at"] = now
                rule["blocked_reason"] = reason
            else:
                rule.pop("blocked_at", None)
                rule.pop("blocked_reason", None)

    blocked_items = [
        item
        for item in memory.get("blocked", [])
        if not (isinstance(item, dict) and item.get("wrong") == wrong and item.get("correct") == correct)
    ]
    if blocked:
        blocked_items.append(
            {
                "id": rule_id,
                "wrong": wrong,
                "correct": correct,
                "blocked_at": now,
                "reason": reason,
            }
        )
    memory["blocked"] = blocked_items
    save_correction_memory(memory)


def diff_opcodes(source_text: str, checked_text: str) -> list[tuple[str, int, int, int, int]]:
    source_lines = source_text.splitlines(keepends=True)
    checked_lines = checked_text.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, source_lines, checked_lines, autojunk=False)
    return matcher.get_opcodes()


def build_change_rows(
    source_text: str,
    checked_text: str,
    review: dict[str, Any],
    reversed_change_ids: set[int],
) -> list[dict[str, Any]]:
    source_lines = source_text.splitlines(keepends=True)
    checked_lines = checked_text.splitlines(keepends=True)
    issues = review.get("issues")
    issues = issues if isinstance(issues, list) else []
    rows: list[dict[str, Any]] = []
    change_id = 0

    for tag, source_start, source_end, checked_start, checked_end in diff_opcodes(source_text, checked_text):
        if tag == "equal":
            continue
        original = "".join(source_lines[source_start:source_end])
        ai_output = "".join(checked_lines[checked_start:checked_end])
        original_preview, original_truncated = truncate_text(original)
        ai_preview, ai_truncated = truncate_text(ai_output)
        issue = issues[change_id] if change_id < len(issues) and isinstance(issues[change_id], dict) else {}
        rows.append(
            {
                "id": change_id,
                "kind": tag,
                "source_line": source_start + 1,
                "checked_line": checked_start + 1,
                "original": original_preview,
                "ai_output": ai_preview,
                "truncated": original_truncated or ai_truncated,
                "reversed": change_id in reversed_change_ids,
                "severity": issue.get("severity"),
                "page_number": issue.get("page_number"),
                "issue": issue.get("issue"),
                "suggestion": issue.get("suggestion"),
            }
        )
        change_id += 1
    return rows


def build_effective_markdown(source_text: str, checked_text: str, reversed_change_ids: set[int]) -> str:
    source_lines = source_text.splitlines(keepends=True)
    checked_lines = checked_text.splitlines(keepends=True)
    chunks: list[str] = []
    change_id = 0
    for tag, source_start, source_end, checked_start, checked_end in diff_opcodes(source_text, checked_text):
        if tag == "equal":
            chunks.extend(checked_lines[checked_start:checked_end])
            continue
        if change_id in reversed_change_ids:
            chunks.extend(source_lines[source_start:source_end])
        else:
            chunks.extend(checked_lines[checked_start:checked_end])
        change_id += 1
    return "".join(chunks)


def apply_markdown_reversals(job: JobState, reversed_change_ids: set[int]) -> Path:
    source_path, checked_path, _, _ = resolve_change_sources(job)
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    checked_text = checked_path.read_text(encoding="utf-8", errors="replace")
    write_reversed_change_ids(job, reversed_change_ids)

    if not reversed_change_ids:
        update_job(job.job_id, markdown_path=str(checked_path))
        return checked_path

    user_path = user_markdown_path(checked_path)
    user_path.write_text(build_effective_markdown(source_text, checked_text, reversed_change_ids), encoding="utf-8")
    update_job(job.job_id, markdown_path=str(user_path))
    return user_path


def build_changes_preview(job: JobState) -> dict[str, Any]:
    source_path, checked_path, review_path, review = resolve_change_sources(job)
    source_text = source_path.read_text(encoding="utf-8", errors="replace")
    checked_text = checked_path.read_text(encoding="utf-8", errors="replace")
    reversed_change_ids = read_reversed_change_ids(job)
    rows = build_change_rows(source_text, checked_text, review, reversed_change_ids)
    return {
        "name": review_path.name,
        "summary": review.get("summary", ""),
        "model": review.get("model"),
        "confidence": review.get("confidence"),
        "source_markdown": source_path.name,
        "checked_markdown": checked_path.name,
        "current_markdown": Path(job.markdown_path).name if job.markdown_path else checked_path.name,
        "changes": rows,
        "reversed_count": sum(1 for row in rows if row["reversed"]),
        "download_url": f"/api/jobs/{job.job_id}/download/md",
    }


def serialize_job(job: JobState) -> dict[str, Any]:
    data = asdict(job)
    if job.started_at is None:
        elapsed_seconds = None
    else:
        end = job.finished_at if job.finished_at is not None else time.time()
        elapsed_seconds = max(0.0, end - job.started_at)

    markdown_path = path_in_job(job, job.markdown_path)
    metadata_path = path_in_job(job, job.metadata_path)
    log_path = Path(job.log_path)
    files = discover_output_files(Path(job.output_dir))
    review_path = resolve_path_in_job(job, files.get("review"))
    reversed_count = len(read_reversed_change_ids(job)) if review_path else 0

    data.update(
        {
            "created_at": now_iso(job.created_at),
            "started_at": now_iso(job.started_at),
            "finished_at": now_iso(job.finished_at),
            "elapsed_seconds": elapsed_seconds,
            "markdown_name": markdown_path.name if markdown_path else None,
            "metadata_name": metadata_path.name if metadata_path else None,
            "changes_available": bool(review_path and review_path.exists()),
            "changes_name": review_path.name if review_path and review_path.exists() else None,
            "reversed_changes_count": reversed_count,
            "log_available": log_path.exists() and log_path.stat().st_size > 0,
        }
    )
    data.pop("input_path", None)
    data.pop("output_dir", None)
    data.pop("log_path", None)
    data.pop("markdown_path", None)
    data.pop("metadata_path", None)
    return data


def serialize_job_light(job: JobState) -> dict[str, Any]:
    if job.started_at is None:
        elapsed_seconds = None
    else:
        end = job.finished_at if job.finished_at is not None else time.time()
        elapsed_seconds = max(0.0, end - job.started_at)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "input_name": job.input_name,
        "created_at": now_iso(job.created_at),
        "started_at": now_iso(job.started_at),
        "finished_at": now_iso(job.finished_at),
        "elapsed_seconds": elapsed_seconds,
        "gemini_status": job.gemini_status,
        "has_markdown": bool(job.markdown_path and Path(job.markdown_path).exists()),
        "error": job.error,
    }


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text)
        handle.flush()


def run_job(job_id: str) -> None:
    job = require_job(job_id)
    input_path = Path(job.input_path)
    output_dir = Path(job.output_dir)
    log_path = Path(job.log_path)
    update_job(job_id, status="running", started_at=time.time(), error=None)

    cmd = [
        sys.executable,
        "-u",
        str(PIPELINE_PATH),
        "--inputs",
        str(input_path),
        "--output_dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    append_log(log_path, f"$ {' '.join(cmd)}\n")
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as log_handle:
            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
            returncode = process.wait()

        markdown_path, metadata_path, metadata = discover_outputs(output_dir)
        gemini_status, token_usage = summarize_gemini(metadata)
        if returncode == 0:
            status = "succeeded"
            error = None
        else:
            status = "failed"
            error = f"Pipeline exited with code {returncode}"
        update_job(
            job_id,
            status=status,
            finished_at=time.time(),
            returncode=returncode,
            error=error,
            markdown_path=str(markdown_path) if markdown_path else None,
            metadata_path=str(metadata_path) if metadata_path else None,
            gemini_status=gemini_status,
            token_usage=token_usage,
        )
        upsert_history(require_job(job_id))
    except Exception as exc:
        append_log(log_path, f"\nWeb runner error: {exc}\n")
        markdown_path, metadata_path, metadata = discover_outputs(output_dir)
        gemini_status, token_usage = summarize_gemini(metadata)
        update_job(
            job_id,
            status="failed",
            finished_at=time.time(),
            error=str(exc),
            markdown_path=str(markdown_path) if markdown_path else None,
            metadata_path=str(metadata_path) if metadata_path else None,
            gemini_status=gemini_status,
            token_usage=token_usage,
        )
        upsert_history(require_job(job_id))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = safe_filename(file.filename or "upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    job_id = uuid.uuid4().hex
    job_dir = RUNS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / filename
    with input_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    log_path = job_dir / "job.log"
    job = JobState(
        job_id=job_id,
        status="queued",
        input_name=filename,
        input_path=str(input_path),
        output_dir=str(output_dir),
        log_path=str(log_path),
        created_at=time.time(),
    )
    with jobs_lock:
        jobs[job_id] = job
    executor.submit(run_job, job_id)
    return serialize_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return serialize_job(require_job(job_id))


@app.get("/api/jobs/{job_id}/logs")
def get_logs(job_id: str, offset: int = 0) -> dict[str, Any]:
    job = require_job(job_id)
    log_path = Path(job.log_path)
    if offset < 0:
        offset = 0
    if not log_path.exists():
        return {"text": "", "offset": 0}

    size = log_path.stat().st_size
    if offset > size:
        offset = 0
    with log_path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read(LOG_READ_LIMIT)
        new_offset = handle.tell()
    return {"text": chunk.decode("utf-8", errors="replace"), "offset": new_offset}


@app.get("/api/jobs/{job_id}/download/md")
def download_markdown(job_id: str) -> FileResponse:
    job = require_job(job_id)
    markdown_path = path_in_job(job, job.markdown_path)
    if not markdown_path or not markdown_path.exists():
        raise HTTPException(status_code=404, detail="Markdown output not found")
    return FileResponse(markdown_path, media_type="text/markdown", filename=markdown_path.name)


@app.get("/api/jobs/{job_id}/preview/md")
def preview_markdown(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    markdown_path = path_in_job(job, job.markdown_path)
    if not markdown_path or not markdown_path.exists():
        raise HTTPException(status_code=404, detail="Markdown output not found")
    text, truncated = read_text_preview(markdown_path)
    return {
        "name": markdown_path.name,
        "text": text,
        "truncated": truncated,
        "download_url": f"/api/jobs/{job_id}/download/md",
    }


@app.get("/api/jobs/{job_id}/download/metadata")
def download_metadata(job_id: str) -> FileResponse:
    job = require_job(job_id)
    metadata_path = path_in_job(job, job.metadata_path)
    if not metadata_path or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata output not found")
    return FileResponse(metadata_path, media_type="application/json", filename=metadata_path.name)


@app.get("/api/jobs/{job_id}/preview/metadata")
def preview_metadata(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    metadata_path = path_in_job(job, job.metadata_path)
    if not metadata_path or not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata output not found")
    text, truncated = read_text_preview(metadata_path)
    if truncated:
        pretty_text = text
    else:
        payload = read_metadata(metadata_path)
        pretty_text = json.dumps(payload, ensure_ascii=False, indent=2) if payload else text
    return {
        "name": metadata_path.name,
        "text": pretty_text,
        "truncated": truncated,
        "download_url": f"/api/jobs/{job_id}/download/metadata",
    }


@app.get("/api/jobs/{job_id}/preview/changes")
def preview_changes(job_id: str) -> dict[str, Any]:
    return build_changes_preview(require_job(job_id))


def set_change_reversal(job_id: str, change_id: int, reversed_state: bool) -> dict[str, Any]:
    if change_id < 0:
        raise HTTPException(status_code=400, detail="Invalid change id")
    job = require_job(job_id)
    preview = build_changes_preview(job)
    valid_ids = {int(change["id"]) for change in preview["changes"]}
    if change_id not in valid_ids:
        raise HTTPException(status_code=404, detail="AI change not found")

    reversed_change_ids = read_reversed_change_ids(job)
    change_text = change_text_by_id(job, change_id)
    if reversed_state:
        reversed_change_ids.add(change_id)
        log_action = "reversed"
    else:
        reversed_change_ids.discard(change_id)
        log_action = "restored"

    current_path = apply_markdown_reversals(job, reversed_change_ids)
    if change_text:
        try:
            set_correction_memory_block(
                change_text[0],
                change_text[1],
                reversed_state,
                f"web_ui_change_{log_action}",
            )
        except OSError as exc:
            append_log(Path(job.log_path), f"Web UI: correction memory block update failed: {exc}\n")
    append_log(Path(job.log_path), f"Web UI: {log_action} AI change #{change_id}; current markdown: {current_path.name}\n")
    return build_changes_preview(require_job(job_id))


@app.post("/api/jobs/{job_id}/changes/{change_id}/reverse")
def reverse_change(job_id: str, change_id: int) -> dict[str, Any]:
    return set_change_reversal(job_id, change_id, True)


@app.post("/api/jobs/{job_id}/changes/{change_id}/apply")
def apply_change(job_id: str, change_id: int) -> dict[str, Any]:
    return set_change_reversal(job_id, change_id, False)


@app.get("/api/jobs/{job_id}/download/log")
def download_log(job_id: str) -> FileResponse:
    job = require_job(job_id)
    log_path = Path(job.log_path)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Job log not found")
    return FileResponse(log_path, media_type="text/plain", filename=f"{job_id}.log")


@app.get("/api/history")
def list_history() -> dict[str, Any]:
    with history_lock:
        records = _read_history_file()
    return {"records": records, "count": len(records)}


@app.get("/api/queue")
def list_queue() -> dict[str, Any]:
    with jobs_lock:
        all_jobs = sorted(jobs.values(), key=lambda j: j.created_at)
    return {"jobs": [serialize_job_light(j) for j in all_jobs]}
