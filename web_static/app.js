// ── DOM refs ──────────────────────────────────────────
const fileInput = document.querySelector("#fileInput");
const chooseButton = document.querySelector("#chooseButton");
const startButton = document.querySelector("#startButton");
const clearLogButton = document.querySelector("#clearLogButton");
const dropzone = document.querySelector("#dropzone");
const fileName = document.querySelector("#fileName");
const fileMeta = document.querySelector("#fileMeta");
const message = document.querySelector("#message");
const statusBadge = document.querySelector("#statusBadge");
const logMeta = document.querySelector("#logMeta");
const logOutput = document.querySelector("#logOutput");
const previewModal = document.querySelector("#previewModal");
const previewKind = document.querySelector("#previewKind");
const previewTitle = document.querySelector("#previewTitle");
const previewNotice = document.querySelector("#previewNotice");
const previewBody = document.querySelector("#previewBody");
const changesBody = document.querySelector("#changesBody");
const previewDownload = document.querySelector("#previewDownload");
const closePreview = document.querySelector("#closePreview");
const queueList = document.querySelector("#queueList");

// ── State ─────────────────────────────────────────────
const finalStatuses = new Set(["succeeded", "failed"]);
let selectedFiles = [];
let activeJobIds = [];
let activeJobId = null;      // tracks which job the preview modal is showing
let logJobId = null;
let userPinnedJobId = null;
let logOffset = 0;
let queueTimer = null;
let logTimer = null;

// ── Utilities ─────────────────────────────────────────
function setMessage(text, tone = "muted") {
  message.textContent = text;
  message.dataset.tone = tone;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatElapsed(seconds) {
  if (!Number.isFinite(seconds)) return "N/A";
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function formatStatus(status) {
  if (!status) return "Idle";
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function formatTokens(tokenUsage) {
  if (!tokenUsage || Object.keys(tokenUsage).length === 0) return "N/A";
  const fields = [
    ["prompt", "prompt_token_count"],
    ["cached", "cached_content_token_count"],
    ["output", "candidates_token_count"],
    ["thinking", "thoughts_token_count"],
    ["tool", "tool_use_prompt_token_count"],
    ["total", "total_token_count"],
  ];
  const parts = fields
    .filter(([, key]) => tokenUsage[key] !== undefined && tokenUsage[key] !== null)
    .map(([label, key]) => `${label}=${tokenUsage[key]}`);
  return parts.length ? parts.join(", ") : "N/A";
}

function setDownload(link, enabled, href) {
  if (enabled) {
    link.href = href;
    link.classList.remove("disabled");
    link.setAttribute("aria-disabled", "false");
    link.removeAttribute("tabindex");
  } else {
    link.href = "#";
    link.classList.add("disabled");
    link.setAttribute("aria-disabled", "true");
    link.setAttribute("tabindex", "-1");
  }
}

function statusClassFor(status) {
  if (status === "succeeded") return "status-badge succeeded";
  if (status === "failed") return "status-badge failed";
  if (status === "running") return "status-badge running";
  if (status === "queued") return "status-badge queued";
  return "status-badge";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
}

function formatTimestamp(epoch) {
  if (!Number.isFinite(epoch)) return "—";
  return new Date(epoch * 1000).toLocaleString();
}

// ── File selection ─────────────────────────────────────
function setSelectedFiles(files) {
  selectedFiles = files ? Array.from(files) : [];
  if (selectedFiles.length === 0) {
    fileName.textContent = "Drop files here";
    fileMeta.textContent = "or choose from disk";
    startButton.disabled = true;
  } else if (selectedFiles.length === 1) {
    fileName.textContent = selectedFiles[0].name;
    fileMeta.textContent = formatBytes(selectedFiles[0].size);
    startButton.disabled = false;
  } else {
    const total = selectedFiles.reduce((s, f) => s + f.size, 0);
    fileName.textContent = `${selectedFiles.length} files đã chọn`;
    fileMeta.textContent = formatBytes(total);
    startButton.disabled = false;
  }
}

// ── Job creation ───────────────────────────────────────
async function createJobs() {
  if (!selectedFiles.length) return;
  startButton.disabled = true;
  setMessage(`Đang tải lên ${selectedFiles.length} file...`);

  for (const file of selectedFiles) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/jobs", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      setMessage(`Lỗi ${file.name}: ${err.detail || res.statusText}`, "error");
      continue;
    }
    const job = await res.json();
    activeJobIds.push(job.job_id);
  }

  setMessage("");
  setSelectedFiles([]);
  fileInput.value = "";
  startQueuePolling();
}

// ── Queue polling ──────────────────────────────────────
function stopPolling() {
  clearInterval(queueTimer);
  clearInterval(logTimer);
  queueTimer = null;
  logTimer = null;
}

function startQueuePolling() {
  stopPolling();
  queueTimer = window.setInterval(() => {
    pollQueue().catch(() => {});
  }, 1500);
  logTimer = window.setInterval(() => {
    if (logJobId) pollLogs().catch(() => {});
  }, 800);
}

async function pollQueue() {
  const res = await fetch("/api/queue");
  if (!res.ok) return;
  const data = await res.json();
  const allJobs = data.jobs || [];
  const sessionJobs = allJobs.filter(j => activeJobIds.includes(j.job_id));
  renderQueue(sessionJobs);
  autoSelectLogJob(sessionJobs);
  updateTopbarStatus(sessionJobs);

  const allDone = sessionJobs.every(j => finalStatuses.has(j.status));
  if (allDone && sessionJobs.length > 0) {
    clearInterval(queueTimer);
    queueTimer = null;
  }
}

async function pollLogs() {
  if (!logJobId) return;
  const response = await fetch(`/api/jobs/${logJobId}/logs?offset=${logOffset}`);
  if (!response.ok) return;
  const payload = await response.json();
  if (payload.text) {
    const shouldStick =
      logOutput.scrollTop + logOutput.clientHeight >= logOutput.scrollHeight - 24;
    logOutput.textContent += payload.text;
    if (shouldStick) {
      logOutput.scrollTop = logOutput.scrollHeight;
    }
  }
  logOffset = payload.offset;
}

function autoSelectLogJob(sessionJobs) {
  if (userPinnedJobId) {
    const stillHere = sessionJobs.find(j => j.job_id === userPinnedJobId);
    if (stillHere) { logJobId = userPinnedJobId; return; }
    userPinnedJobId = null;
  }
  const running = sessionJobs.find(j => j.status === "running");
  const target = running || sessionJobs.filter(j => j.status === "queued").at(-1) || sessionJobs.at(-1);
  if (target && logJobId !== target.job_id) {
    logJobId = target.job_id;
    logOffset = 0;
    logOutput.textContent = "";
    logMeta.textContent = target.input_name || logJobId;
  }
}

function renderQueue(sessionJobs) {
  const clearBtn = document.querySelector("#clearDoneButton");
  const queueMeta = document.querySelector("#queueMeta");
  if (!sessionJobs.length) {
    queueList.innerHTML = '<p class="queue-empty">Chưa có file nào được nạp.</p>';
    if (queueMeta) queueMeta.textContent = "Không có job nào";
    if (clearBtn) clearBtn.disabled = true;
    return;
  }

  const running = sessionJobs.filter(j => j.status === "running").length;
  const queued = sessionJobs.filter(j => j.status === "queued").length;
  const done = sessionJobs.filter(j => finalStatuses.has(j.status)).length;
  const parts = [];
  if (running) parts.push(`${running} đang chạy`);
  if (queued) parts.push(`${queued} chờ`);
  if (done) parts.push(`${done} xong`);
  if (queueMeta) queueMeta.textContent = parts.join(", ") || "Không có job nào";
  if (clearBtn) clearBtn.disabled = done === 0;

  queueList.innerHTML = "";
  for (const job of sessionJobs) {
    const row = document.createElement("div");
    row.className = `queue-row${logJobId === job.job_id ? " active" : ""}`;
    row.dataset.jobId = job.job_id;
    const elapsed = Number.isFinite(job.elapsed_seconds) ? formatElapsed(job.elapsed_seconds) : "";
    const ai = job.gemini_status && job.gemini_status !== "unavailable" && job.gemini_status !== "disabled"
      ? `AI: ${job.gemini_status}` : "";
    const canPreview = job.has_markdown;
    const actionsHtml = finalStatuses.has(job.status) ? `
      <div class="queue-row-actions">
        <button type="button" class="secondary" data-preview-job="${escapeHtml(job.job_id)}" ${canPreview ? "" : "disabled"}>Preview</button>
        <a class="download ${canPreview ? "" : "disabled"}" href="/api/jobs/${escapeHtml(job.job_id)}/download/md" ${canPreview ? "" : 'aria-disabled="true" tabindex="-1"'}>Tải MD</a>
      </div>` : "";
    row.innerHTML = `
      <div class="queue-row-title">
        <strong>${escapeHtml(job.input_name || job.job_id)}</strong>
        <span class="${statusClassFor(job.status)}">${formatStatus(job.status)}</span>
      </div>
      <div class="queue-row-meta">
        ${elapsed ? `<span>${elapsed}</span>` : ""}
        ${ai ? `<span>${escapeHtml(ai)}</span>` : ""}
        ${job.error ? `<span style="color:var(--danger)">${escapeHtml(job.error)}</span>` : ""}
      </div>
      ${actionsHtml}
    `;
    queueList.appendChild(row);
  }

  if (logJobId) {
    const logJob = sessionJobs.find(j => j.job_id === logJobId);
    if (logJob) logMeta.textContent = logJob.input_name || logJobId;
  }
}

function updateTopbarStatus(sessionJobs) {
  const running = sessionJobs.find(j => j.status === "running");
  const queued = sessionJobs.some(j => j.status === "queued");
  if (running) {
    statusBadge.textContent = "Running";
    statusBadge.className = "status-badge running";
  } else if (queued) {
    statusBadge.textContent = "Queued";
    statusBadge.className = "status-badge queued";
  } else if (sessionJobs.length && sessionJobs.every(j => finalStatuses.has(j.status))) {
    const anyFailed = sessionJobs.some(j => j.status === "failed");
    statusBadge.textContent = anyFailed ? "Failed" : "Succeeded";
    statusBadge.className = `status-badge ${anyFailed ? "failed" : "succeeded"}`;
  } else if (!sessionJobs.length) {
    statusBadge.textContent = "Idle";
    statusBadge.className = "status-badge";
  }
}

// ── Preview modal ──────────────────────────────────────
function showPreviewModal() {
  previewModal.classList.remove("hidden");
  closePreview.focus();
}

function hidePreviewModal() {
  previewModal.classList.add("hidden");
}

async function openPreview(kind) {
  if (!activeJobId) return;
  const label = kind === "metadata" ? "Metadata" : "Markdown";
  previewKind.textContent = "Preview";
  previewTitle.textContent = label;
  previewBody.classList.remove("hidden");
  changesBody.classList.add("hidden");
  changesBody.replaceChildren();
  previewBody.textContent = "Loading...";
  previewNotice.textContent = "";
  previewNotice.classList.remove("visible");
  previewDownload.textContent = "Download";
  setDownload(previewDownload, false, "#");
  showPreviewModal();

  const response = await fetch(`/api/jobs/${activeJobId}/preview/${kind}`);
  if (!response.ok) {
    let detail = `Could not load ${label.toLowerCase()} preview`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }

  const payload = await response.json();
  previewTitle.textContent = payload.name || label;
  previewBody.textContent = payload.text || "";
  setDownload(previewDownload, Boolean(payload.download_url), payload.download_url || "#");
  if (payload.truncated) {
    previewNotice.textContent = "Preview is truncated. Download the file to view the full content.";
    previewNotice.classList.add("visible");
  }
}

function textOrDash(value) {
  if (value === undefined || value === null || value === "") return "N/A";
  return String(value);
}

function appendTextCell(row, text, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.textContent = textOrDash(text);
  row.appendChild(cell);
  return cell;
}

function appendSnippetCell(row, text) {
  const cell = document.createElement("td");
  cell.className = "snippet-cell";
  const block = document.createElement("pre");
  block.textContent = text || "N/A";
  cell.appendChild(block);
  row.appendChild(cell);
  return cell;
}

function renderChanges(payload) {
  previewKind.textContent = "AI Changes";
  previewTitle.textContent = payload.name || "AI Changes";
  previewBody.classList.add("hidden");
  changesBody.classList.remove("hidden");
  changesBody.replaceChildren();
  previewNotice.textContent = "";
  previewNotice.classList.remove("visible");
  previewDownload.textContent = "Download Markdown";
  setDownload(previewDownload, Boolean(payload.download_url), payload.download_url || "#");

  const summary = document.createElement("div");
  summary.className = "changes-summary";
  const meta = document.createElement("div");
  meta.className = "changes-meta";
  [
    ["Model", payload.model],
    ["Source", payload.source_markdown],
    ["AI checked", payload.checked_markdown],
    ["Current", payload.current_markdown],
    ["Reversed", payload.reversed_count || 0],
  ].forEach(([label, value]) => {
    const item = document.createElement("span");
    item.textContent = `${label}: ${textOrDash(value)}`;
    meta.appendChild(item);
  });
  summary.appendChild(meta);
  if (payload.summary) {
    const note = document.createElement("p");
    note.textContent = payload.summary;
    summary.appendChild(note);
  }
  changesBody.appendChild(summary);

  if (!payload.changes || payload.changes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "changes-empty";
    empty.textContent = "No text diff was found between raw Markdown and AI checked Markdown.";
    changesBody.appendChild(empty);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "changes-table-wrap";
  const table = document.createElement("table");
  table.className = "changes-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Status", "Page", "Severity", "Issue", "Suggestion", "Original", "AI Output", "Action"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  });
  head.appendChild(headRow);
  table.appendChild(head);

  const body = document.createElement("tbody");
  payload.changes.forEach((change) => {
    const row = document.createElement("tr");
    row.className = change.reversed ? "is-reversed" : "";

    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = `change-status ${change.reversed ? "reversed" : "applied"}`;
    status.textContent = change.reversed ? "Reversed" : "Applied";
    statusCell.appendChild(status);
    row.appendChild(statusCell);

    appendTextCell(row, change.page_number);
    appendTextCell(row, change.severity);
    appendTextCell(row, change.issue, "issue-cell");
    appendTextCell(row, change.suggestion, "issue-cell");
    appendSnippetCell(row, change.original);
    appendSnippetCell(row, change.ai_output);

    const actionCell = document.createElement("td");
    const action = document.createElement("button");
    action.type = "button";
    action.className = change.reversed ? "secondary compact-action" : "compact-action";
    action.dataset.changeId = String(change.id);
    action.dataset.action = change.reversed ? "apply" : "reverse";
    action.textContent = change.reversed ? "Apply AI" : "Reverse";
    actionCell.appendChild(action);
    row.appendChild(actionCell);

    body.appendChild(row);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  changesBody.appendChild(wrap);

  if (payload.changes.some((change) => change.truncated)) {
    previewNotice.textContent = "Some long diff snippets are truncated in the table. Download Markdown to inspect the full output.";
    previewNotice.classList.add("visible");
  }
}

async function openChanges() {
  if (!activeJobId) return;
  previewKind.textContent = "AI Changes";
  previewTitle.textContent = "AI Changes";
  previewBody.classList.add("hidden");
  changesBody.classList.remove("hidden");
  changesBody.textContent = "Loading...";
  previewNotice.textContent = "";
  previewNotice.classList.remove("visible");
  previewDownload.textContent = "Download Markdown";
  setDownload(previewDownload, false, "#");
  showPreviewModal();

  const response = await fetch(`/api/jobs/${activeJobId}/preview/changes`);
  if (!response.ok) {
    let detail = "Could not load AI changes";
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  renderChanges(await response.json());
}

async function setChangeAction(changeId, action, button) {
  if (!activeJobId) return;
  button.disabled = true;
  const response = await fetch(`/api/jobs/${activeJobId}/changes/${changeId}/${action}`, {
    method: "POST",
  });
  if (!response.ok) {
    let detail = "Could not update AI change";
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch (_) {
      detail = response.statusText || detail;
    }
    throw new Error(detail);
  }
  renderChanges(await response.json());
  pollQueue().catch(() => {});
  if (logJobId) pollLogs().catch(() => {});
}

// ── Input events ───────────────────────────────────────
chooseButton.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  setSelectedFiles(fileInput.files && fileInput.files.length > 0 ? fileInput.files : null);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer.files;
  if (!files || !files.length) return;
  fileInput.files = files;
  setSelectedFiles(files);
});

startButton.addEventListener("click", () => {
  createJobs().catch((error) => {
    startButton.disabled = !selectedFiles.length;
    setMessage(error.message, "error");
  });
});

// ── Preview modal events ───────────────────────────────
changesBody.addEventListener("click", (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const button = target ? target.closest("button[data-change-id]") : null;
  if (!button) return;
  const changeId = button.dataset.changeId;
  const action = button.dataset.action;
  setChangeAction(changeId, action, button).catch((error) => {
    previewNotice.textContent = error.message;
    previewNotice.classList.add("visible");
    button.disabled = false;
  });
});

closePreview.addEventListener("click", hidePreviewModal);

previewModal.addEventListener("click", (event) => {
  if (event.target === previewModal) hidePreviewModal();
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !previewModal.classList.contains("hidden")) {
    hidePreviewModal();
  }
});

clearLogButton.addEventListener("click", () => {
  logOutput.textContent = "";
});

// ── Queue panel events ─────────────────────────────────
queueList?.addEventListener("click", (ev) => {
  const previewBtn = ev.target.closest("[data-preview-job]");
  if (previewBtn && !previewBtn.disabled) {
    activeJobId = previewBtn.dataset.previewJob;
    openPreview("md").catch((err) => { previewBody.textContent = err.message; });
    return;
  }
  const row = ev.target.closest(".queue-row");
  if (!row) return;
  userPinnedJobId = row.dataset.jobId;
  logJobId = row.dataset.jobId;
  logOffset = 0;
  logOutput.textContent = "";
  const jobName = row.querySelector(".queue-row-title strong")?.textContent;
  if (jobName) logMeta.textContent = jobName;
  pollLogs().catch(() => {});
  document.querySelectorAll(".queue-row").forEach(r =>
    r.classList.toggle("active", r.dataset.jobId === userPinnedJobId)
  );
});

document.querySelector("#clearDoneButton")?.addEventListener("click", () => {
  const doneIds = new Set();
  document.querySelectorAll(".queue-row").forEach(r => {
    const badge = r.querySelector(".status-badge");
    if (badge && (badge.classList.contains("succeeded") || badge.classList.contains("failed"))) {
      doneIds.add(r.dataset.jobId);
    }
  });
  activeJobIds = activeJobIds.filter(id => !doneIds.has(id));
  if (doneIds.has(logJobId)) { logJobId = null; logOffset = 0; }
  if (doneIds.has(userPinnedJobId)) userPinnedJobId = null;
  pollQueue().catch(() => {});
});

// ── History drawer ─────────────────────────────────────
const historyToggle = document.querySelector("#historyToggle");
const historyClose = document.querySelector("#historyClose");
const historyRefresh = document.querySelector("#historyRefresh");
const historyDrawer = document.querySelector("#historyDrawer");
const historyBackdrop = document.querySelector("#historyBackdrop");
const historyList = document.querySelector("#historyList");
const historyMeta = document.querySelector("#historyMeta");

async function loadHistory() {
  historyMeta.textContent = "Đang tải...";
  historyList.innerHTML = "";
  try {
    const res = await fetch("/api/history");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderHistory(data.records || []);
  } catch (err) {
    historyMeta.textContent = `Lỗi: ${err.message}`;
  }
}

function renderHistory(records) {
  historyMeta.textContent = `${records.length} lần chạy`;
  if (!records.length) {
    historyList.innerHTML = '<p class="history-empty">Chưa có lịch sử OCR.</p>';
    return;
  }
  historyList.innerHTML = "";
  for (const r of records) {
    const card = document.createElement("article");
    card.className = "history-card";
    const elapsed = Number.isFinite(r.duration_seconds) ? formatElapsed(r.duration_seconds) : "—";
    const pages = r.page_count != null ? `${r.page_count} trang` : "";
    const ai = r.ai_check_status && r.ai_check_status !== "unavailable" ? `AI: ${r.ai_check_status}` : "";
    const canUse = r.has_markdown;
    card.innerHTML = `
      <div class="history-card-title">
        <strong>${escapeHtml(r.input_name || r.job_id)}</strong>
        <span class="${statusClassFor(r.status)}">${formatStatus(r.status)}</span>
      </div>
      <div class="history-card-meta">
        <span>${formatTimestamp(r.created_at)}</span>
        <span>${elapsed}</span>
        ${pages ? `<span>${escapeHtml(pages)}</span>` : ""}
        ${ai ? `<span>${escapeHtml(ai)}</span>` : ""}
      </div>
      <div class="history-card-actions">
        <button type="button" class="secondary" data-action="preview" data-id="${escapeHtml(r.job_id)}" ${canUse ? "" : "disabled"}>Preview</button>
        <a class="download ${canUse ? "" : "disabled"}" href="/api/jobs/${escapeHtml(r.job_id)}/download/md" ${canUse ? "" : 'aria-disabled="true" tabindex="-1"'}>Tải MD</a>
      </div>
    `;
    historyList.appendChild(card);
  }
}

function openHistoryDrawer() {
  historyDrawer.classList.remove("hidden");
  historyBackdrop.classList.remove("hidden");
  historyDrawer.setAttribute("aria-hidden", "false");
  loadHistory();
}

function closeHistoryDrawer() {
  historyDrawer.classList.add("hidden");
  historyBackdrop.classList.add("hidden");
  historyDrawer.setAttribute("aria-hidden", "true");
}

historyToggle?.addEventListener("click", openHistoryDrawer);
historyClose?.addEventListener("click", closeHistoryDrawer);
historyBackdrop?.addEventListener("click", closeHistoryDrawer);
historyRefresh?.addEventListener("click", loadHistory);

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !historyDrawer.classList.contains("hidden")) {
    closeHistoryDrawer();
  }
});

historyList?.addEventListener("click", (ev) => {
  const target = ev.target.closest("[data-action='preview']");
  if (!target || target.disabled) return;
  const jobId = target.dataset.id;
  if (!jobId) return;
  activeJobId = jobId;
  openPreview("md").catch((error) => {
    previewBody.textContent = error.message;
    setDownload(previewDownload, false, "#");
  });
});
