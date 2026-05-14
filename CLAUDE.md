# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepDoc + VietOCR is a Vietnamese document OCR pipeline extracted from the [RAGFlow](https://github.com/infiniflow/ragflow) project. It replaces DeepDoc's default PaddleOCR text recognizer with VietOCR (vgg_seq2seq) for better Vietnamese text accuracy, while keeping PaddleOCR's text detector and YOLOv10-based layout/table-structure recognizers — all running as ONNX models on CPU.

## Setup

```bash
pip install -r requirements.txt
```

ONNX model files (`det.onnx`, `layout.onnx`, `tsr.onnx`) live in `onnx/`. If absent, they are auto-downloaded from HuggingFace `InfiniFlow/deepdoc`. On Windows with HuggingFace access issues, set:

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

VietOCR weights must be placed at `vietocr/weight/vgg_seq2seq.pth` (default) or `vgg_transformer.pth`.

## Entry Point Commands

```bash
# OCR only — outputs annotated image + .txt per input
python t_ocr.py --inputs=<path_to_images_or_pdfs> --output_dir=./ocr_outputs

# Layout recognition — outputs annotated image per input
python t_recognizer.py --inputs=<path> --threshold=0.2 --mode=layout --output_dir=./layouts_outputs

# Table structure recognition — outputs annotated image + .md per input
python t_recognizer.py --inputs=<path> --threshold=0.2 --mode=tsr --output_dir=./layouts_outputs

# Full pipeline (layout → table markdown + OCR remaining areas → unified .md)
python full_pipeline.py --inputs=<path> --output_dir=./table_markdown_outputs --threshold=0.5
```

**Note:** All scripts redirect `sys.stdout`/`sys.stderr` to log files in `log/` at startup. You will not see output on the console — check `log/t_ocr.log`, `log/t_recognizer.log`, or `log/full_pipeline.log` instead.

## Architecture

### Processing Pipeline

```
Input image/PDF
      │
      ▼
TextDetector (det.onnx — PaddleOCR DBNet)
      │  quadrilateral bounding boxes
      ▼
TextRecognizer (VietOCR vgg_seq2seq.pth)    ← swappable, see below
      │  (text, confidence) per box
      ▼
LayoutRecognizer4YOLOv10 (layout.onnx)      ← optional, used in full_pipeline
      │  10 region types: Text, Title, Figure, Table, etc.
      ▼
TableStructureRecognizer (tsr.onnx)         ← optional, used for table regions
      │  5 structure types: column, row, header, projected header, spanning cell
      ▼
Output: annotated image / .txt / .md
```

### Module Responsibilities

| File | Role |
|------|------|
| `module/ocr.py` | `OCR` class: combines `TextDetector` + VietOCR `TextRecognizer`; model caching via `loaded_models` global |
| `module/ocr_onnx.py` | Drop-in `OCR` replacement using VietOCR exported as ONNX (`cnn.onnx`, `encoder.onnx`, `decoder.onnx`) — faster, slightly less accurate |
| `module/recognizer.py` | Base `Recognizer` class with ONNX inference, bbox sorting/overlap utilities shared by layout and TSR |
| `module/layout_recognizer.py` | `LayoutRecognizer4YOLOv10` (used by default via `__init__.py`) and original `LayoutRecognizer` |
| `module/table_structure_recognizer.py` | `TableStructureRecognizer` + `construct_table()` for markdown output |
| `module/operators.py` | Image preprocessing transforms (resize, normalize, etc.) |
| `module/postprocess.py` | `DBPostProcess` for converting text detection heatmaps to bounding boxes |
| `module/seeit.py` | `draw_box()` for visualizing detection results |
| `utils/file_utils.py` | `get_project_base_directory()`, `traversal_files()`, config loaders |
| `utils/settings.py` | `PARALLEL_DEVICES` (GPU count for multi-GPU) |
| `vietocr/` | VietOCR library (locally bundled) |

### Switching the OCR Backend

In `t_ocr.py`, `t_recognizer.py`, and `full_pipeline.py`, swap the import:

```python
# Default (VietOCR PyTorch — recommended)
from module.ocr import OCR

# ONNX VietOCR (faster, slightly lower accuracy)
from module.ocr_onnx import OCR
```

To switch from vgg_seq2seq to vgg_transformer, edit `module/ocr.py` `TextRecognizer.__init__`:

```python
# Default
config = Cfg.load_config_from_name('vgg_seq2seq')
config['weights'] = r"vietocr\weight\vgg_seq2seq.pth"

# Slower, not recommended
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = r"vietocr\weight\vgg_transformer.pth"
```

### ONNX Model Loading

`module/ocr.py:load_model()` caches sessions in the module-level `loaded_models` dict keyed by `(model_path + device_id)`. It auto-selects `CUDAExecutionProvider` when a GPU is available, otherwise `CPUExecutionProvider`. GPU memory is capped at 512 MB per device.

### Box Coordinate Convention

Detection results throughout the pipeline use two formats:
- **ONNX model output**: `{"type", "bbox": [x0, y0, x1, y1], "score"}`
- **Internal pipeline format**: `{"x0", "x1", "top", "bottom", "text", "layout_type", "page_number"}`

`LayoutRecognizer4YOLOv10` uses a custom `preprocess`/`postprocess` (letterbox padding + NMS at IoU 0.45) that differs from the base `Recognizer` class.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **deepdoc_vietocr** (2144 symbols, 4028 relationships, 132 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/deepdoc_vietocr/context` | Codebase overview, check index freshness |
| `gitnexus://repo/deepdoc_vietocr/clusters` | All functional areas |
| `gitnexus://repo/deepdoc_vietocr/processes` | All execution flows |
| `gitnexus://repo/deepdoc_vietocr/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
