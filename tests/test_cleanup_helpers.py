import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import web_app
from utils.ai_postcheck import (
    compact_metadata_for_gemini,
    format_token_usage as format_ai_token_usage,
    openai_compatible_base_url,
    openai_compatible_chat_url,
    parse_ai_json_response,
    run_gemini_postcheck,
)
from utils.document_analysis import add_text_risks, apply_heading_detection
from utils.pipeline_artifacts import (
    correction_rule_id,
    load_correction_memory,
    merged_markdown_output_path,
    metadata_output_path,
    ocr_raw_output_path,
    relative_output_path,
    save_correction_memory,
    vlm_checked_output_path,
    vlm_review_output_path,
)
from utils.runtime_env import env_bool, env_value, load_env_file
from utils.web_helpers import read_text_preview as read_text_preview_file, safe_filename as safe_filename_value, summarize_postcheck


class PipelineArtifactsTests(unittest.TestCase):
    def test_document_artifact_paths_share_output_folder_stem(self) -> None:
        page_output = os.path.join("output", "invoice_01", "invoice_01_0.jpg")
        markdown_path = merged_markdown_output_path(page_output)

        self.assertEqual(markdown_path, os.path.join("output", "invoice_01", "invoice_01_full.md"))
        self.assertEqual(metadata_output_path(markdown_path), os.path.join("output", "invoice_01", "invoice_01_metadata.json"))
        self.assertEqual(ocr_raw_output_path(markdown_path), os.path.join("output", "invoice_01", "invoice_01_ocr_raw.md"))
        self.assertEqual(
            vlm_checked_output_path(markdown_path),
            os.path.join("output", "invoice_01", "invoice_01_vlm_checked.md"),
        )
        self.assertEqual(
            vlm_review_output_path(markdown_path),
            os.path.join("output", "invoice_01", "invoice_01_vlm_review.json"),
        )

    def test_relative_output_path_normalizes_separators(self) -> None:
        nested_path = os.path.join("output", "invoice_01", "assets", "crop.png")
        base_dir = os.path.join("output", "invoice_01")
        self.assertEqual(relative_output_path(nested_path, base_dir), "assets/crop.png")

    def test_correction_memory_round_trip_and_invalid_payload_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_path = os.path.join(tmp_dir, "memory.json")
            payload = {"version": 1, "rules": [{"id": "abc"}], "blocked": []}

            save_correction_memory(memory_path, payload)
            self.assertEqual(load_correction_memory(memory_path, 1), payload)

            Path(memory_path).write_text("[]", encoding="utf-8")
            self.assertEqual(load_correction_memory(memory_path, 1), {"version": 1, "rules": [], "blocked": []})

    def test_correction_rule_id_is_stable(self) -> None:
        self.assertEqual(correction_rule_id("foo", "bar"), correction_rule_id("foo", "bar"))


class RuntimeEnvTests(unittest.TestCase):
    def test_load_env_file_populates_missing_values_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = os.path.join(tmp_dir, ".env")
            Path(env_path).write_text("NEW_VALUE=abc\nKEEP_VALUE=from_file\n", encoding="utf-8")
            with patch.dict(os.environ, {"KEEP_VALUE": "existing"}, clear=False):
                load_env_file(env_path)
                self.assertEqual(os.environ["NEW_VALUE"], "abc")
                self.assertEqual(os.environ["KEEP_VALUE"], "existing")

    def test_env_bool_and_env_value(self) -> None:
        with patch.dict(os.environ, {"FEATURE_FLAG": "yes", "APP_MODE": "prod"}, clear=False):
            self.assertTrue(env_bool("FEATURE_FLAG"))
            self.assertEqual(env_value("APP_MODE", "dev"), "prod")
            self.assertFalse(env_bool("MISSING_FLAG"))


class WebAppCorrectionMemoryTests(unittest.TestCase):
    def test_web_app_correction_memory_wrappers_use_shared_storage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_path = os.path.join(tmp_dir, "web_memory.json")
            payload = {"version": 1, "rules": [{"id": "r1", "wrong": "a", "correct": "b"}], "blocked": []}

            with patch.dict(os.environ, {"OCR_CORRECTION_MEMORY_PATH": memory_path}, clear=False):
                web_app.save_correction_memory(payload)
                self.assertEqual(web_app.load_correction_memory(), payload)
                self.assertEqual(web_app.correction_rule_id("a", "b"), correction_rule_id("a", "b"))


class WebHelpersTests(unittest.TestCase):
    def test_safe_filename_strips_path_and_unsafe_chars(self) -> None:
        self.assertEqual(safe_filename_value("../bad<>name?.pdf"), "badname.pdf")
        self.assertEqual(web_app.safe_filename("../bad<>name?.pdf"), "badname.pdf")

    def test_read_text_preview_reports_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "preview.txt"
            text_path.write_text("abcdef", encoding="utf-8")
            preview, truncated = read_text_preview_file(text_path, 4)
            self.assertEqual(preview, "abcd")
            self.assertTrue(truncated)

    def test_summarize_postcheck_matches_web_app_wrapper(self) -> None:
        metadata = {"ai_postcheck": {"status": "ok", "token_usage": {"total_tokens": 12}}}
        self.assertEqual(summarize_postcheck(metadata), ("ok", {"total_tokens": 12}))
        self.assertEqual(web_app.summarize_gemini(metadata), ("ok", {"total_tokens": 12}))


class DocumentAnalysisTests(unittest.TestCase):
    def test_apply_heading_detection_promotes_numbered_section(self) -> None:
        text, headings = apply_heading_detection("1. Phạm vi áp dụng", 1, {}, enabled=True, max_chars=180)
        self.assertEqual(text, "### 1. Phạm vi áp dụng")
        self.assertEqual(headings[0]["reason"], "numbered_section")

    def test_add_text_risks_flags_sensitive_and_checkbox_patterns(self) -> None:
        page_metadata = {"page_number": 1, "vlm_risks": []}
        add_text_risks(page_metadata, "Số tiền 1.000 đồng [x]", [0, 0, 500, 500], 0.00025)
        categories = {risk["category"] for risk in page_metadata["vlm_risks"]}
        self.assertIn("sensitive_value", categories)
        self.assertIn("checkbox_or_selection", categories)


class AIPostcheckTests(unittest.TestCase):
    def test_compact_metadata_for_gemini_keeps_review_relevant_fields(self) -> None:
        metadata = {
            "input": "invoice.pdf",
            "output_markdown": "invoice_full.md",
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "width": 100,
                    "height": 200,
                    "ocr_confidence": {"avg": 0.9},
                    "layout_confidence": {"avg": 0.8},
                    "vlm_risks": [{"risk_id": "r1"}],
                    "regions": [
                        {
                            "content_type": "text",
                            "layout_type": "title",
                            "bbox": [0, 0, 10, 10],
                            "ocr_confidence": {"avg": 0.95},
                            "asset_path": "assets/title.png",
                            "ignored_field": "drop-me",
                        }
                    ],
                }
            ],
        }
        compact = compact_metadata_for_gemini(metadata)
        self.assertEqual(compact["pages"][0]["regions"][0]["asset_path"], "assets/title.png")
        self.assertNotIn("ignored_field", compact["pages"][0]["regions"][0])

    def test_parse_ai_json_response_supports_fenced_content(self) -> None:
        payload = parse_ai_json_response("```json\n{\"checked_markdown\": \"ok\"}\n```")
        self.assertEqual(payload["checked_markdown"], "ok")

    def test_openai_chat_url_appends_suffix_once(self) -> None:
        class Args:
            ai_base_url = "https://example.com/v1"

        self.assertEqual(openai_compatible_base_url(Args()), "https://example.com/v1")
        self.assertEqual(openai_compatible_chat_url(Args()), "https://example.com/v1/chat/completions")

    def test_format_ai_token_usage_compacts_known_fields(self) -> None:
        formatted = format_ai_token_usage({"prompt_token_count": 10, "total_token_count": 20})
        self.assertEqual(formatted, "prompt=10, total=20")

    def test_run_gemini_postcheck_uses_openai_compatible_response(self) -> None:
        class Args:
            gemini_system_prompt = None
            gemini_max_pages = 0
            gemini_image_max_side = 1600
            gemini_model = "gpt-test"
            gemini_timeout = 30
            ai_response_format = "json_object"
            ai_base_url = "https://example.com/v1"

        fake_response = unittest.mock.Mock()
        fake_response.json.return_value = {
            "choices": [{"message": {"content": "{\"checked_markdown\": \"ok\", \"summary\": \"done\", \"issues\": [], \"image_notes\": [], \"vlm_findings\": [], \"confidence\": 0.9}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        fake_response.raise_for_status.return_value = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_path = Path(tmp_dir) / "prompt.txt"
            prompt_path.write_text("system prompt", encoding="utf-8")
            with patch.dict(os.environ, {"AI_INCLUDE_IMAGES": "0"}, clear=False):
                with patch("utils.ai_postcheck.requests.post", return_value=fake_response) as post_mock:
                    result = run_gemini_postcheck("raw markdown", {"vlm_risks": [], "pages": []}, [], Args(), "secret", str(prompt_path))

        self.assertEqual(result["checked_markdown"], "ok")
        self.assertEqual(result["_token_usage"]["total_token_count"], 12)
        post_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
