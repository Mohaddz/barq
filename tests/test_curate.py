"""Offline curation gates exercised against real, tiny prepare artifacts."""

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq
import yaml

import test_review as fixtures


def chat(prompt, answer):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


class CurateIntegrationTests(unittest.TestCase):
    def setUp(self):
        from barq import curate
        self.curate = curate
        # Composition shares prepare fixtures and network guards without rerunning
        # the review test class through inheritance or module-level class imports.
        self.fixture = fixtures.ReviewIntegrationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config_path = self.root / "curation.yaml"
        self.config = {
            "schema_version": 1, "seed": 42, "batch_size": 2, "sample_per_group": 2,
            "repair_flags": ["underlying_text_changed", "no_diacritics_added", "invalid_sentiment_label"],
            "review_sources": {"news_commentary": "translation_alignment_pending"},
            "review_tasks": {"summarization": "summary_grounding_pending"},
        }
        self.write_config(self.config)

    def write_config(self, config):
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def execute(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            output, report = self.curate.run(
                self.fixture.input_dir, config_path=self.config_path, **kwargs,
            )
        return {
            "output": output, "report": report,
            "candidates": pq.read_table(output / "candidates.parquet").to_pylist(),
            "decisions": pq.read_table(output / "decisions.parquet").to_pylist(),
            "samples": fixtures.read_jsonl(report / "review_samples.jsonl"),
            "manifest": json.loads((report / "manifest.json").read_text(encoding="utf-8")),
        }

    def test_gate_priority_holds_preservation_and_evaluation_isolation(self):
        rows = [
            fixtures.candidate(0, task="grammar", messages=chat(
                "صحّح: الطالبان ذهب إلى المدرسه", "ذَهَبَ الطَّالِبَانِ إِلَى الْمَدْرَسَةِ."),
                metadata={"note": "احتفظ بهذا الحقل"}),
            fixtures.candidate(1, source="diacritization", task="diacritization", messages=chat(
                "أضف التشكيل إلى النص التالي:\n\nكتب الطالب", "كَتَبَ الطَّالِبُ")),
            fixtures.candidate(2, source="diacritization", task="diacritization", messages=chat(
                "أضف التشكيل إلى النص التالي:\n\nكتب الطالب", "كتب الطالب")),
            fixtures.candidate(3, messages=chat("مرحبا", "")),
            fixtures.candidate(4, source="news_commentary", task="translation", messages=chat("Hello", "مرحبا")),
            fixtures.candidate(5, task="summarization", messages=chat("لخّص: عاد الطفل إلى البيت.", "عاد الطفل.")),
            fixtures.candidate(6, task="instruction_following", messages=chat(
                "Write a Python function.", "```python\ndef broken(:\n```")),
            fixtures.candidate(7, source="news_commentary", task="diacritization", messages=chat(
                "أضف التشكيل إلى النص التالي:\n\nكتب الطالب", "كتب الطالب")),
            fixtures.candidate(8, source="twitter_sentiment", task="sentiment_analysis", messages=chat(
                "ما هو شعور النص التالي؟\n\nأحب هذا المكان", "سعيد")),
            fixtures.candidate(9, messages=[
                {"role": "user", "content": "مرحبا"},
                {"role": "assistant", "content": "أهلا", "think": "تفكير يحتاج المراجعة"},
            ]),
            fixtures.candidate(10, source="diacritization", task="diacritization", messages=chat(
                "شكّل هذا النص: كتب الطالب", "كَتَبَ الطَّالِبُ")),
        ]
        # Valid JSON, invalid tool structure: exclusion takes priority over label repair.
        rows[8]["tools_json"] = '["invalid tool"]'
        skipped = [fixtures.candidate(90, split="validation"),
                   fixtures.candidate(91, labeled=False), fixtures.candidate(92, split="test")]
        for row in skipped:
            for column in ("messages_json", "tools_json", "metadata_json"):
                row[column] = "invalid JSON: evaluation inputs must not be decoded"
        blind = fixtures.candidate(200, split="test", labeled=False)
        blind["messages_json"] = "invalid JSON: do not read holdout payload"
        original_manifest = self.fixture.write_fixture(rows + skipped, holdout=[blind])
        before = self.fixture.input_hashes()
        destination = self.root / "curated_workspace"
        result = self.execute(output_root=destination)

        expected = ["accept", "accept", "repair", "exclude", "review", "review", "review", "repair", "exclude", "review", "review"]
        decisions = {row["id"]: row for row in result["decisions"]}
        self.assertEqual([decisions[row["id"]]["decision"] for row in rows], expected)
        self.assertEqual(result["candidates"], rows[:2])
        self.assertIn("no_diacritics_added", json.loads(decisions[rows[2]["id"]]["reasons_json"]))
        self.assertIn("translation_alignment_pending", json.loads(decisions[rows[4]["id"]]["reasons_json"]))
        self.assertIn("summary_grounding_pending", json.loads(decisions[rows[5]["id"]]["reasons_json"]))
        self.assertIn("python_syntax_invalid", json.loads(decisions[rows[6]["id"]]["reasons_json"]))
        self.assertIn("unsupported_task_wrapper", json.loads(decisions[rows[10]["id"]]["reasons_json"]))
        forbidden = {row["id"] for row in skipped + [blind]}
        self.assertFalse(forbidden & set(decisions))
        self.assertFalse(forbidden & {row["id"] for row in result["samples"]})
        for sample in result["samples"]:
            original = next(row for row in rows if row["id"] == sample["id"])
            self.assertEqual(fixtures.messages_of(sample), json.loads(original["messages_json"]))
        manifest = result["manifest"]
        self.assertEqual(manifest["counts"], dict(Counter(expected)))
        self.assertEqual(manifest["training_rows"], 11)
        self.assertEqual(manifest["skipped_candidate_rows"], {"validation": 1, "unlabeled": 1, "test": 1})
        self.assertEqual(manifest["input_manifest_sha256"], before[self.fixture.manifest_path])
        self.assertEqual(manifest["benchmarks"], original_manifest["benchmarks"])
        self.assertEqual((manifest["schema_version"], manifest["mode"], manifest["status"]), (1, "curate", "complete"))
        self.assertIs(manifest["training_ready"], False)
        self.assertTrue(result["output"].is_relative_to(destination / "data" / "processed" / "curate"))
        self.assertTrue(result["report"].is_relative_to(destination / "reports" / "curate"))
        self.assertTrue((result["report"] / "curation.md").exists())
        self.assertEqual(self.fixture.input_hashes(), before)

    def test_samples_are_deterministic_bounded_and_separate_decisions(self):
        rows = []
        for index in range(12):
            row = fixtures.candidate(index, messages=chat(f"مرحبا {index}", "أهلا"))
            if index >= 6:
                row["tools_json"] = '["invalid tool"]'
            rows.append(row)
        self.fixture.write_fixture(rows)
        first, second = self.execute(), self.execute()
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(first["decisions"], second["decisions"])
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual(len(first["samples"]), 4)
        self.assertEqual(Counter(row["decision"] for row in first["samples"]), {"accept": 2, "exclude": 2})
        self.assertEqual(first["manifest"]["counts"], {"accept": 6, "repair": 0, "review": 0, "exclude": 6})
        self.assertEqual(first["manifest"]["review_sample_rows"], 4)

    def test_incomplete_or_inconsistent_inputs_fail_before_scanning(self):
        manifest = self.fixture.write_fixture([fixtures.candidate(0)])
        for key, value in (("status", "running"), ("mode", "audit"), ("schema_version", 2)):
            with self.subTest(key=key):
                self.fixture.write_manifest({**manifest, key: value})
                with self.assertRaises(ValueError):
                    self.execute()
        altered = deepcopy(manifest)
        altered["counts"]["keep"] += 1
        self.fixture.write_manifest(altered)
        with patch.object(pq.ParquetFile, "iter_batches", side_effect=AssertionError("Must validate footers before scanning")):
            with self.assertRaises(ValueError):
                self.execute()

    def test_invalid_config_is_rejected(self):
        for key, value in (("schema_version", 2), ("seed", True), ("batch_size", 0),
                           ("sample_per_group", 0), ("repair_flags", "invalid_sentiment_label"),
                           ("review_sources", []), ("review_tasks", {"summarization": ""}),
                           ("unknown_option", True)):
            with self.subTest(key=key):
                altered = deepcopy(self.config)
                altered[key] = value
                self.write_config(altered)
                with self.assertRaises(ValueError):
                    self.curate.read_config(self.config_path)

    def test_corrupt_training_json_fails_without_publishing_outputs(self):
        rows = [fixtures.candidate(i, messages=chat(f"مرحبا {i}", "أهلا")) for i in range(3)]
        rows[-1]["messages_json"] = "{not valid JSON"
        self.fixture.write_fixture(rows)
        before = self.fixture.input_hashes()
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            self.execute()
        manifests = list((self.root / "reports" / "curate").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text(encoding="utf-8"))["status"], "failed")
        self.assertFalse(list((self.root / "data" / "processed" / "curate").rglob("*.parquet")))
        self.assertEqual(self.fixture.input_hashes(), before)


if __name__ == "__main__":
    unittest.main()
