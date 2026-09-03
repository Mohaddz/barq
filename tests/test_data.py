"""Small end-to-end fixtures: no network calls or dataset downloads."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow.parquet as pq
import yaml

from barq import data


def mix_row(prompt="وش أخبارك؟", answer="بخير، الحمد لله", source="native_dialogue"):
    return {
        "dataset_name": source,
        "task_type": "dialogue",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def aisa_row(prompt="كيف الجو بالرياض؟", *, blind=False):
    tools = [{"function": {
        "name": "weather",
        "description": "حالة الطقس",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unrelated": None},
            "required": [],
        },
    }}]
    return {
        "text": "<bos>هذا قالب أصلي لا ينبغي إدخاله في المحادثة",
        "messages": [
            {"role": "developer", "content": "التاريخ الحالي 2026-01-01", "tool_calls": None},
            {"role": "user", "content": prompt, "think": None},
            {
                "role": "assistant", "content": "", "think": None,
                "_think_for_train": None,
                "tool_calls": None if blind else [{"function": {
                    "name": "weather", "arguments": {"city": "الرياض", "unrelated": None},
                }}],
            },
        ],
        "tools_sampled": tools,
        "tools": tools + [{"function": {"name": "registry_only_tool"}}],
        "requires_function": None if blind else True,
        "tool_called": None if blind else "weather",
        "negative_category": None,
        "dialect": "gulf",
    }


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "configs" / "data.yaml"
        self.config_path.parent.mkdir()

    def config(self, rows):
        return {
            "seed": 42,
            "validation_fraction": 0.4,
            "audit_rows": 100,
            "sample_per_group": 2,
            "batch_size": 2,
            "datasets": [{
                "name": name,
                "repo_id": f"fixture/{name}",
                "revision": "1" * 40,
                "config": "default",
                "adapter": name,
                "preserve_splits": name == "aisa",
                "splits": {split: len(values) for split, values in splits.items()},
                "unlabeled_splits": ["test"] if name == "aisa" and "test" in splits else [],
            } for name, splits in rows.items()],
            "benchmarks": [{"name": "ArabicMMLU", "path": None}],
        }

    def execute(self, rows, *, config=None, mode="audit"):
        config = config or self.config(rows)
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        def preview(source, split, *_):
            return iter(enumerate(deepcopy(rows[source["name"]][split])))

        def download(source, _):
            return {"fixture_dataset": source["name"]}

        def local(files, split, _):
            return iter(enumerate(deepcopy(rows[files["fixture_dataset"]][split])))

        with (
            patch.object(data, "audit_rows", side_effect=preview) as audit,
            patch.object(data, "download_source", side_effect=download) as downloads,
            patch.object(data, "prepare_rows", side_effect=local) as prepared,
            redirect_stdout(io.StringIO()),
        ):
            output, report = data.run(config, self.config_path, mode, self.root)
            if mode == "audit":
                downloads.assert_not_called()
                prepared.assert_not_called()
            else:
                audit.assert_not_called()
                self.assertEqual(downloads.call_count, len(rows))
        return {
            "output": output,
            "candidates": pq.read_table(output / "candidates.parquet").to_pylist(),
            "holdout": pq.read_table(output / "holdout.parquet").to_pylist(),
            "decisions": pq.read_table(output / "decisions.parquet").to_pylist(),
            "manifest": json.loads((report / "manifest.json").read_text(encoding="utf-8")),
        }

    def test_aisa_padding_cleanup_preserves_valid_null_zero_false_and_empty_string(self):
        row = aisa_row()
        schema = row["tools_sampled"][0]["function"]["parameters"]
        schema["properties"].update({
            "days": {"type": "integer"},
            "enabled": {"type": "boolean"},
            "note": {"type": ["string", "null"]},
            "empty": {"type": "string"},
            "anything": True,
        })
        arguments = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        arguments.update(days=0, enabled=False, note=None, empty="", anything=None)
        original = deepcopy(row)
        result = self.execute({"aisa": {"train": [row]}})
        self.assertEqual(result["manifest"]["counts"], {"keep": 1})
        candidate = result["candidates"][0]
        messages = json.loads(candidate["messages_json"])
        self.assertEqual(messages[-1]["tool_calls"][0]["function"]["arguments"], {
            "city": "الرياض", "days": 0, "enabled": False,
            "note": None, "empty": "", "anything": None,
        })
        tools = json.loads(candidate["tools_json"])
        self.assertEqual([tool["function"]["name"] for tool in tools], ["weather"])
        self.assertNotIn("unrelated", tools[0]["function"]["parameters"]["properties"])
        self.assertEqual(messages[1]["content"], original["messages"][1]["content"])
        self.assertEqual(row, original)

    def test_blind_test_goes_only_to_holdout_while_dev_keeps_its_label(self):
        result = self.execute({"aisa": {
            "test": [aisa_row("طقس جدة؟", blind=True)],
            "dev": [aisa_row("طقس الرياض؟")],
        }})
        self.assertEqual(len(result["holdout"]), 1)
        self.assertFalse(result["holdout"][0]["is_labeled"])
        self.assertEqual(result["holdout"][0]["split"], "test")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertTrue(result["candidates"][0]["is_labeled"])
        self.assertEqual(result["candidates"][0]["original_split"], "dev")
        self.assertEqual(result["candidates"][0]["split"], "validation")
        self.assertEqual(result["manifest"]["counts"], {"holdout": 1, "keep": 1})

    def test_official_test_and_dev_inputs_exclude_matching_training_rows(self):
        result = self.execute({"aisa": {
            "train": [aisa_row("السؤال المشترك مع الاختبار"), aisa_row("السؤال المشترك مع التطوير")],
            "dev": [aisa_row("السؤال المشترك مع التطوير")],
            "test": [aisa_row("السؤال المشترك مع الاختبار", blind=True)],
        }})
        training = [row for row in result["decisions"] if row["original_split"] == "train"]
        self.assertEqual(len(training), 2)
        for row in training:
            self.assertEqual(row["decision"], "exclude")
            self.assertIn("official_holdout_overlap", json.loads(row["reasons_json"]))
        self.assertEqual([row["original_split"] for row in result["candidates"]], ["dev"])
        self.assertEqual(len(result["holdout"]), 1)

    def test_malformed_dev_target_still_protects_its_valid_input_from_training(self):
        dev = aisa_row("سؤال محجوز للتطوير")
        dev["messages"][-1]["tool_calls"][0]["function"]["arguments"] = "not valid JSON"
        result = self.execute({"aisa": {
            "train": [aisa_row("سؤال محجوز للتطوير")],
            "dev": [dev],
        }})
        decisions = {row["original_split"]: row for row in result["decisions"]}
        self.assertEqual(decisions["dev"]["decision"], "quarantine")
        self.assertIn("invalid_tool_arguments", json.loads(decisions["dev"]["reasons_json"]))
        self.assertEqual(decisions["train"]["decision"], "exclude")
        self.assertIn("official_holdout_overlap", json.loads(decisions["train"]["reasons_json"]))
        self.assertEqual(result["candidates"], [])

    def test_mix_splits_are_deterministic_across_order_and_duplicate_sources(self):
        examples = [mix_row(f"سؤال رقم {index}", f"جواب رقم {index}") for index in range(30)]
        duplicate = deepcopy(examples[4])
        duplicate["dataset_name"] = "another_original_source"
        first = self.execute({"mix": {"train": examples + [duplicate]}})
        second = self.execute({"mix": {"train": [duplicate] + list(reversed(examples))}})
        assignment = lambda result: {row["input_hash"]: row["split"] for row in result["candidates"]}
        self.assertEqual(assignment(first), assignment(second))
        self.assertEqual(set(assignment(first).values()), {"train", "validation"})
        self.assertEqual(len(first["candidates"]), 30)
        for result in (first, second):
            duplicates = [row for row in result["decisions"] if "exact_duplicate" in json.loads(row["reasons_json"])]
            self.assertEqual(len(duplicates), 1)
            self.assertIn(duplicates[0]["duplicate_of"], {row["id"] for row in result["candidates"]})

    def test_unconfigured_benchmarks_are_reported_unchecked_and_arabic_is_unchanged(self):
        row = mix_row("صحّح: الطالبان ذهب إلى المدرسه", "ذَهَبَ الطَّالِبَانِ إِلَى الْمَدْرَسَةِ.", "grammar_corrections")
        row["task_type"] = "grammar_correction"
        result = self.execute({"mix": {"train": [row]}})
        self.assertEqual(result["manifest"]["benchmarks"][0]["status"], "not_checked")
        candidate = result["candidates"][0]
        self.assertEqual(json.loads(candidate["messages_json"]), row["messages"])
        self.assertEqual(candidate["source"], "grammar_corrections")
        self.assertEqual(candidate["task"], "grammar_correction")

    def test_local_benchmark_variants_match_nfc_whitespace_but_not_arabic_letters(self):
        reference = self.root / "reference.jsonl"
        reference.write_text(json.dumps({
            "prompt": "سؤال آخر", "variants": ["أكل الطعام"],
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        rows = {"mix": {"train": [mix_row("  ا\u0654كل\n الطعام "), mix_row("اكل الطعام")]}}
        config = self.config(rows)
        config["benchmarks"][0]["path"] = "../reference.jsonl"
        result = self.execute(rows, config=config)
        self.assertEqual(result["manifest"]["benchmarks"][0]["status"], "checked_exact")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(json.loads(result["candidates"][0]["messages_json"])[0]["content"], "اكل الطعام")
        excluded = next(row for row in result["decisions"] if row["decision"] == "exclude")
        self.assertIn("benchmark_overlap", json.loads(excluded["reasons_json"]))
        self.assertEqual(json.loads(excluded["benchmark_matches_json"]), ["ArabicMMLU"])

    def test_missing_configured_benchmark_fails_before_fetching_and_records_failure(self):
        rows = {"mix": {"train": [mix_row()]}}
        config = self.config(rows)
        config["benchmarks"][0]["path"] = "missing.jsonl"
        with patch.object(data, "audit_rows") as audit, patch.object(data, "download_source") as download:
            with self.assertRaises(FileNotFoundError):
                data.run(config, self.config_path, "audit", self.root)
            audit.assert_not_called()
            download.assert_not_called()
        manifest_path = next((self.root / "reports" / "audit").glob("*/manifest.json"))
        self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["status"], "failed")
        self.assertFalse(list((self.root / "data" / "processed").rglob("candidates.parquet")))

    def test_prepare_uses_downloaded_rows_and_writes_fresh_batched_outputs(self):
        rows = {"mix": {"train": [mix_row(f"سؤال {i}", f"جواب {i}") for i in range(5)]}}
        first = self.execute(rows, mode="prepare")
        original_bytes = (first["output"] / "candidates.parquet").read_bytes()
        second = self.execute(rows, mode="prepare")
        self.assertNotEqual(first["output"], second["output"])
        self.assertEqual((first["output"] / "candidates.parquet").read_bytes(), original_bytes)
        self.assertEqual(first["manifest"]["inspected_by_split"], {"mix/train": 5})
        self.assertEqual(len(second["candidates"]), 5)
        self.assertEqual({row["id"] for row in first["candidates"]}, {row["id"] for row in second["candidates"]})


if __name__ == "__main__":
    unittest.main()
