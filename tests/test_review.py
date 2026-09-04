"""Offline review integration tests using small, real Parquet artifacts."""

from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from barq.data import CANDIDATE_SCHEMA, DECISION_SCHEMA


def candidate(index, *, source="native_a", task="dialogue", dataset="mix",
              split="train", labeled=True, messages=None, metadata=None):
    return {
        "id": hashlib.sha256(f"fixture-{index}".encode()).hexdigest(),
        "dataset": dataset, "revision": "1" * 40,
        "source": source, "task": task, "original_split": "test" if split == "test" else "train",
        "split": split, "row_index": index, "is_labeled": labeled,
        "example_hash": hashlib.sha256(f"example-{index}".encode()).hexdigest(),
        "input_hash": hashlib.sha256(f"input-{index}".encode()).hexdigest(),
        "messages_json": json.dumps(messages or [
            {"role": "user", "content": f"اكتب قصة قصيرة رقم {index}"},
            {"role": "assistant", "content": "مشى الطفل إلى المدرسة فرحًا."},
        ], ensure_ascii=False),
        "tools_json": "[]", "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
    }


def decision(row, status="keep", reasons=None):
    return {
        **{name: row[name] for name in (
            "id", "dataset", "revision", "source", "task", "original_split",
            "split", "row_index", "is_labeled",
        )},
        "decision": status, "reasons_json": json.dumps(reasons or []),
        "adapter_notes_json": "[]", "duplicate_of": None, "benchmark_matches_json": "[]",
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def messages_of(row):
    return row["messages"] if "messages" in row else json.loads(row["messages_json"])


def metadata_of(row):
    if "metadata" in row:
        return row["metadata"]
    if "metadata_json" in row:
        return json.loads(row["metadata_json"])
    return {"dialect": row.get("dialect", "")}


class ReviewIntegrationTests(unittest.TestCase):
    def setUp(self):
        from barq import review
        self.review = review
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_id = "20260904T000000Z-abcdef12"
        self.input_dir = self.root / "data" / "processed" / "prepare" / self.run_id
        self.input_dir.mkdir(parents=True)
        self.manifest_path = self.root / "reports" / "prepare" / self.run_id / "manifest.json"
        self.manifest_path.parent.mkdir(parents=True)
        self.addCleanup(patch.stopall)
        patch("socket.create_connection", side_effect=AssertionError("Review must be offline")).start()
        patch("socket.socket.connect", side_effect=AssertionError("Review must be offline")).start()

    def write_fixture(self, rows, *, decisions=None, holdout=None):
        holdout = holdout or []
        if decisions is None:
            decisions = [decision(row) for row in rows] + [decision(row, "holdout") for row in holdout]
        for name, records, schema in (
            ("candidates", rows, CANDIDATE_SCHEMA),
            ("decisions", decisions, DECISION_SCHEMA),
            ("holdout", holdout, CANDIDATE_SCHEMA),
        ):
            pq.write_table(pa.Table.from_pylist(records, schema=schema), self.input_dir / f"{name}.parquet")
        inspected = Counter(f"{row['dataset']}/{row['original_split']}" for row in decisions)
        manifest = {
            "schema_version": 1, "status": "complete", "mode": "prepare", "run_id": self.run_id,
            "counts": dict(Counter(row["decision"] for row in decisions)),
            "inspected_by_split": dict(inspected), "output": str(self.input_dir),
            "config": {"datasets": [], "seed": 42},
            "benchmarks": [{"name": "ArabicMMLU", "status": "not_checked", "reason": "No reference file configured"}],
        }
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest):
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def run_review(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return self.review.run(self.input_dir, **kwargs)

    def input_hashes(self):
        paths = list(self.input_dir.glob("*.parquet")) + [self.manifest_path]
        return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    def test_rejects_noncomplete_audit_wrong_schema_and_wrong_run_id(self):
        manifest = self.write_fixture([candidate(0)])
        for key, value in (("status", "running"), ("mode", "audit"), ("schema_version", 2), ("run_id", "different-run")):
            with self.subTest(key=key):
                altered = {**manifest, key: value}
                self.write_manifest(altered)
                with self.assertRaises(ValueError):
                    self.run_review()

    def test_footer_count_mismatch_fails_before_scanning_rows(self):
        manifest = self.write_fixture([candidate(0)])
        manifest["counts"]["keep"] += 1
        self.write_manifest(manifest)
        with patch.object(pq.ParquetFile, "iter_batches", side_effect=AssertionError("No scan before count validation")):
            with self.assertRaises(ValueError):
                self.run_review()

    def test_deterministic_bounded_samples_exclude_validation_and_unlabeled_rows(self):
        rows = [candidate(i, source="native_a" if i < 6 else "native_b") for i in range(12)]
        for dialect_index, dialect in enumerate(("gulf", "msa")):
            for behavior_index, behavior in enumerate(("call", "no_call")):
                for i in range(6):
                    row = candidate(20 + 12 * dialect_index + 6 * behavior_index + i,
                                    dataset="aisa", source="TuwaiqAcademy/AISA-ArabicFC", task="tool_use",
                                    metadata={"dialect": dialect})
                    if behavior == "call":
                        messages = json.loads(row["messages_json"])
                        messages[-1] = {"role": "assistant", "content": "", "tool_calls": [{
                            "function": {"name": "weather", "arguments": {"city": "الرياض"}},
                        }]}
                        row["messages_json"] = json.dumps(messages, ensure_ascii=False)
                        row["tools_json"] = json.dumps([{"function": {"name": "weather", "parameters": {
                            "type": "object", "properties": {"city": {"type": "string"}},
                        }}}])
                    rows.append(row)
        rows += [candidate(100, split="validation"), candidate(101, labeled=False)]
        blind = candidate(200, dataset="aisa", split="test", labeled=False)
        blind["messages_json"] = "invalid JSON: holdout content must never be read"
        self.write_fixture(rows, holdout=[blind])
        first = self.run_review(per_group=2, seed=42)
        second = self.run_review(per_group=2, seed=42)
        first_samples = read_jsonl(first / "review_samples.jsonl")
        second_samples = read_jsonl(second / "review_samples.jsonl")
        self.assertNotEqual(first, second)
        self.assertEqual({row["id"] for row in first_samples}, {row["id"] for row in second_samples})
        self.assertEqual(len(first_samples), 12)
        self.assertEqual(len({row["id"] for row in first_samples}), 12)
        group_counts = Counter((row["source"], metadata_of(row).get("dialect", ""), row.get("tool_behavior", ""))
                               for row in first_samples)
        self.assertEqual(sorted(group_counts.values()), [2, 2, 2, 2, 2, 2])
        for row in first_samples:
            self.assertEqual(row["split"], "train")
            self.assertTrue(row["is_labeled"])
            self.assertEqual(row["review"], {
                "decision": None, "correctness": None, "arabic_quality": None,
                "instruction_following": None, "notes": "",
            })
        disallowed = {rows[-2]["id"], rows[-1]["id"], blind["id"]}
        self.assertFalse(disallowed & {row["id"] for row in first_samples})

    def test_flags_and_diagnostic_samples_preserve_text_and_original_files(self):
        messages = [
            {"role": "user", "content": "لخّص النص التالي:\n\nعاد الطَّالِبُ. مواضيع قد تهمك نهاية"},
            {"role": "assistant", "content": "عاد الطَّالِبُ."},
        ]
        rows = [candidate(i, source="palm_ara", task="summarization", messages=messages) for i in range(7)]
        rows.append(candidate(99, source="palm_ara", task="summarization", messages=[
            {"role": "user", "content": "لخّص: عاد الطالب إلى بيته."},
            {"role": "assistant", "content": "عاد الطالب."},
        ]))
        self.write_fixture(rows)
        before = self.input_hashes()
        report = self.run_review(per_group=2, flagged_per_group=2)
        flags = pq.read_table(report / "flags.parquet").to_pylist()
        self.assertEqual({row["id"] for row in flags}, {row["id"] for row in rows[:7]})
        for row in flags:
            self.assertIn("source_boilerplate", json.loads(row["flags_json"]))
        diagnostics = read_jsonl(report / "flagged_samples.jsonl")
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(len({row["id"] for row in diagnostics}), 2)
        for row in diagnostics:
            self.assertEqual(messages_of(row), messages)
        self.assertEqual(self.input_hashes(), before)

    def test_benchmark_not_checked_status_is_carried_to_review(self):
        self.write_fixture([candidate(0)])
        report = self.run_review(per_group=1)
        manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["benchmarks"][0]["status"], "not_checked")
        self.assertIn("not_checked", (report / "review.md").read_text(encoding="utf-8").lower())

    def test_overlapping_quarantine_reasons_do_not_inflate_decision_row_totals(self):
        kept = candidate(0)
        quarantined = candidate(1)
        excluded = candidate(2)
        decisions = [decision(kept), decision(quarantined, "quarantine", ["empty_assistant", "invalid_message_content"]),
                     decision(excluded, "exclude", ["exact_duplicate"])]
        self.write_fixture([kept], decisions=decisions)
        report = self.run_review(per_group=1)
        manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["decision_counts"], {"keep": 1, "quarantine": 1, "exclude": 1})
        self.assertEqual(manifest["reason_counts"]["empty_assistant"], 1)
        self.assertEqual(manifest["reason_counts"]["invalid_message_content"], 1)
        self.assertEqual(sum(manifest["decision_counts"].values()), 3)

    def test_explicit_manifest_supports_moved_inputs_without_changing_them(self):
        self.write_fixture([candidate(0)])
        moved = self.root / "moved" / self.run_id
        moved.mkdir(parents=True)
        for path in self.input_dir.glob("*.parquet"):
            (moved / path.name).write_bytes(path.read_bytes())
        output_root = self.root / "review_workspace"
        with redirect_stdout(io.StringIO()):
            report = self.review.run(moved, manifest_path=self.manifest_path, output_root=output_root, per_group=1)
        self.assertTrue(report.is_relative_to(output_root / "reports" / "review"))
        self.assertEqual(len(read_jsonl(report / "review_samples.jsonl")), 1)
        flat_manifest = moved.parent / "original-manifest.json"
        flat_manifest.write_bytes(self.manifest_path.read_bytes())
        with redirect_stdout(io.StringIO()), patch.object(Path, "cwd", return_value=output_root):
            report = self.review.run(moved, manifest_path=flat_manifest, per_group=1)
        self.assertTrue(report.is_relative_to(output_root / "reports" / "review"))

    def test_bad_candidate_json_records_failed_run_without_publishing_final_flags(self):
        row = candidate(0)
        row["messages_json"] = "{not valid JSON"
        self.write_fixture([row])
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            self.run_review()
        manifests = list((self.root / "reports" / "review").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((manifests[0].parent / "flags.parquet").exists())


if __name__ == "__main__":
    unittest.main()
