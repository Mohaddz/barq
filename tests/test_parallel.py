"""Exercise real worker processes with small, entirely local input fixtures."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from barq import data


def mix_row(prompt, answer="جواب", source="dialogue"):
    return {"dataset_name": source, "task_type": "dialogue", "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]}


def aisa_row(prompt, *, blind=False):
    return {
        "messages": [
            {"role": "developer", "content": "التاريخ الحالي: 2026-01-01"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": None if blind else [
                {"function": {"name": "weather", "arguments": {"city": "الرياض"}}},
            ]},
        ],
        "tools_sampled": [{"function": {"name": "weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }}}],
        "dialect": "gulf", "requires_function": None if blind else True,
        "tool_called": None if blind else "weather", "negative_category": None,
    }


class ParallelPrepareTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config_path = self.root / "data.json"

    def execute(self, rows, workers):
        config = {
            "seed": 42, "validation_fraction": 0.4, "audit_rows": 100,
            "sample_per_group": 2, "batch_size": 2, "benchmarks": [],
            "datasets": [{
                "name": name, "repo_id": f"fixture/{name}", "revision": "1" * 40,
                "config": "default", "adapter": name, "preserve_splits": name == "aisa",
                "splits": {split: len(values) for split, values in splits.items()},
                "unlabeled_splits": ["test"] if name == "aisa" and "test" in splits else [],
            } for name, splits in rows.items()],
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        # These readers execute in the parent. Workers receive ordinary row batches.
        def download(source, _):
            return {"fixture_dataset": source["name"]}

        def read(files, split, _):
            return iter(enumerate(deepcopy(rows[files["fixture_dataset"]][split])))

        with (
            patch.object(data, "download_source", side_effect=download) as downloads,
            patch.object(data, "prepare_rows", side_effect=read),
            redirect_stdout(io.StringIO()),
        ):
            output, report = data.run(
                config, self.config_path, "prepare", self.root, workers=workers,
            )
            self.assertEqual(downloads.call_count, len(rows))
        manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        return {
            **{name: pq.read_table(output / f"{name}.parquet").to_pylist()
               for name in ("candidates", "decisions", "holdout")},
            "samples": (report / "samples.jsonl").read_text(encoding="utf-8"),
            "counts": manifest["counts"], "inspected": manifest["inspected_by_split"],
        }

    def test_parallel_matches_serial_with_global_dedup_and_holdout_priority(self):
        shared = mix_row("نصٌّ مشترك", "الجواب الأول")
        other_source = deepcopy(shared)
        other_source["dataset_name"] = "another_source"
        malformed_dev = aisa_row("سؤال التطوير ذو الهدف المعطوب")
        malformed_dev["messages"][-1]["tool_calls"][0]["function"]["arguments"] = "{bad JSON"
        rows = {
            # Deliberately list mix/train first: official holdouts must run first.
            "mix": {"train": [shared]
                    + [mix_row(f"سؤال رقم {index}") for index in range(12)]
                    + [mix_row("نصٌّ مشترك", "الجواب الثاني"), other_source,
                       mix_row("سؤال بلا إجابة", "")]},
            "aisa": {
                "train": [aisa_row("سؤال الاختبار"), aisa_row("سؤال التدريب"),
                          aisa_row("سؤال التطوير ذو الهدف المعطوب"),
                          aisa_row("سؤال التطوير الصحيح"), aisa_row("سؤال التدريب")],
                "dev": [malformed_dev, aisa_row("سؤال التطوير الصحيح")],
                "test": [aisa_row("سؤال الاختبار", blind=True)],
            },
        }
        original = deepcopy(rows)
        serial = self.execute(rows, workers=1)
        parallel = self.execute(rows, workers=2)
        self.assertEqual(rows, original)
        # Exact order, row IDs, duplicate_of, metadata and reservoir samples all agree.
        self.assertEqual(parallel, serial)

        decisions = {(row["dataset"], row["original_split"], row["row_index"]): row
                     for row in parallel["decisions"]}
        for index in (0, 2, 3):
            row = decisions["aisa", "train", index]
            self.assertEqual(row["decision"], "exclude")
            self.assertIn("official_holdout_overlap", json.loads(row["reasons_json"]))
        self.assertEqual(decisions["aisa", "dev", 0]["decision"], "quarantine")
        self.assertIn("invalid_tool_arguments", json.loads(
            decisions["aisa", "dev", 0]["reasons_json"]))
        self.assertEqual(len(parallel["holdout"]), 1)
        self.assertFalse(parallel["holdout"][0]["is_labeled"])

        # Duplicates occur several batches apart, including across source labels.
        for duplicate_key, original_key in (
            (("mix", "train", 14), ("mix", "train", 0)),
            (("aisa", "train", 4), ("aisa", "train", 1)),
        ):
            duplicate, first = decisions[duplicate_key], decisions[original_key]
            self.assertEqual(duplicate["decision"], "exclude")
            self.assertIn("exact_duplicate", json.loads(duplicate["reasons_json"]))
            self.assertEqual(duplicate["duplicate_of"], first["id"])
        alternatives = [row for row in parallel["candidates"]
                        if row["dataset"] == "mix" and row["row_index"] in (0, 13)]
        self.assertEqual(len(alternatives), 2)
        self.assertEqual(len({row["input_hash"] for row in alternatives}), 1)
        self.assertEqual(len({row["split"] for row in alternatives}), 1)
        self.assertEqual(len({row["example_hash"] for row in alternatives}), 2)

    def test_worker_exception_records_failure_without_publishing_parquet(self):
        # A non-record input forces an exception inside the real adaptation worker.
        rows = {"mix": {"train": [mix_row("الأول"), mix_row("الثاني"), None]}}
        with self.assertRaisesRegex(AttributeError, "get"):
            self.execute(rows, workers=2)
        manifests = list((self.root / "reports" / "prepare").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(list((self.root / "data" / "processed").rglob("*.parquet")))

    def test_checked_rows_bounds_input_consumption_and_cancels_pending_work(self):
        class Future:
            def __init__(self, batch, fail):
                self.batch, self.fail, self.cancelled = batch, fail, False

            def result(self):
                if self.fail:
                    raise RuntimeError("worker failed")
                return self.batch

            def cancel(self):
                self.cancelled = True

        class Executor:
            def __init__(self, fail):
                self.fail, self.futures = fail, []

            def submit(self, function, batch, source, split):
                future = Future(batch, self.fail and not self.futures)
                self.futures.append(future)
                return future

        for fail in (False, True):
            with self.subTest(worker_failure=fail):
                consumed = []

                def rows():
                    for index in range(1000):
                        consumed.append(index)
                        yield index, {"value": index}

                executor = Executor(fail)
                checked = data.checked_rows(rows(), {}, "train", 2, executor, workers=2)
                try:
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "worker failed"):
                            next(checked)
                    else:
                        self.assertEqual(next(checked), (0, {"value": 0}))
                    # Four queued batches plus the batch currently being yielded.
                    self.assertLessEqual(len(consumed), 10)
                finally:
                    checked.close()
                self.assertTrue(all(future.cancelled for future in executor.futures[1:]))

    def test_parquet_reader_preserves_file_order_indices_and_projects_aisa_fields(self):
        rows = [aisa_row(f"سؤال {index}") for index in range(3)]
        for index, row in enumerate(rows):
            row.update(text="قالب أصلي", tools=["unused_registry"], marker=index)
            for message in row["messages"]:
                message.setdefault("tool_calls", None)
        # Supply reverse alphabetical file names to verify the caller's order wins.
        paths = [self.root / "z-first.parquet", self.root / "a-second.parquet"]
        pq.write_table(pa.Table.from_pylist(rows[:2]), paths[0], row_group_size=1)
        pq.write_table(pa.Table.from_pylist(rows[2:]), paths[1])
        actual = list(data.prepare_rows({"train": [str(path) for path in paths]}, "train", self.root))
        expected = [(index, {key: value for key, value in row.items() if key not in {"text", "tools"}})
                    for index, row in enumerate(rows)]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
