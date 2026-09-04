"""Durable boundaries and conservative reuse, using tiny real offline stage outputs."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from barq import persistent
from barq.data import CANDIDATE_SCHEMA


class PersistentPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "volume"
        self.config_dir = Path(temporary.name) / "configs"
        self.config_dir.mkdir()
        self.config = {
            "seed": 42, "validation_fraction": 0, "audit_rows": 1,
            "sample_per_group": 1, "batch_size": 2, "benchmarks": [],
            "datasets": [{"name": name, "repo_id": "fixture/" + name, "revision": "a" * 40,
                          "config": "default", "adapter": "mix", "preserve_splits": False,
                          "splits": {"train": 1}, "unlabeled_splits": []} for name in ("mix", "other")],
        }
        self.write_config()
        (self.config_dir / "curation.yaml").write_text(yaml.safe_dump({
            "schema_version": 1, "seed": 42, "batch_size": 2, "sample_per_group": 1,
            "repair_flags": [], "review_sources": {}, "review_tasks": {},
        }), encoding="utf-8")
        self.events = []
        real_prepare, real_curate = persistent.prepare_run, persistent.curate_run

        def tracked_prepare(*args, **kwargs):
            self.events.append("prepare:start")
            result = real_prepare(*args, **kwargs)
            self.events.append("prepare:closed")
            return result

        def tracked_curate(*args, **kwargs):
            self.events.append("curate:start")
            result = real_curate(*args, **kwargs)
            self.events.append("curate:closed")
            return result

        def cached_source(source, raw_root):
            cache = raw_root / source["name"] / "cached.txt"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if not cache.exists():
                cache.write_text("keep original cache", encoding="utf-8")
            return {"name": source["name"]}

        def local_rows(files, split, raw_root):
            yield 0, {"dataset_name": "fixture", "task_type": "dialogue", "messages": [
                {"role": "user", "content": "مرحبا " + files["name"]},
                {"role": "assistant", "content": "أهلا"},
            ]}

        self.prepare = self.enterContext(patch.object(persistent, "prepare_run", side_effect=tracked_prepare))
        self.curate = self.enterContext(patch.object(persistent, "curate_run", side_effect=tracked_curate))
        self.enterContext(patch("barq.data.download_source", side_effect=cached_source))
        self.enterContext(patch("barq.data.prepare_rows", side_effect=local_rows))
        self.enterContext(patch("socket.create_connection", side_effect=AssertionError("No network in tests")))

    def write_config(self):
        (self.config_dir / "data.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def commit(self):
        self.events.append("commit:latest" if (self.root / "latest.json").exists() else "commit:stages")

    def execute(self, commit=None):
        with redirect_stdout(io.StringIO()):
            return persistent.run(self.root, config_dir=self.config_dir, commit=commit or self.commit, workers=1)

    def manifest(self, result, stage):
        path = self.root / result[stage + "_manifest"]
        return path, json.loads(path.read_bytes())

    def test_closed_stages_are_committed_before_latest_and_completed_runs_are_reused(self):
        first = self.execute()
        self.assertEqual(self.events, ["prepare:start", "prepare:closed", "commit:stages",
                                      "curate:start", "curate:closed", "commit:stages", "commit:latest"])
        self.assertEqual(json.loads((self.root / "latest.json").read_bytes()), first)
        self.assertEqual(set(first), {"prepare_run_id", "curate_run_id", "prepare_report",
                                     "prepare_manifest", "curate_report", "curate_manifest"})
        self.assertFalse(any(Path(value).is_absolute() for value in first.values()))
        self.assertFalse(list(self.root.rglob("*.partial")))
        second = self.execute()
        self.assertEqual(second, first)
        self.assertEqual((self.prepare.call_count, self.curate.call_count), (1, 1))

    def test_interrupted_curate_keeps_failed_report_and_reuses_completed_prepare(self):
        with patch("barq.curate.assess", side_effect=KeyboardInterrupt("interrupted")):
            with self.assertRaises(KeyboardInterrupt):
                self.execute()
        self.assertFalse((self.root / "latest.json").exists())
        failed_path = next((self.root / "reports" / "curate").glob("*/manifest.json"))
        self.assertEqual(json.loads(failed_path.read_bytes())["status"], "failed")
        prepare_path = next((self.root / "reports" / "prepare").glob("*/manifest.json"))
        self.assertEqual(json.loads(prepare_path.read_bytes())["status"], "complete")
        result = self.execute()
        self.assertEqual(result["prepare_run_id"], prepare_path.parent.name)
        self.assertNotEqual(result["curate_run_id"], failed_path.parent.name)
        self.assertEqual((self.prepare.call_count, self.curate.call_count), (1, 2))
        self.assertTrue(failed_path.exists())
        self.assertEqual((self.root / "data/raw/mix/cached.txt").read_text(), "keep original cache")

    def test_config_changes_invalidate_only_the_necessary_stage(self):
        first = self.execute()
        with (self.config_dir / "data.yaml").open("a", encoding="utf-8") as stream:
            stream.write("\n# Formatting-only edit does not change the canonical data config.\n")
        self.assertEqual(self.execute(), first)
        with (self.config_dir / "curation.yaml").open("a", encoding="utf-8") as stream:
            stream.write("\n# Curation identity deliberately uses file bytes.\n")
        second = self.execute()
        self.assertEqual(second["prepare_run_id"], first["prepare_run_id"])
        self.assertNotEqual(second["curate_run_id"], first["curate_run_id"])
        self.config["seed"] = 43
        self.write_config()
        third = self.execute()
        self.assertNotEqual(third["prepare_run_id"], first["prepare_run_id"])
        self.assertEqual((self.prepare.call_count, self.curate.call_count), (2, 3))

    def test_old_code_hashes_and_partial_dataset_selection_are_not_reused(self):
        result = self.execute()
        path, manifest = self.manifest(result, "curate")
        manifest["implementation_sha256"] = "0" * 64  # A completed run from older implementation.
        path.write_text(json.dumps(manifest), encoding="utf-8")
        changed = self.execute()
        self.assertEqual(changed["prepare_run_id"], result["prepare_run_id"])
        self.assertNotEqual(changed["curate_run_id"], result["curate_run_id"])
        for alteration in ({"implementation_sha256": "0" * 64}, {"selected_datasets": ["mix"]}):
            path, manifest = self.manifest(changed, "prepare")
            manifest.update(alteration)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            next_result = self.execute()
            self.assertNotEqual(next_result["prepare_run_id"], changed["prepare_run_id"])
            self.assertTrue(path.exists())
            changed = next_result

    def test_footer_mismatches_skipped_counts_and_missing_review_samples_force_curate(self):
        result = self.execute()
        for damage in ("candidate_footer", "decision_footer", "negative_skipped", "missing_coverage", "samples"):
            with self.subTest(damage=damage):
                path, manifest = self.manifest(result, "curate")
                output = self.root / "data/processed/curate" / result["curate_run_id"]
                if damage == "candidate_footer":
                    pq.write_table(pa.Table.from_pylist([], schema=CANDIDATE_SCHEMA), output / "candidates.parquet")
                elif damage == "decision_footer":
                    pq.write_table(pa.Table.from_pylist([], schema=persistent.CURATION_SCHEMA), output / "decisions.parquet")
                elif damage == "samples":
                    (path.parent / "review_samples.jsonl").unlink()
                else:
                    manifest["skipped_candidate_rows"] = {"validation": -1 if damage == "negative_skipped" else 1}
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                updated = self.execute()
                self.assertEqual(updated["prepare_run_id"], result["prepare_run_id"])
                self.assertNotEqual(updated["curate_run_id"], result["curate_run_id"])
                result = updated
        self.assertEqual(self.prepare.call_count, 1)

    def test_changed_benchmark_content_invalidates_prepare_without_config_edit(self):
        self.config["benchmarks"] = [{"name": "fixture", "path": "benchmark.jsonl"}]
        self.write_config()
        benchmark = self.config_dir / "benchmark.jsonl"
        benchmark.write_text(json.dumps({"prompt": "unrelated"}) + "\n", encoding="utf-8")
        first = self.execute()
        benchmark.write_text(json.dumps({"prompt": "مرحبا mix"}) + "\n", encoding="utf-8")
        second = self.execute()
        self.assertNotEqual(second["prepare_run_id"], first["prepare_run_id"])
        _, manifest = self.manifest(second, "prepare")
        self.assertEqual(manifest["counts"], {"exclude": 1, "keep": 1})

    def test_failure_commit_does_not_mask_original_stage_error(self):
        commits = []

        def unreliable_commit():
            commits.append(True)
            if len(commits) > 1:
                raise RuntimeError("storage unavailable")

        with patch("barq.curate.assess", side_effect=ValueError("original stage failure")):
            with self.assertRaisesRegex(ValueError, "original stage failure") as caught:
                self.execute(commit=unreliable_commit)
        self.assertIn("storage unavailable", " ".join(caught.exception.__notes__))
        self.assertFalse((self.root / "latest.json").exists())
        path = next((self.root / "reports/curate").glob("*/manifest.json"))
        self.assertEqual(json.loads(path.read_bytes())["status"], "failed")

    def test_prepare_commit_failure_prevents_curate_and_latest_publication(self):
        with self.assertRaisesRegex(RuntimeError, "commit failed"):
            self.execute(commit=lambda: (_ for _ in ()).throw(RuntimeError("commit failed")))
        self.assertEqual(self.prepare.call_count, 1)
        self.curate.assert_not_called()
        self.assertFalse((self.root / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
