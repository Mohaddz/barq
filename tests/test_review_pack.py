from pathlib import Path
from tempfile import TemporaryDirectory
import json
import tarfile
import unittest

from barq import review_pack as rp
from test_judge import example


class ReviewPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.jsonl"
        self.history = self.root / "history.jsonl"
        self.pack = self.root / "pack"

    def build(self):
        rows = [example(i) for i in range(9)]
        rows[1]["input_hash"] = rows[0]["input_hash"]
        rows[2]["split"] = "validation"
        rows[3]["is_labeled"] = False
        rows[4]["original_split"] = "dev"
        for row in rows[5:]:
            row["messages"][-1].update(content="</script><script>alert('INJECTION')</script>", think="THINK_SENTINEL")
            row.update(task_hint="HINT_SENTINEL", flags=["FLAG_SENTINEL"])
        rp.write_rows(self.source, rows)
        rp.write_rows(self.history, [rows[0]])
        return rp.build(self.source, [self.history], self.pack, limit=4)

    def labels(self):
        manifest, examples = rp.pack_rows(self.pack)
        return [{"schema_version": 1, "pack_id": manifest["pack_id"],
                 **{k: row[k] for k in rp.KEYS}, "label": "usable", "note": ""} for row in examples.values()]

    def test_fresh_training_only_and_blind_html(self):
        manifest = self.build()
        _, rows = rp.pack_rows(self.pack)
        self.assertEqual(len(rows), 4)
        self.assertEqual(manifest["excluded_by_history"], 2)
        self.assertTrue(all(r["row_index"] >= 5 for r in rows.values()))
        html = (self.pack / "review.html").read_text(encoding="utf-8")
        self.assertNotIn("</script><script>alert", html)
        for hidden in ("THINK_SENTINEL", "HINT_SENTINEL", "FLAG_SENTINEL"):
            self.assertNotIn(hidden, html)
        self.assertIn("\\u003c/script\\u003e", html)
        other = self.root / "other"
        rp.build(self.source, [self.history], other, limit=4)
        self.assertEqual((self.pack / "examples.jsonl").read_bytes(), (other / "examples.jsonl").read_bytes())
        with self.assertRaises(FileExistsError):
            rp.build(self.source, [self.history], self.pack, limit=4)

    def test_history_archive_no_extraction_and_fail_on_missing_history(self):
        self.build()
        archive = self.root / "history.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(self.history, arcname="../../sample.jsonl")
        self.assertEqual(rp.history_rows(archive), rp.read_jsonl(self.history)[0])
        with self.assertRaises(ValueError):
            rp.build(self.source, [], self.root / "empty", limit=1)
        with self.assertRaises(ValueError):
            rp.build(self.source, [self.history], self.root / "too-many", limit=100)

    def test_freeze_requires_complete_matching_labels_and_prevents_overwrite(self):
        self.build()
        labels = self.labels()
        path = self.root / "labels.jsonl"
        for invalid in (labels[:-1], labels + [labels[0]], [{**labels[0], "input_hash": "wrong"}, *labels[1:]]):
            rp.write_rows(path, invalid)
            with self.assertRaises(ValueError):
                rp.freeze(self.pack, path)
        rp.write_rows(path, labels)
        rp.freeze(self.pack, path)
        with self.assertRaises(FileExistsError):
            rp.freeze(self.pack, path)

    def test_comparison_preserves_uncertainty_and_detects_tampering(self):
        self.build()
        labels = self.labels()
        labels[0]["label"] = "flawed"
        labels[1]["label"] = "unsure"
        path = self.root / "labels.jsonl"
        rp.write_rows(path, labels)
        rp.freeze(self.pack, path)
        judgments = [{**{k: row[k] for k in rp.KEYS}, "status": "complete", "judgment": {"decision": "keep"}} for row in labels]
        judgments[2]["judgment"]["decision"] = "repair"
        judgments[3] = {**judgments[3], "status": "error"}
        target = self.root / "judgments.jsonl"
        rp.write_rows(target, judgments)
        counts = rp.compare(self.pack, target)["creative_writing"]
        self.assertEqual(counts["flawed_kept"], 1)
        self.assertEqual(counts["usable_rejected"], 1)
        self.assertEqual(counts["unjudged"], 1)
        labels[0]["label"] = "usable"
        rp.write_rows(self.pack / "human_labels.jsonl", labels)
        with self.assertRaises(ValueError):
            rp.compare(self.pack, target)


if __name__ == "__main__":
    unittest.main()
