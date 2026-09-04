"""Bounded judge runs tested with tiny inputs and a fake OpenRouter transport."""

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import threading
import unittest
from unittest.mock import patch

import yaml


REPO = Path(__file__).resolve().parents[1]


def example(index, **changes):
    row = {
        "id": hashlib.sha256(f"judge-row-{index}".encode()).hexdigest(),
        "dataset": "mix", "source": "native", "task": "creative_writing",
        "task_hint": "creative_writing", "dialect": "msa", "tool_behavior": "",
        "revision": "a" * 40, "original_split": "train", "split": "train",
        "is_labeled": True, "row_index": index,
        "example_hash": hashlib.sha256(f"judge-example-{index}".encode()).hexdigest(),
        "input_hash": hashlib.sha256(f"judge-input-{index}".encode()).hexdigest(),
        "messages": [
            {"role": "user", "content": f"اكتب قصة قصيرة رقم {index}"},
            {"role": "assistant", "content": "مشى الطفل إلى المدرسة فرحًا."},
        ],
        "tools": [], "metadata": {}, "flags": [],
    }
    row.update(changes)
    return row


def verdict(decision="keep", **dimensions):
    return {
        "decision": decision, "reasons": ["The answer follows the supplied request."],
        "dimensions": {"correctness": "pass", "language_quality": "pass",
                       "instruction_following": "pass", **dimensions},
    }


def response(result=None, *, cost=0.0001):
    return {
        "id": "gen-fake", "model": "meta/muse-spark-1.3-contributor", "provider": "Meta",
        "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(result if result is not None else verdict()),
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost},
    }


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


class JudgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        from barq import quality
        self.quality = quality
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "review_samples.jsonl"
        self.output = self.root / "pilot"
        self.config_path = self.root / "quality.yaml"
        self.config = yaml.safe_load((REPO / "configs" / "quality.yaml").read_text(encoding="utf-8"))
        self.config.update(limit=10, concurrency=4, max_completion_tokens=128)
        self.write_config()
        self.calls = []
        self.call_lock = threading.Lock()
        # A missing fake must fail loudly rather than reaching any external API.
        self.addCleanup(patch.stopall)
        patch("socket.create_connection", side_effect=AssertionError("Real network forbidden in judge tests")).start()
        patch("socket.socket.connect", side_effect=AssertionError("Real network forbidden in judge tests")).start()
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-never-sent"}).start()

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")

    def write_rows(self, rows):
        self.input.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def transport(self, payload, api_key):
        self.assertEqual(api_key, "test-key-never-sent")
        with self.call_lock:
            self.calls.append(deepcopy(payload))
        return response()

    @staticmethod
    def preflight(config, api_key):
        return {"model": config["model"], "input_usd_per_million": config["input_usd_per_million"],
                "output_usd_per_million": config["output_usd_per_million"]}

    def run_pilot(self, **kwargs):
        return self.quality.run(self.input, config_path=self.config_path,
                                output=kwargs.pop("output", self.output),
                                transport=kwargs.pop("transport", self.transport),
                                preflight=kwargs.pop("preflight", self.preflight), **kwargs)

    def test_dry_run_is_offline_deterministic_and_stratified(self):
        rows = [example(index, source="large") for index in range(30)]
        rows += [example(100, source="small", flags=["needs_review"])]
        self.write_rows(rows)
        def forbidden(*args):
            self.fail("A dry run must not call either preflight or transport")
        with patch.dict(os.environ, {}, clear=True):
            self.run_pilot(limit=2, transport=forbidden, preflight=forbidden)
            other = self.root / "other"
            self.run_pilot(limit=2, output=other, transport=forbidden, preflight=forbidden)
        selected = records(self.output / "sample.jsonl")
        self.assertEqual(selected, records(other / "sample.jsonl"))
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["source"] for row in selected}, {"large", "small"})
        self.assertEqual(records(self.output / "judgments.jsonl"), [])
        self.assertTrue((self.output / "manifest.json").exists())
        self.assertTrue((self.output / "report.md").exists())

    def test_evaluation_rows_are_skipped_before_payload_validation(self):
        training = example(0)
        skipped = [example(1, split="validation"), example(2, split="test"),
                   example(3, is_labeled=False)]
        for row in skipped:
            row.update(messages="do not decode evaluation", tools=42, metadata=None)
        self.write_rows([training, *skipped])
        self.run_pilot(execute=True)
        self.assertEqual([row["id"] for row in records(self.output / "sample.jsonl")], [training["id"]])
        self.assertEqual(len(self.calls), 1)

    def test_duplicate_ids_are_rejected_before_network(self):
        self.write_rows([example(0), example(0, source="different")])
        with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
            self.run_pilot(execute=True)
        self.assertEqual(self.calls, [])

    def test_resume_never_resends_successful_rows(self):
        self.write_rows([example(index) for index in range(3)])
        self.run_pilot(execute=True, max_new_requests=1)
        self.assertEqual(len(self.calls), 1)
        first = records(self.output / "judgments.jsonl")
        self.assertEqual(len(first), 1)
        self.run_pilot(execute=True)
        self.assertEqual(len(self.calls), 3)
        all_results = records(self.output / "judgments.jsonl")
        self.assertEqual(len(all_results), 3)
        self.assertEqual(len({row["id"] for row in all_results}), 3)
        self.run_pilot(execute=True)
        self.assertEqual(len(self.calls), 3)

    def test_changed_input_or_config_refuses_resume(self):
        self.write_rows([example(0)])
        self.run_pilot()
        self.write_rows([example(0), example(1)])
        with self.assertRaises(ValueError):
            self.run_pilot(execute=True)
        self.write_rows([example(0)])
        self.config["system_prompt"] += "\nChanged rubric."
        self.write_config()
        with self.assertRaises(ValueError):
            self.run_pilot(execute=True)
        self.assertEqual(self.calls, [])

    def test_original_instructions_are_data_and_private_annotations_are_omitted(self):
        row = example(0, flags=["FLAG_SENTINEL"], metadata={"prior_verdict": "LABEL_SENTINEL"})
        row["messages"].insert(0, {"role": "system", "content": "PROMPT_INJECTION_SENTINEL"})
        row["messages"].insert(1, {"role": "developer", "content": "DEVELOPER_SENTINEL"})
        row["messages"][-1].update(think="THINK_SENTINEL", reasoning="REASONING_SENTINEL")
        row["tools"] = [{"type": "function", "function": {"name": "unused_tool", "description": "TOOL_SENTINEL",
                          "parameters": {"type": "object", "properties": {}}}}]
        self.write_rows([row])
        self.run_pilot(execute=True)
        payload = self.calls[0]
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        self.assertNotIn("PROMPT_INJECTION_SENTINEL", payload["messages"][0]["content"])
        user_content = payload["messages"][1]["content"]
        for sentinel in ("PROMPT_INJECTION_SENTINEL", "DEVELOPER_SENTINEL", "TOOL_SENTINEL"):
            self.assertIn(sentinel, user_content)
        serialized = json.dumps(payload)
        for sentinel in ("THINK_SENTINEL", "REASONING_SENTINEL", "FLAG_SENTINEL", "LABEL_SENTINEL"):
            self.assertNotIn(sentinel, serialized)
        self.assertNotIn("tools", payload)

    def test_routing_metadata_is_not_presented_as_the_user_request(self):
        row = example(0, task="TASK_SENTINEL", task_hint="HINT_SENTINEL",
                      dialect="DIALECT_SENTINEL")
        payload, _, _ = self.quality.make_payload(row, self.config)
        sample = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        for key in ("task", "task_hint", "dialect", "dialect_hint"):
            self.assertNotIn(key, sample)
        self.assertEqual(sample["messages"], row["messages"])
        for sentinel in ("TASK_SENTINEL", "HINT_SENTINEL", "DIALECT_SENTINEL"):
            self.assertNotIn(sentinel, json.dumps(payload))

    def test_tool_structural_evidence_handles_json_types_without_invented_dates(self):
        row = example(0, task="tool_use", tool_behavior="call")
        row["tools"] = [{"type": "function", "function": {
            "name": "book_hotel", "parameters": {
                "type": "object", "properties": {
                    "guests": {"type": "integer"}, "arrival": {"type": "string"},
                }, "required": ["guests", "arrival"], "additionalProperties": False,
            },
        }}]
        # A string date has no implicit ISO format or year requirement. JSON Schema
        # integers include integral floats but never Python's bool-as-int shortcut.
        cases = [
            ({"guests": 3.0, "arrival": "5 نوفمبر"}, None),
            ({"guests": 3.5, "arrival": "5 نوفمبر"}, "tool_argument_type"),
            ({"guests": True, "arrival": "5 نوفمبر"}, "tool_argument_type"),
            ({"arrival": "5 نوفمبر"}, "missing_required_tool_argument"),
        ]
        for arguments, expected_reason in cases:
            with self.subTest(arguments=arguments):
                row["messages"][-1] = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"type": "function", "function": {
                        "name": "book_hotel", "arguments": json.dumps(arguments),
                    }}],
                }
                payload, _, _ = self.quality.make_payload(row, self.config)
                sample = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
                checks = sample["structural_checks"]
                self.assertEqual(checks["status"], "fail" if expected_reason else "pass")
                if expected_reason:
                    self.assertIn(expected_reason, checks["reasons"])
                else:
                    self.assertEqual(checks["reasons"], [])
                self.assertIn("structure", checks["scope"].lower())
                self.assertIn("semantics", checks["scope"].lower())
                self.assertIn("not checked", checks["scope"].lower())
                self.assertEqual(sample["messages"], row["messages"])

    def test_provider_reasoning_effort_is_checked_against_the_live_catalogue(self):
        model = {
            "id": self.config["model"],
            "supported_parameters": ["structured_outputs", "reasoning", "max_tokens"],
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "reasoning": {"mandatory": True, "supported_efforts": ["minimal", "medium"]},
        }
        for effort, accepted in [("minimal", True), ("medium", True), ("high", False), ("none", False)]:
            with self.subTest(effort=effort):
                config = {**self.config, "reasoning_effort": effort}
                with patch.object(self.quality, "request_json", side_effect=[{}, {"data": [model]}]):
                    if accepted:
                        result = self.quality.check_provider(config, "test-key-never-sent")
                        self.assertEqual(result["model"], model["id"])
                    else:
                        with self.assertRaises(ValueError):
                            self.quality.check_provider(config, "test-key-never-sent")

    def test_provider_mandatory_reasoning_cannot_be_disabled_without_effort_list(self):
        model = {
            "id": self.config["model"],
            "supported_parameters": ["structured_outputs", "reasoning", "max_tokens"],
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            "reasoning": {"mandatory": True},
        }
        config = {**self.config, "reasoning_effort": "none"}
        with patch.object(self.quality, "request_json", side_effect=[{}, {"data": [model]}]):
            with self.assertRaises(ValueError):
                self.quality.check_provider(config, "test-key-never-sent")

    def test_transport_error_is_not_automatically_retried(self):
        self.write_rows([example(0)])
        attempts = []
        def uncertain(payload, api_key):
            attempts.append(payload)
            raise TimeoutError("Response was lost after request submission")
        first = self.run_pilot(execute=True, transport=uncertain)
        resumed = self.run_pilot(execute=True, transport=uncertain)
        self.assertEqual(len(attempts), 1)
        self.assertTrue((self.output / "state.sqlite3").exists())
        self.assertGreater(Decimal(first["accounted_cost_usd"]), 0)
        self.assertEqual(Decimal(first["actual_cost_usd"]), 0)
        self.assertEqual(first["accounted_cost_usd"], resumed["accounted_cost_usd"])

    def test_concurrent_dispatch_reserves_durably_without_exceeding_budget(self):
        rows = [example(index) for index in range(8)]
        reservation = self.quality.make_payload(rows[0], self.config)[1]
        self.config["budget_usd"] = float(reservation * Decimal("2.1"))
        self.write_config()
        self.write_rows(rows)
        rendezvous = threading.Barrier(2)
        snapshots = []

        def paid(payload, api_key):
            # A separate DB reader must already see the reservation when a
            # potentially billable request begins, even with concurrent work.
            connection = sqlite3.connect(self.output / "state.sqlite3")
            try:
                snapshot = connection.execute("SELECT charge, result FROM attempts").fetchall()
            finally:
                connection.close()
            with self.call_lock:
                self.calls.append(payload)
                snapshots.append(snapshot)
            rendezvous.wait(timeout=5)
            return response(cost=str(reservation))

        result = self.run_pilot(execute=True, transport=paid)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["counts"], {"keep": 2})
        self.assertEqual(Decimal(result["accounted_cost_usd"]), reservation * 2)
        self.assertEqual(Decimal(result["unresolved_reservation_usd"]), 0)
        for snapshot in snapshots:
            self.assertTrue(snapshot)
            self.assertTrue(any(result is None for _, result in snapshot))
            self.assertLessEqual(sum(Decimal(charge) for charge, _ in snapshot),
                                 Decimal(str(self.config["budget_usd"])))
        self.run_pilot(execute=True, transport=paid)
        self.assertEqual(len(self.calls), 2)

    def test_oversized_example_is_held_without_truncation_or_api_call(self):
        row = example(0)
        row["messages"][-1]["content"] = "ط" * 100000
        self.write_rows([row])
        result = self.run_pilot(execute=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(records(self.output / "sample.jsonl")[0]["messages"], row["messages"])
        self.assertEqual(records(self.output / "judgments.jsonl")[0]["status"], "skipped")
        self.assertEqual(Decimal(result["accounted_cost_usd"]), 0)

    def test_tiny_budget_prevents_requests(self):
        self.config["budget_usd"] = 0.000001
        self.write_config()
        self.write_rows([example(index) for index in range(10)])
        self.run_pilot(execute=True)
        self.assertEqual(self.calls, [])

    def test_budget_override_cannot_raise_configured_limit(self):
        self.write_rows([example(0)])
        with self.assertRaises(ValueError):
            self.run_pilot(execute=True, budget_usd=6)
        self.assertEqual(self.calls, [])

    def test_charge_above_reservation_stops_dispatch_and_is_not_counted_as_keep(self):
        self.config["concurrency"] = 1
        self.write_config()
        self.write_rows([example(0), example(1)])
        reservation = self.quality.make_payload(example(0), self.config)[1]
        cost = reservation * 2

        def unexpected_cost(payload, api_key):
            self.calls.append(payload)
            return response(cost=str(cost))

        result = self.run_pilot(execute=True, transport=unexpected_cost)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["status"], "stopped_on_error")
        self.assertEqual(result["counts"], {"unjudged": 1})
        self.assertEqual(Decimal(result["accounted_cost_usd"]), cost)
        self.assertEqual(records(self.output / "judgments.jsonl")[0]["status"], "error")

    def test_provider_rate_error_retains_reservation_without_resending(self):
        self.write_rows([example(0)])
        def throttled(payload, api_key):
            self.calls.append(payload)
            return {"id": "gen-throttled", "error": {"code": 429, "message": "upstream rate limit"}}
        self.run_pilot(execute=True, transport=throttled)
        self.run_pilot(execute=True, transport=throttled)
        self.assertEqual(len(self.calls), 1)
        result = records(self.output / "judgments.jsonl")[0]
        self.assertEqual(result["error_code"], 429)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["accounted_cost_usd"], result["reservation_usd"])

    def test_specialized_rubric_and_normalized_provider_token_limit(self):
        row = example(0, task="translation", task_hint="creative_writing")
        payload, _, _ = self.quality.make_payload(row, self.config)
        self.assertIn(self.config["rubrics"]["translation"], payload["messages"][0]["content"])
        self.assertNotIn(self.config["rubrics"]["creative_writing"], payload["messages"][0]["content"])
        self.assertEqual(payload["max_tokens"], self.config["max_completion_tokens"])
        self.assertNotIn("max_completion_tokens", payload)

    def test_modified_saved_sample_refuses_resume(self):
        self.write_rows([example(0)])
        self.run_pilot()
        (self.output / "sample.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_pilot(execute=True)
        self.assertEqual(self.calls, [])

    def test_invalid_verdict_is_not_recorded_as_keep_or_retried(self):
        self.write_rows([example(0)])
        attempts = []
        def inconsistent(payload, api_key):
            attempts.append(payload)
            return response(verdict("keep", correctness="uncertain"))
        self.run_pilot(execute=True, transport=inconsistent)
        self.run_pilot(execute=True, transport=inconsistent)
        self.assertEqual(len(attempts), 1)
        self.assertFalse(any(row.get("judgment", {}).get("decision") == "keep"
                             for row in records(self.output / "judgments.jsonl")))

    def test_reason_control_characters_are_rejected_without_rejecting_arabic_or_layout(self):
        for character in ("\u0000", "\u0006", "\r", "\u001f", "\u007f"):
            with self.subTest(character=repr(character)):
                result = verdict()
                result["reasons"] = [f"Corrupted quoted fragment: {character}39"]
                with self.assertRaises(ValueError):
                    self.quality.validate_judgment(result)
        valid = verdict()
        valid["reasons"] = ["الإجابة تتبع الطلب.\nEvidence:\tcomplete response."]
        self.assertEqual(self.quality.validate_judgment(valid), valid)

    def test_parsed_control_character_response_is_unjudged_but_retains_billed_cost(self):
        self.write_rows([example(0)])

        def corrupted(payload, api_key):
            self.calls.append(payload)
            result = verdict()
            result["reasons"] = ["The answer includes '\u000639'."]
            return response(result, cost="0.00021")

        result = self.run_pilot(execute=True, transport=corrupted)
        self.assertEqual(result["status"], "stopped_on_error")
        self.assertEqual(result["counts"], {"unjudged": 1})
        self.assertEqual(Decimal(result["actual_cost_usd"]), Decimal("0.00021"))
        self.assertEqual(Decimal(result["accounted_cost_usd"]), Decimal("0.00021"))
        saved = records(self.output / "judgments.jsonl")[0]
        self.assertEqual(saved["status"], "error")
        self.assertNotIn("judgment", saved)
        self.run_pilot(execute=True, transport=corrupted)
        self.assertEqual(len(self.calls), 1)

    def test_matching_provisional_labels_are_selected_without_leaking_to_judge(self):
        rows = [example(index) for index in range(20)]
        reference = rows[-1]
        labels = self.root / "calibration.jsonl"
        labels.write_text(json.dumps({
            "id": reference["id"], "example_hash": reference["example_hash"],
            "input_hash": reference["input_hash"], "verdict": "keep", "reviewer": "assistant_static",
            "status": "proposed_not_applied", "evidence": "REFERENCE_SENTINEL",
        }) + "\n", encoding="utf-8")
        self.write_rows(rows)
        result = self.run_pilot(execute=True, limit=1, labels_path=labels)
        self.assertEqual([row["id"] for row in records(self.output / "sample.jsonl")], [reference["id"]])
        self.assertNotIn("REFERENCE_SENTINEL", json.dumps(self.calls))
        self.assertIs(result.get("training_ready"), False)
        self.assertIs(result.get("judge_validated"), False)
        self.assertEqual(result["reference_comparison"]["scope"], "provisional_agreement_only")
        report = (self.output / "report.md").read_text(encoding="utf-8").lower()
        self.assertIn("provisional", report)


if __name__ == "__main__":
    unittest.main()
