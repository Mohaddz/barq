"""A bounded, resumable OpenRouter judge pilot over existing review JSONL samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from barq.rules import validate_example


API = "https://openrouter.ai/api/v1"
DECISIONS = ("keep", "review", "repair", "drop")
DIMENSIONS = ("correctness", "language_quality", "instruction_following")
RATINGS = ("pass", "fail", "uncertain", "not_applicable")
GROUP_FIELDS = ("source", "task", "task_hint", "dialect", "tool_behavior")
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["decision", "reasons", "dimensions"],
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "dimensions": {"type": "object", "additionalProperties": False,
                       "required": list(DIMENSIONS),
                       "properties": {key: {"type": "string", "enum": list(RATINGS)} for key in DIMENSIONS}},
    },
}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else json_text(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def money(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("Invalid monetary amount.") from error
    if not result.is_finite() or result < 0:
        raise ValueError("Monetary amounts must be finite and nonnegative.")
    return result


def read_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "model", "reasoning_effort", "max_completion_tokens", "concurrency",
                "budget_usd", "input_usd_per_million", "output_usd_per_million", "seed", "limit",
                "max_input_bytes", "system_prompt", "rubrics"}
    if not isinstance(config, dict) or set(config) != expected or config["schema_version"] != 1:
        raise ValueError("Quality config requires schema_version 1 and the documented settings.")
    if (not isinstance(config["model"], str) or len(config["model"].split("/")) != 2
            or not all(config["model"].split("/")) or any(char.isspace() for char in config["model"])):
        raise ValueError("model must be an explicit OpenRouter provider/model ID.")
    for key, maximum in (("concurrency", 16), ("limit", 10000), ("max_completion_tokens", 16384),
                         ("max_input_bytes", 200000)):
        if type(config[key]) is not int or not 1 <= config[key] <= maximum:
            raise ValueError(f"{key} must be an integer from 1 to {maximum}.")
    if type(config["seed"]) is not int or config["reasoning_effort"] not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Invalid seed or reasoning_effort.")
    for key in ("budget_usd", "input_usd_per_million", "output_usd_per_million"):
        if not money(config[key]):
            raise ValueError(f"{key} must be positive.")
    if not isinstance(config["system_prompt"], str) or not config["system_prompt"].strip():
        raise ValueError("system_prompt cannot be empty.")
    if not isinstance(config["rubrics"], dict) or not config["rubrics"].get("default") or any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in config["rubrics"].items()
    ):
        raise ValueError("rubrics must contain nonempty strings and a default rubric.")
    return config


def read_jsonl(path):
    with path.open("rb") as stream:
        raw = stream.read(50 * 1024 * 1024 + 1)
    if len(raw) > 50 * 1024 * 1024:
        raise ValueError("Use a bounded review JSONL sample (at most 50 MiB), not the full dataset.")
    rows = []
    for line in raw.decode("utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Each JSONL record must be an object.")
            rows.append(row)
    return rows, digest(raw)


def load_samples(path):
    rows, source_hash = read_jsonl(path)
    selected, seen, skipped = [], set(), Counter()
    for row in rows:
        # Official dev/test and unlabeled examples never become judge inputs.
        if row.get("split") != "train" or row.get("is_labeled") is not True or row.get("original_split") != "train":
            skipped["outside_labeled_train"] += 1
            continue
        for key in ("id", "source", "task", "example_hash", "input_hash", "revision"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Training sample requires {key}.")
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        seen.add(row["id"])
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
            for message in messages
        ) or not any(message["role"] == "assistant" for message in messages):
            raise ValueError(f"{row['id']}: invalid conversation.")
        if not isinstance(row.get("tools", []), list):
            raise ValueError(f"{row['id']}: tools must be a list.")
        selected.append(row)
    if not selected:
        raise ValueError("No labeled training samples found.")
    return selected, source_hash, dict(skipped)


def load_labels(path, samples):
    if path is None:
        return {}, None
    rows, file_hash = read_jsonl(path)
    candidates = {row["id"]: row for row in samples}
    labels = {}
    for row in rows:
        row_id = row.get("id")
        if row_id not in candidates:
            continue
        if row_id in labels or row.get("example_hash") != candidates[row_id]["example_hash"]:
            raise ValueError("Reference labels contain duplicate IDs or mismatched example hashes.")
        verdict = row.get("verdict", row.get("decision"))
        scope = "whole_row_provisional"
        if candidates[row_id].get("tool_behavior") == "call":
            verdict = row.get("call_target_verdict")
            scope = "call_target_only"
            if verdict is None:
                continue  # Historical reasoning-only judgments do not match this pilot's scope.
        verdict = "drop" if verdict == "exclude" else verdict
        if verdict not in DECISIONS:
            raise ValueError("Reference verdict must be keep, review, repair, drop or exclude.")
        labels[row_id] = {"decision": verdict, "scope": scope,
                          "reviewer": row.get("reviewer", "unspecified"),
                          "status": row.get("status", "provisional")}
    return labels, file_hash


def select_samples(rows, labels, limit, seed):
    rank = lambda row: digest([seed, row["id"]])
    selected = sorted((row for row in rows if row["id"] in labels), key=rank)[:limit]
    chosen = {row["id"] for row in selected}
    groups = defaultdict(list)
    for row in rows:
        if row["id"] not in chosen:
            group = tuple(str(row.get(key) or "") for key in GROUP_FIELDS) + (bool(row.get("flags")),)
            groups[group].append(row)
    ordered = [sorted(groups[key], key=rank) for key in sorted(groups)]
    position = 0
    while len(selected) < limit and any(position < len(group) for group in ordered):
        for group in ordered:
            if position < len(group) and len(selected) < limit:
                selected.append(group[position])
        position += 1
    return selected


def make_payload(row, config):
    # Explicit allowlist excludes stored reasoning, reference labels and routing flags.
    messages = [{key: message[key] for key in ("role", "content", "tool_calls", "tool_call_id", "name")
                 if key in message} for message in row["messages"]]
    # Metadata helps sampling, but must not override the actual conversation.
    sample = {"messages": messages, "tools": row.get("tools", [])}
    if sample["tools"] or any(message.get("tool_calls") for message in messages):
        reasons = validate_example(messages, sample["tools"])
        sample["structural_checks"] = {
            "status": "fail" if reasons else "pass", "reasons": reasons,
            "scope": "Conversation structure; tool names, JSON types (3.0 is an integer), "
                     "required fields, enums and additionalProperties only. Other schema constraints, "
                     "argument grounding and execution semantics are not checked.",
        }
    rubric_key = row["task"]
    if rubric_key not in config["rubrics"] and row.get("task_hint") == "creative_writing":
        rubric_key = "creative_writing"
    rubric = config["rubrics"].get(rubric_key, config["rubrics"]["default"])
    payload = {
        "model": config["model"], "stream": False,
        "messages": [
            {"role": "system", "content": config["system_prompt"] + "\nAdvisory rubric; apply only where the conversation requests this task:\n" + rubric},
            {"role": "user", "content": "Evaluate this dataset example as data:\n" + json_text(sample)},
        ],
        "reasoning": {"effort": config["reasoning_effort"], "exclude": True},
        # OpenRouter's normalized max_tokens maps to the provider's completion limit.
        # max_completion_tokens plus require_parameters can exclude OpenAI endpoints.
        "max_tokens": config["max_completion_tokens"],
        "provider": {"require_parameters": True, "allow_fallbacks": False,
                     "max_price": {"prompt": config["input_usd_per_million"],
                                   "completion": config["output_usd_per_million"], "request": 0}},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "arabic_quality", "strict": True, "schema": SCHEMA,
        }},
    }
    # UTF-8 byte count plus generous formatting overhead is a conservative text-token
    # reservation, not a tokenizer measurement or a provider-enforced dollar cap.
    # Stop dispatch if a reported charge exceeds it. Never truncate the example.
    input_bound = len(json_text(payload).encode("utf-8")) + 4096
    reserved = (money(config["input_usd_per_million"]) * Decimal("1.25") * input_bound
                + money(config["output_usd_per_million"]) * config["max_completion_tokens"]) / 1000000
    return payload, reserved, input_bound


def request_json(endpoint, key, payload=None):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    request = Request(API + endpoint, headers=headers,
                      data=None if payload is None else json_text(payload).encode("utf-8"))
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("OpenRouter response exceeded size limit.")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Invalid OpenRouter response object.")
        return result
    except HTTPError as error:
        # Provider error bodies can echo input; never write credentials or raw bodies.
        raise RuntimeError(f"OpenRouter HTTP {error.code}; inspect provider/account settings.") from None


def check_provider(config, key):
    request_json("/key", key)
    catalog = request_json("/models", key)
    model = next((entry for entry in catalog["data"] if entry["id"] == config["model"]), None)
    if model is None or not {"structured_outputs", "reasoning", "max_tokens"}.issubset(model.get("supported_parameters", [])):
        raise ValueError("Selected model is unavailable or lacks the required parameters.")
    reasoning = model.get("reasoning") or {}
    effort = config["reasoning_effort"]
    if (reasoning.get("mandatory") and effort == "none"
            or reasoning.get("supported_efforts") and effort not in reasoning["supported_efforts"]):
        raise ValueError("Selected model does not support the configured reasoning effort.")
    pricing = model["pricing"]
    if (money(pricing["prompt"]) * 1000000 > money(config["input_usd_per_million"])
            or money(pricing["completion"]) * 1000000 > money(config["output_usd_per_million"])
            or money(pricing.get("request", 0)) > 0):
        raise ValueError("Live provider pricing exceeds the configured ceilings.")
    return {"checked_at": now(), "model": model["id"], "pricing": pricing, "reasoning": reasoning}


def judge_request(payload, key):
    return request_json("/chat/completions", key, payload)


def validate_judgment(value):
    if not isinstance(value, dict) or set(value) != {"decision", "reasons", "dimensions"}:
        raise ValueError("Invalid judgment keys.")
    if value["decision"] not in DECISIONS or not isinstance(value["reasons"], list) or not 1 <= len(value["reasons"]) <= 3:
        raise ValueError("Invalid judgment decision or reasons.")
    if any(not isinstance(reason, str) or not reason.strip() or len(reason) > 2000 for reason in value["reasons"]):
        raise ValueError("Judgment reasons must be short nonempty strings.")
    if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127
           for reason in value["reasons"] for char in reason):
        raise ValueError("Judgment reasons contain invalid control characters.")
    dimensions = value["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS) or any(rating not in RATINGS for rating in dimensions.values()):
        raise ValueError("Invalid judgment dimensions.")
    if value["decision"] == "keep" and any(rating in {"fail", "uncertain"} for rating in dimensions.values()):
        raise ValueError("Keep contradicts a failed or uncertain dimension.")
    if value["decision"] in {"repair", "drop"} and "fail" not in dimensions.values():
        raise ValueError("Repair/drop requires a concrete failed dimension.")
    return value


def evaluate(payload, key, transport):
    try:
        response = transport(payload, key)
    except Exception as error:
        # Do not automatically retry an uncertain request: it may already be billed.
        message = str(error) if isinstance(error, RuntimeError) and str(error).startswith("OpenRouter HTTP ") else type(error).__name__
        return {"status": "error", "error": message}
    result = {"status": "error"}
    try:
        if not isinstance(response, dict):
            raise ValueError("Invalid OpenRouter response object.")
        if response.get("error"):
            error = response["error"]
            code = error.get("code", "unknown") if isinstance(error, dict) else "unknown"
            result.update(generation_id=response.get("id"), error_code=code)
            raise ValueError(f"OpenRouter error {code}; " + ("upstream rate limit, retry later with a new reviewed attempt."
                             if str(code) == "429" else "inspect provider/account settings."))
        usage = response.get("usage", {})
        result.update(generation_id=response.get("id"), model=response.get("model"), provider=response.get("provider"))
        cost = money(usage["cost"])
        result.update(actual_cost_usd=str(cost), usage={key: usage.get(key) for key in
                      ("prompt_tokens", "completion_tokens", "completion_tokens_details", "prompt_tokens_details")})
        if response.get("model") != payload["model"]:
            raise ValueError("Provider returned a different model.")
        choices = response["choices"]
        if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise ValueError("Incomplete or refused judge response.")
        result["judgment"] = validate_judgment(json.loads(choices[0]["message"]["content"]))
        result["status"] = "complete"
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["error"] = str(error) if type(error) is ValueError else "Invalid response schema or missing usage."
    return result


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a+b") as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Another process is using this quality output directory.") from None
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish(output, manifest, connection, selected, labels):
    attempts = {row_id: (reservation, charge, json.loads(result) if result else None)
                for row_id, reservation, charge, result in connection.execute(
                    "SELECT id, reservation, charge, result FROM attempts")}
    decisions, groups, comparisons, confusion = Counter(), defaultdict(Counter), [], Counter()
    judgments = []
    actual, accounted = Decimal(0), Decimal(0)
    for row in selected:
        if row["id"] not in attempts:
            continue
        reservation, charge, result = attempts[row["id"]]
        accounted += money(charge)
        result = result or {"status": "unknown", "error": "Interrupted request; reservation retained, not retried."}
        actual += money(result.get("actual_cost_usd", 0))
        record = {key: row.get(key) for key in ("id", "dataset", "revision", "example_hash", "input_hash", *GROUP_FIELDS)}
        record.update(result, reservation_usd=reservation, accounted_cost_usd=charge)
        decision = result["judgment"]["decision"] if result["status"] == "complete" else "unjudged"
        decisions[decision] += 1
        groups[(row["source"], row["task"])][decision] += 1
        if row["id"] in labels and result["status"] == "complete":
            reference = labels[row["id"]]
            record["reference"] = reference
            comparisons.append({"id": row["id"], "reference": reference["decision"], "judge": decision})
            confusion[(reference["decision"], decision)] += 1
        judgments.append(record)
    temporary = output / "judgments.jsonl.tmp"
    temporary.write_text("".join(json_text(row) + "\n" for row in judgments), encoding="utf-8")
    temporary.replace(output / "judgments.jsonl")
    manifest.update(updated_at=now(), counts=dict(decisions), attempted_rows=len(attempts),
                    pending_rows=len(selected) - len(attempts), actual_cost_usd=str(actual),
                    accounted_cost_usd=str(accounted),
                    unresolved_reservation_usd=str(accounted - actual),
                    reference_comparison={"scope": "provisional_agreement_only", "rows": len(comparisons),
                        "agreements": sum(row["reference"] == row["judge"] for row in comparisons),
                        "reference_nonkeep_judge_keep": sum(row["reference"] != "keep" and row["judge"] == "keep" for row in comparisons),
                        "confusion": [{"reference": a, "judge": b, "rows": count} for (a, b), count in sorted(confusion.items())]})
    write_json(output / "manifest.json", manifest)
    lines = ["# Arabic quality judge pilot", "", f"Status: **{manifest['status']}**; selected: **{len(selected):,}**.",
             f"Model: `{manifest['config']['model']}`; reported inference cost: **${actual:.6f}**; "
             f"cost plus unresolved reservations: **${accounted:.6f}**; budget: **${money(manifest['config']['budget_usd']):.2f}**.",
             "", "This is a balanced sample of an existing review sample, not a corpus quality estimate or SFT export.",
             "Original data, benchmark status and training splits are unchanged. All decisions require validation before scaling.",
             "", "## Decisions", "", "| Source | Task | Keep | Review | Repair | Drop | Unjudged |", "|---|---|---:|---:|---:|---:|---:|"]
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    for (source, task), counts in sorted(groups.items()):
        lines.append(f"| {clean(source)} | {clean(task)} | " + " | ".join(str(counts[key]) for key in (*DECISIONS, "unjudged")) + " |")
    matched = manifest["reference_comparison"]
    lines += ["", "## Reference comparison", "",
              f"Compared {matched['rows']} examples; {matched['agreements']} exact decision agreements.",
              f"Judge kept {matched['reference_nonkeep_judge_keep']} examples with a non-keep reference: inspect these disagreements first.",
              "Existing assistant assessments are provisional and informed rubric design. This is NOT held-out accuracy, "
              "a measured false-accept rate, or human certification. References never enter judge requests.",
              "For AISA calls, comparison uses call_target_verdict when available; stored reasoning is omitted from all judge payloads.",
              "", "## Next step", "", "Review disagreements and an independent sample of keeps. Adjudicate task/family-disjoint "
              "examples with qualified Arabic reviewers before judging reliability. Religious/factual uncertainty needs sources; "
              "a second model's agreement is not verification. Repair suggestions are decisions only; no answers were rewritten.",
              "", "Budget uses durable reservations before dispatch, provider price ceilings, billed usage.cost, and bounded output tokens "
              "including reasoning. Uncertain requests retain their full reservation and are never automatically retried. "
              "The limit applies to this output directory and inference calls, excluding credit-purchase fees and unrelated usage.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_path, *, config_path, output, labels_path=None, limit=None, execute=False, budget_usd=None,
        max_new_requests=None, transport=None, preflight=None):
    config = read_config(config_path)
    if budget_usd is not None:
        if not money(budget_usd) or money(budget_usd) > money(config["budget_usd"]):
            raise ValueError("budget_usd must be positive and no greater than the configured budget.")
        config["budget_usd"] = float(budget_usd)
    if limit is not None:
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("limit must be from 1 to 10000.")
        config["limit"] = limit
    if max_new_requests is not None and (type(max_new_requests) is not int or max_new_requests < 1):
        raise ValueError("max_new_requests must be positive.")
    rows, input_hash, skipped = load_samples(input_path)
    labels, labels_hash = load_labels(labels_path, rows)
    selected = select_samples(rows, labels, config["limit"], config["seed"])
    implementation = {name: digest(Path(__file__).with_name(name).read_bytes())
                      for name in ("quality.py", "rules.py")}
    identity = {"config": config, "input_sha256": input_hash, "labels_sha256": labels_hash,
                "sample_sha256": digest(selected), "implementation_sha256": digest(implementation)}
    output = output.resolve()
    with run_lock(output), closing(sqlite3.connect(output / "state.sqlite3")) as connection:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS meta (identity TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, reservation TEXT NOT NULL, "
                           "charge TEXT NOT NULL, request_sha256 TEXT NOT NULL, result TEXT)")
        previous = connection.execute("SELECT identity FROM meta").fetchone()
        if previous and previous[0] != json_text(identity):
            raise ValueError("Cannot resume: input, references, config, selection or implementation changed; use a new output directory.")
        if not previous:
            connection.execute("INSERT INTO meta VALUES (?)", (json_text(identity),))
            connection.commit()
        manifest = {"schema_version": 1, "mode": "quality", "status": "dry_run", **identity,
                    "input": str(input_path.resolve()), "labels": str(labels_path.resolve()) if labels_path else None,
                    "selected_rows": len(selected), "available_training_rows": len(rows), "skipped_input_rows": skipped,
                    "training_ready": False, "benchmarks": "not_checked", "judge_validated": False,
                    "selection_method": "references first, then deterministic round-robin by source/task/hint/dialect/call/flag status"}
        sample_path = output / "sample.jsonl"
        sample_text = "".join(json_text(row) + "\n" for row in selected)
        if sample_path.exists() and sample_path.read_text(encoding="utf-8") != sample_text:
            raise ValueError("Saved sample changed; restore it before resuming this output directory.")
        if not sample_path.exists():
            sample_path.write_text(sample_text, encoding="utf-8")
        publish(output, manifest, connection, selected, labels)
        if not execute:
            print(f"Dry run: {len(selected):,} selected; no API calls. Report: {output / 'report.md'}", flush=True)
            return manifest
        seen = {row[0] for row in connection.execute("SELECT id FROM attempts")}
        remaining = [row for row in selected if row["id"] not in seen]
        if not remaining:
            manifest["status"] = "complete" if all(result and json.loads(result)["status"] == "complete"
                for (result,) in connection.execute("SELECT result FROM attempts")) else "complete_with_errors"
            publish(output, manifest, connection, selected, labels)
            return manifest
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("Set OPENROUTER_API_KEY in your environment; never put it in the config or command arguments.")
        manifest["provider_check"] = (preflight or check_provider)(config, key)
        manifest["status"] = "running"
        budget = money(config["budget_usd"])
        charged = sum((money(row[0]) for row in connection.execute("SELECT charge FROM attempts")), Decimal(0))
        launched, position, stop = 0, 0, None
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            active = {}
            try:
                while (position < len(remaining) and not stop) or active:
                    while position < len(remaining) and len(active) < config["concurrency"] and not stop:
                        if max_new_requests is not None and launched >= max_new_requests:
                            stop = "request_limit"
                            break
                        row = remaining[position]
                        payload, reserved, input_bound = make_payload(row, config)
                        if input_bound > config["max_input_bytes"]:
                            result = {"status": "skipped", "error": "Input too large; not truncated or sent."}
                            connection.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?)",
                                               (row["id"], "0", "0", digest(payload), json_text(result)))
                            connection.commit()
                            position += 1
                            continue
                        if charged + reserved > budget:
                            if active:
                                break  # Wait for actual costs to release excess reservations.
                            stop = "budget_exhausted"
                            break
                        connection.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, NULL)",
                                           (row["id"], str(reserved), str(reserved), digest(payload)))
                        connection.commit()  # Durable before any potentially billable request.
                        charged += reserved
                        active[executor.submit(evaluate, payload, key, transport or judge_request)] = (row, reserved)
                        launched += 1
                        position += 1
                    if active:
                        finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                        for future in finished:
                            row, reserved = active.pop(future)
                            result = future.result()
                            cost = money(result.get("actual_cost_usd", reserved))
                            charged += cost - reserved
                            if cost > reserved:
                                result.update(status="error", error="Provider cost exceeded reservation; dispatch stopped.")
                            connection.execute("UPDATE attempts SET charge=?, result=? WHERE id=?",
                                               (str(cost), json_text(result), row["id"]))
                            connection.commit()
                            if result["status"] != "complete":
                                stop = "stopped_on_error"
                            publish(output, manifest, connection, selected, labels)
                            if manifest["attempted_rows"] % 25 == 0 or result["status"] != "complete":
                                print(f"quality: {manifest['counts']} / {len(selected):,}; "
                                      f"reported ${manifest['actual_cost_usd']}; reserved total ${charged:.4f}", flush=True)
                manifest["status"] = stop or ("complete" if all(result and json.loads(result)["status"] == "complete"
                    for (result,) in connection.execute("SELECT result FROM attempts")) else "complete_with_errors")
            except BaseException:
                manifest["status"] = "interrupted"
                raise
            finally:
                publish(output, manifest, connection, selected, labels)
        print(f"Quality {manifest['status']}: {manifest['counts']}; cost ${manifest['actual_cost_usd']}\n"
              f"Report: {output / 'report.md'}", flush=True)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing review/curation JSONL sample; no dataset downloads")
    parser.add_argument("--config", type=Path, default=Path("configs/quality.yaml"))
    parser.add_argument("--labels", type=Path, help="Optional provisional reference JSONL; never sent to the judge")
    parser.add_argument("--output", type=Path, required=True, help="Persistent pilot directory; repeat the same command to resume")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--budget-usd", type=float, help="Optional lower spending cap for this run")
    parser.add_argument("--execute", action="store_true", help="Make paid API calls; default is an offline dry run")
    parser.add_argument("--max-new-requests", type=int, help="Optional smoke-test bound within the same resumable pilot")
    args = parser.parse_args(argv)
    try:
        result = run(args.input, config_path=args.config, output=args.output, labels_path=args.labels,
                     limit=args.limit, execute=args.execute, budget_usd=args.budget_usd,
                     max_new_requests=args.max_new_requests)
        if result["status"] in {"stopped_on_error", "budget_exhausted", "complete_with_errors"}:
            raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same output path; uncertain requests will not be resent.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


def cli():
    # Keep the preparation implementation untouched so its persisted runs remain reusable.
    if len(sys.argv) > 1 and sys.argv[1] == "quality":
        return main(sys.argv[2:])
    from barq.data import main as data_main
    return data_main()


if __name__ == "__main__":
    main()
