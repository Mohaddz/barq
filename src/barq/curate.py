"""Offline curation decisions from a completed prepare run; no model calls or repairs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from barq.data import CANDIDATE_SCHEMA, COMMON_FIELDS, ParquetSink, json_text
from barq.review import GROUP_FIELDS as REVIEW_GROUP_FIELDS, Samples, markdown_cell, open_inputs
from barq.rules import curation_checks, validate_example


DECISIONS = ("accept", "review", "repair", "exclude")
GROUP_FIELDS = (*REVIEW_GROUP_FIELDS, "decision")
CURATION_SCHEMA = pa.schema(COMMON_FIELDS + [(name, pa.string()) for name in (
    "example_hash", "input_hash", "task_hint", "dialect", "tool_behavior",
    "decision", "reasons_json", "flags_json", "checks_json", "scope",
)])


def read_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "seed", "batch_size", "sample_per_group",
                "repair_flags", "review_sources", "review_tasks"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("Curation config requires exactly: " + ", ".join(sorted(expected)))
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Curation schema_version must be 1.")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer.")
    for name in ("batch_size", "sample_per_group"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    codes = config["repair_flags"]
    if not isinstance(codes, list) or any(not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in codes):
        raise ValueError("repair_flags must be a list of reason codes.")
    if len(set(codes)) != len(codes):
        raise ValueError("repair_flags cannot contain duplicates.")
    for name in ("review_sources", "review_tasks"):
        rules = config[name]
        if not isinstance(rules, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", value) for key, value in rules.items()
        ):
            raise ValueError(f"{name} must map source/task names to reason codes.")
    return config


def assess(row, config):
    """Inspect one training candidate without modifying its text, metadata or hashes."""
    try:
        messages, tools, metadata = (json.loads(row[name]) for name in
                                     ("messages_json", "tools_json", "metadata_json"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{row['id']}: invalid prepared JSON; input may be corrupt.") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{row['id']}: metadata_json must contain an object.")
    structural = validate_example(messages, tools)
    result = {"flags": [], "checks": [], "task_hint": None} if structural else curation_checks(
        messages, tools, row["source"], row["task"])
    flags = sorted(set(result["flags"]))
    if any(code.endswith("_skipped_unknown_wrapper") for code in result["checks"]):
        flags = sorted(set([*flags, "unsupported_task_wrapper"]))
    checks = sorted(set(["structure", "semantic_quality_skipped_offline", *result["checks"]]))
    policy = [rules[key] for rules, key in ((config["review_sources"], row["source"]),
                                           (config["review_tasks"], row["task"])) if key in rules]
    reasons = sorted(set([*structural, *flags, *policy]))
    decision = ("exclude" if structural else "repair" if set(flags).intersection(config["repair_flags"])
                else "review" if flags or policy else "accept")
    dialect = str(metadata.get("dialect") or "unspecified") if row["task"] == "tool_use" else ""
    behavior = ""
    if row["task"] == "tool_use":
        behavior = "call" if isinstance(messages, list) and any(
            isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls")
            for message in messages) else "no_call"
    record = {key: row[key] for key in CANDIDATE_SCHEMA.names if not key.endswith("_json")}
    record.update(task_hint=result["task_hint"] or "", dialect=dialect, tool_behavior=behavior,
                  decision=decision, reasons_json=json_text(reasons or ["offline_checks_passed"]),
                  flags_json=json_text(flags), checks_json=json_text(checks), scope="offline_checks_only")
    sample = {**record, "messages": messages, "tools": tools, "metadata": metadata,
              "reasons": reasons or ["offline_checks_passed"], "flags": flags, "checks": checks,
              "review": {"decision": None, "notes": ""}}
    for name in ("reasons_json", "flags_json", "checks_json"):
        sample.pop(name)
    return record, sample


def write_report(path, manifest):
    counts = manifest["counts"]
    lines = ["# Phase 1 offline curation", "", f"Prepare run: `{manifest['input_run_id']}`", "",
             f"Assessed **{manifest['training_rows']:,} labeled training rows**. "
             + "; ".join(f"{name}: **{counts[name]:,}**" for name in DECISIONS) + ".", "",
             "Acceptance means passing the implemented offline checks only. Semantic quality, "
             "fact verification and benchmark readiness are not certified. This is not a balanced SFT export.", "",
             "## Outputs", "",
             "- `candidates.parquet`: accepted labeled training rows, with original text and hashes unchanged.",
             "- `decisions.parquet`: one decision per assessed training row, including reasons, checks and provenance.",
             "- `review_samples.jsonl`: bounded samples from every observed source/task/hint/dialect/call/decision group.",
             "- `manifest.json`: input identity, configuration, implementation hash, coverage and run status.", "",
             "`repair` requests a correction and recheck; no replacement answer was generated. "
             "`review` includes uncertain signals and configured source/task holds. "
             "`exclude` marks structural invalidity in the current row. No original rows were deleted.", "",
             "Preparation exclusions/quarantines remain in the original prepare decision log. "
             "Evaluation and unlabeled rows remain in the original prepared files; they are never sampled "
             "or exported as training candidates here. Splits and source weights were not changed.", "",
             "## Decision coverage", "", "| Source | Task | Hint | Dialect | Calls | Decision | Rows | Sampled |",
             "|---|---|---|---|---|---|---:|---:|"]
    for group in manifest["groups"]:
        labels = [group[key] or "—" for key in GROUP_FIELDS if key != "dataset"]
        lines.append("| " + " | ".join(map(markdown_cell, labels))
                     + f" | {group['population']:,} | {group['sampled']:,} |")
    for title, field in (("Reasons (can overlap)", "reason_counts"), ("Detected flags", "flag_counts"),
                         ("Applied checks", "check_counts"), ("Checks not completed", "skipped_check_counts"),
                         ("Preserved candidate rows outside training", "skipped_candidate_rows")):
        lines += ["", f"## {title}", ""]
        lines += [f"- `{code}`: {count:,}" for code, count in sorted(manifest[field].items())] or ["- None."]
    lines += ["", "## Limits and next step", "",
              "Numeric checks flag tokens not found in supplied context; arithmetic, relative dates, "
              "unit conversions and spelling can be valid explanations. These signals request review, "
              "not an automatic finding of fabrication. Schema checks cannot interpret all constraints "
              "written in tool descriptions. Python is parsed but never executed. Label vocabulary and "
              "diacritic preservation do not establish semantic correctness.", "",
              "Stored assistant reasoning is preserved in intermediate data and flagged for a supervision "
              "decision; it has not been approved for training. Generic AISA no-call targets need a "
              "conversational adaptation decision. Alignment, creative quality, religion, factual knowledge "
              "and dialect authenticity need calibrated semantic review.", "",
              "Inspect the samples, adjudicate a small calibration set, then measure a judge on separate "
              "examples before a larger paid pass. Preserve provenance when repairing targets. "
              "Configure benchmark references and balance accepted data before the final SFT export.", "",
              "## Benchmark status", "", "Inherited from preparation; no checks were added or rerun.", ""]
    lines += [f"- {item['name']}: **{item['status']}**" for item in manifest["benchmarks"]] or ["- NOT CHECKED: no references configured."]
    lines += ["", f"Data directory: `{manifest['output']}`", "",
              "Share `curation.md` and `manifest.json` first. The full prepared dataset can stay on the VM. "
              "Generated reports and data are ignored by Git; retain or back up the storage before deleting the VM.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(input_dir, *, config_path=Path("configs/curation.yaml"), manifest_path=None, output_root=None):
    started = time.perf_counter()
    config_path = config_path.resolve(strict=True)
    config = read_config(config_path)
    input_dir, manifest_path, preparation, manifest_hash, files = open_inputs(input_dir, manifest_path)
    if output_root is None:
        standard = (len(manifest_path.parents) >= 4 and manifest_path.parent.name == preparation["run_id"]
                    and manifest_path.parents[1].name == "prepare" and manifest_path.parents[2].name == "reports")
        output_root = manifest_path.parents[3] if standard else Path.cwd()
    root = output_root.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output, report = (root / "data" / "processed" / "curate" / run_id, root / "reports" / "curate" / run_id)
    output.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "mode": "curate", "status": "running", "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(), "input_run_id": preparation["run_id"],
        "input_manifest": str(manifest_path), "input_manifest_sha256": manifest_hash, "input_files": files,
        "input_identity_scope": "prepare manifest hash, file sizes/footer counts/schema hashes, original row IDs and hashes",
        "config_path": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config": config, "output": str(output), "report": str(report),
        "implementation_sha256": hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
            for name in ("curate.py", "rules.py", "data.py", "review.py"))).hexdigest(),
        "python": sys.version.split()[0], "packages": {name: version(name) for name in ("barq", "pyarrow", "pyyaml")},
        "scope": "labeled training candidates only; offline checks only", "training_ready": False,
        "group_fields": list(GROUP_FIELDS), "sampling_method": "reservoir per group; fixed input order and seed",
        "benchmarks": preparation.get("benchmarks", []), "benchmarks_rerun": False,
        "preparation_counts": preparation["counts"], "training_rows": 0,
    }
    manifest_file = report / "manifest.json"
    manifest_file.write_text(json_text(manifest) + "\n", encoding="utf-8")
    samples = Samples(config["sample_per_group"], config["seed"])
    counts, reasons, flags, checks, skipped_checks, skipped_rows = (Counter() for _ in range(6))
    sinks = []
    closed = False
    try:
        accepted_sink = ParquetSink(output / "candidates.parquet", CANDIDATE_SCHEMA, config["batch_size"])
        sinks.append(accepted_sink)
        decision_sink = ParquetSink(output / "decisions.parquet", CURATION_SCHEMA, config["batch_size"])
        sinks.append(decision_sink)
        print("curate: checking existing labeled training candidates (offline)", flush=True)
        with pq.ParquetFile(input_dir / "candidates.parquet") as parquet:
            for batch in parquet.iter_batches(batch_size=config["batch_size"], columns=CANDIDATE_SCHEMA.names):
                for row in batch.to_pylist():
                    if not row["is_labeled"] or row["split"] != "train":
                        skipped_rows["unlabeled" if not row["is_labeled"] else row["split"]] += 1
                        continue
                    decision, sample = assess(row, config)
                    counts[decision["decision"]] += 1
                    manifest["training_rows"] += 1
                    reasons.update(sample["reasons"])
                    flags.update(sample["flags"])
                    checks.update(code for code in sample["checks"] if "_skipped_" not in code)
                    skipped_checks.update(code for code in sample["checks"] if "_skipped_" in code)
                    decision_sink.append(decision)
                    if decision["decision"] == "accept":
                        accepted_sink.append(row)
                    samples.add(tuple(decision[key] for key in GROUP_FIELDS), sample)
                    if manifest["training_rows"] % 100000 == 0:
                        rate = manifest["training_rows"] / max(time.perf_counter() - started, 0.001)
                        print(f"  inspected {manifest['training_rows']:,} training rows ({rate:,.0f} rows/s)", flush=True)
        if manifest["training_rows"] + sum(skipped_rows.values()) != files["candidates"]["rows"]:
            raise ValueError("Inspected candidate count differs from prepare footer count.")
        for sink in sinks:
            sink.close()
        closed = True
        manifest.update(
            counts={name: counts[name] for name in DECISIONS}, reason_counts=dict(reasons), flag_counts=dict(flags),
            check_counts=dict(checks), skipped_check_counts=dict(skipped_checks), skipped_candidate_rows=dict(skipped_rows),
            groups=[{**dict(zip(GROUP_FIELDS, group)), "population": samples.seen[group], "sampled": len(samples.rows[group])}
                    for group in sorted(samples.rows)],
            review_sample_rows=samples.write(report / "review_samples.jsonl"),
        )
        write_report(report / "curation.md", manifest)
        for sink in sinks:
            sink.publish()
        manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                        output_files={sink.path.name: {"path": str(sink.path), "size_bytes": sink.path.stat().st_size}
                                      for sink in sinks})
    except BaseException as error:
        if not closed:
            for sink in sinks:
                try:
                    sink.close()
                except Exception:
                    pass
        manifest.update(status="failed", error=str(error), counts={name: counts[name] for name in DECISIONS})
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        manifest_file.write_text(json_text(manifest) + "\n", encoding="utf-8")
    print(f"Done: {manifest['counts']}\nReport: {report / 'curation.md'}\n"
          f"Samples: {report / 'review_samples.jsonl'}\nData: {output}", flush=True)
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="barq curate", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing data/processed/prepare/<run-id> directory")
    parser.add_argument("--config", type=Path, default=Path("configs/curation.yaml"))
    parser.add_argument("--manifest", type=Path, help="Prepare manifest if inputs were moved; retain run directory name")
    parser.add_argument("--output", type=Path, help="Workspace root for data/processed/curate and reports/curate")
    args = parser.parse_args(argv)
    try:
        run(args.input, config_path=args.config, manifest_path=args.manifest, output_root=args.output)
    except KeyboardInterrupt:
        print("Curation interrupted. Original preparation outputs are unchanged.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
