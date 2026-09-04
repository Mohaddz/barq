"""Offline quality signals and bounded review samples from a completed prepare run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from barq.data import CANDIDATE_SCHEMA, DECISION_SCHEMA, ParquetSink, digest, json_text
from barq.rules import review_checks


GROUP_FIELDS = ("dataset", "source", "task", "task_hint", "dialect", "tool_behavior")
FLAG_SCHEMA = pa.schema([(name, pa.string()) for name in
                         ("id", "dataset", "source", "task", "split", "flags_json", "checks_json")])


def open_inputs(input_dir, manifest_path):
    """Check completion, run identity and all footer counts before scanning any rows."""
    input_dir = input_dir.resolve(strict=True)
    if not input_dir.is_dir():
        raise ValueError("--input must be the directory of a completed prepare run.")
    if manifest_path is None:
        if len(input_dir.parents) < 4 or input_dir.parent.name != "prepare":
            raise ValueError("Cannot locate the prepare manifest; supply --manifest.")
        root = input_dir.parents[3]
        manifest_path = root / "reports" / "prepare" / input_dir.name / "manifest.json"
    manifest_path = manifest_path.resolve(strict=True)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if (manifest.get("status"), manifest.get("mode"), manifest.get("schema_version")) != ("complete", "prepare", 1):
        raise ValueError("Review requires a complete Phase 0 prepare manifest with schema_version 1.")
    if manifest.get("run_id") != input_dir.name:
        raise ValueError("Prepare run_id does not match the input directory name.")
    counts = manifest.get("counts", {})
    if not counts or any(key not in {"keep", "holdout", "exclude", "quarantine"}
                         or type(value) is not int or value < 0 for key, value in counts.items()):
        raise ValueError("Prepare manifest contains invalid decision counts.")
    expected = {"candidates": counts.get("keep", 0), "holdout": counts.get("holdout", 0),
                "decisions": sum(counts.values())}
    files = {}
    for name in expected:
        path = input_dir / f"{name}.parquet"
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != expected[name]:
                raise ValueError(f"{path.name}: footer row count does not match the prepare manifest.")
            required = DECISION_SCHEMA if name == "decisions" else CANDIDATE_SCHEMA
            if not set(required.names).issubset(parquet.schema_arrow.names):
                raise ValueError(f"{path.name}: expected Phase 0 columns are missing.")
            files[name] = {"path": str(path), "rows": parquet.metadata.num_rows,
                           "size_bytes": path.stat().st_size,
                           "schema_sha256": digest(str(parquet.schema_arrow))}
    return input_dir, manifest_path, manifest, hashlib.sha256(raw_manifest).hexdigest(), files


def parquet_rows(path, columns):
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=512, columns=columns):
            yield from batch.to_pylist()


class Samples:
    """One bounded reservoir and independent seeded RNG per sampling group."""

    def __init__(self, size, seed):
        self.size, self.seed = size, seed
        self.seen = Counter()
        self.rows = defaultdict(list)
        self.random = {}

    def add(self, group, row):
        self.seen[group] += 1
        if len(self.rows[group]) < self.size:
            self.rows[group].append(row)
            return
        if group not in self.random:
            self.random[group] = random.Random(digest(f"{self.seed}:{json_text(group)}"))
        position = self.random[group].randrange(self.seen[group])
        if position < self.size:
            self.rows[group][position] = row

    def write(self, path):
        count = 0
        with path.open("w", encoding="utf-8") as stream:
            for group in sorted(self.rows):
                for row in sorted(self.rows[group], key=lambda item: item["id"]):
                    stream.write(json_text(row) + "\n")
                    count += 1
        return count


def decision_summary(path):
    decisions, reasons = Counter(), Counter()
    groups = defaultdict(lambda: {"decisions": Counter(), "reasons": Counter()})
    for row in parquet_rows(path, ["dataset", "source", "task", "decision", "reasons_json"]):
        decision = row["decision"]
        if decision not in {"keep", "holdout", "exclude", "quarantine"}:
            raise ValueError(f"Unknown preparation decision: {decision!r}")
        codes = json.loads(row["reasons_json"])
        if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
            raise ValueError("Invalid reasons_json in decisions.parquet.")
        key = (row["dataset"], row["source"], row["task"])
        decisions[decision] += 1
        reasons.update(set(codes))
        groups[key]["decisions"][decision] += 1
        groups[key]["reasons"].update(set(codes))
    grouped = [{"dataset": key[0], "source": key[1], "task": key[2],
                "decisions": dict(value["decisions"]), "reasons": dict(value["reasons"])}
               for key, value in sorted(groups.items())]
    return decisions, reasons, grouped


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(path, manifest):
    lines = ["# Phase 0.1 quality review", "", f"Prepare run: `{manifest['input_run_id']}`", "",
             f"Training candidates inspected: **{manifest['training_rows']:,}**. "
             f"Flagged for review: **{manifest['flagged_rows']:,}**.", "",
             "Original files and splits were not changed. No dataset downloads, model calls, "
             "automatic repairs, or training occurred.", "",
             "## Review files", "",
             f"- `review_samples.jsonl`: {manifest['review_sample_rows']:,} sampled records, up to "
             f"{manifest['per_group']} per source/task/hint/dialect/call-behavior group.",
             f"- `flagged_samples.jsonl`: {manifest['diagnostic_sample_rows']:,} diagnostic records, up to "
             f"{manifest['flagged_per_group']} per source/task/flag. A record can occur under multiple flags.",
             "- `flags.parquet`: IDs and signals for flagged training rows only. An absent ID is not a quality approval.",
             "- `manifest.json`: inputs, counts, groups, check coverage and sampling settings.", "",
             "The main sample includes flagged and unflagged records. Diagnostic samples are deliberately "
             "biased toward flagged cases; do not use them to estimate error rates. Overall quality estimates "
             "from equal-sized group samples must be weighted by group population sizes.", "",
             "Fill each sampled record's `review` fields with a decision (keep/review/repair/exclude), "
             "correctness, Arabic quality and instruction following (pass/fail/uncertain), and notes. "
             "These annotations are for calibration; nothing applies them automatically.", "",
             "## Sample coverage", "", "| Source | Task | Hint | Dialect | Calls | Training rows | Sampled |",
             "|---|---|---|---|---|---:|---:|"]
    for group in manifest["groups"]:
        values = [group[name] or "—" for name in ("source", "task", "task_hint", "dialect", "tool_behavior")]
        lines.append("| " + " | ".join(map(markdown_cell, values)) +
                     f" | {group['population']:,} | {group['sampled']:,} |")
    lines += ["", "## Quality signals", "", "Counts describe checks, not verified semantic error rates. "
              "One row can have several flags.", ""]
    lines += [f"- `{name}`: {count:,}" for name, count in sorted(manifest["flag_counts"].items())] or ["- No flags."]
    lines += ["", "### Check coverage", ""]
    lines += [f"- `{name}` applied: {count:,}" for name, count in sorted(manifest["check_counts"].items())]
    lines += [f"- `{name}`: {count:,}" for name, count in sorted(manifest["skipped_check_counts"].items())]
    lines += ["", "Python is parsed, never executed: syntax does not prove correct behavior. A failed parse "
              "of an unfenced answer is flagged as `python_answer_unparsed` and excluded from applied syntax "
              "counts, because prose and code may be mixed. Sentiment "
              "checks validate label vocabulary only. Diacritization checks preserve underlying text and "
              "detect outputs without vowel marks; they do not verify every vowel. Creative-writing hints "
              "are simple routing heuristics, not authoritative task labels.", "",
              "Factual correctness, arithmetic reasoning, translation alignment, religious evidence, "
              "dialect authenticity and writing quality still require semantic review.", "",
              "## Preparation decisions by source/task", "", "Reasons overlap within rows.", "",
              "| Source | Task | Kept | Excluded | Quarantined | Holdout | Reasons |",
              "|---|---|---:|---:|---:|---:|---|"]
    for group in manifest["preparation_groups"]:
        counts = group["decisions"]
        reasons = "; ".join(f"{key}: {count:,}" for key, count in sorted(group["reasons"].items())) or "—"
        lines.append(f"| {markdown_cell(group['source'])} | {markdown_cell(group['task'])} | " +
                     " | ".join(f"{counts.get(key, 0):,}" for key in ("keep", "exclude", "quarantine", "holdout")) +
                     f" | {markdown_cell(reasons)} |")
    lines += ["", "## Benchmark coverage", "", "Copied from the prepare manifest; no benchmark checks were added or rerun.", ""]
    for item in manifest["benchmarks"]:
        lines.append(f"- {item['name']}: **{item['status']}**")
    if not manifest["benchmarks"]:
        lines.append("- NOT CHECKED: no benchmark references were configured.")
    lines += ["", "Before final training exports: inspect these samples, calibrate any model judge, "
              "supply benchmark references, and choose source weights and targeted repairs.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(input_dir, *, manifest_path=None, per_group=100, flagged_per_group=5, seed=42, output_root=None):
    started = time.perf_counter()
    if any(type(value) is not int or value < 1 for value in (per_group, flagged_per_group)):
        raise ValueError("Sampling counts must be positive integers.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    input_dir, manifest_path, preparation, manifest_hash, files = open_inputs(input_dir, manifest_path)
    if output_root is None:
        # The manifest can move with the reports/prepare/<run-id>/ tree.
        standard_layout = (len(manifest_path.parents) >= 4
                           and manifest_path.parent.name == preparation["run_id"]
                           and manifest_path.parents[1].name == "prepare"
                           and manifest_path.parents[2].name == "reports")
        output_root = manifest_path.parents[3] if standard_layout else Path.cwd()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = output_root.resolve() / "reports" / "review" / run_id
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1, "mode": "review", "status": "running", "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(), "input_run_id": preparation["run_id"],
        "input_manifest": str(manifest_path), "input_manifest_sha256": manifest_hash, "input_files": files,
        "per_group": per_group, "flagged_per_group": flagged_per_group, "seed": seed,
        "sampling_scope": "labeled training candidates only", "group_fields": list(GROUP_FIELDS),
        "sampling_method": "reservoir per group; fixed input order and seed",
        "benchmarks": preparation.get("benchmarks", []),
        "python": sys.version.split()[0], "pyarrow": version("pyarrow"),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes() + Path(__file__).with_name("rules.py").read_bytes()
            + Path(__file__).with_name("data.py").read_bytes()).hexdigest(),
        "training_rows": 0, "flagged_rows": 0,
    }
    manifest_file = output / "manifest.json"
    manifest_file.write_text(json_text(manifest) + "\n", encoding="utf-8")
    samples, diagnostics = Samples(per_group, seed), Samples(flagged_per_group, seed)
    flags, checks, skipped_checks, skipped_rows = Counter(), Counter(), Counter(), Counter()
    sink = None
    closed = False
    try:
        print("review: summarizing existing preparation decisions", flush=True)
        decisions, reasons, preparation_groups = decision_summary(input_dir / "decisions.parquet")
        if decisions != Counter(preparation["counts"]):
            raise ValueError("Decision contents do not match the prepare manifest counts.")
        manifest.update(decision_counts=dict(decisions), reason_counts=dict(reasons),
                        preparation_groups=preparation_groups)
        sink = ParquetSink(output / "flags.parquet", FLAG_SCHEMA, 1000)
        print("review: checking and sampling labeled training candidates", flush=True)
        for row in parquet_rows(input_dir / "candidates.parquet", CANDIDATE_SCHEMA.names):
            if not row["is_labeled"] or row["split"] != "train":
                skipped_rows["unlabeled" if not row["is_labeled"] else row["split"]] += 1
                continue
            messages, tools, metadata = (json.loads(row[field]) for field in
                                         ("messages_json", "tools_json", "metadata_json"))
            if not isinstance(messages, list) or not isinstance(tools, list) or not isinstance(metadata, dict):
                raise ValueError(f"{row['id']}: invalid prepared JSON types.")
            result = review_checks(messages, tools, row["source"], row["task"])
            codes = list(dict.fromkeys(result["flags"]))
            applied = list(dict.fromkeys(result["checks"]))
            task_hint = result["task_hint"] or ""
            dialect = str(metadata.get("dialect") or "unspecified") if row["task"] == "tool_use" else ""
            behavior = ""
            if row["task"] == "tool_use":
                behavior = "call" if any(isinstance(message, dict) and message.get("role") == "assistant"
                                          and message.get("tool_calls") for message in messages) else "no_call"
            group = (row["dataset"], row["source"], row["task"], task_hint, dialect, behavior)
            manifest["training_rows"] += 1
            flags.update(codes)
            checks.update(name for name in applied if "_skipped_" not in name)
            skipped_checks.update(name for name in applied if "_skipped_" in name)
            sample = {key: row[key] for key in CANDIDATE_SCHEMA.names if not key.endswith("_json")}
            sample.update(messages=messages, tools=tools, metadata=metadata, task_hint=task_hint,
                          dialect=dialect, tool_behavior=behavior, flags=codes, checks=applied,
                          review={"decision": None, "correctness": None, "arabic_quality": None,
                                  "instruction_following": None, "notes": ""})
            samples.add(group, sample)
            if codes:
                manifest["flagged_rows"] += 1
                sink.append({**{key: row[key] for key in ("id", "dataset", "source", "task", "split")},
                             "flags_json": json_text(codes), "checks_json": json_text(applied)})
                for code in codes:
                    diagnostics.add((row["dataset"], row["source"], row["task"], code),
                                    {**sample, "diagnostic_flag": code})
            if manifest["training_rows"] % 100000 == 0:
                rate = manifest["training_rows"] / max(time.perf_counter() - started, 0.001)
                print(f"  inspected {manifest['training_rows']:,} training rows ({rate:,.0f} rows/s)", flush=True)
        sink.close()
        closed = True
        manifest.update(
            flag_counts=dict(flags), check_counts=dict(checks), skipped_check_counts=dict(skipped_checks),
            skipped_candidate_rows=dict(skipped_rows),
            groups=[{**dict(zip(GROUP_FIELDS, group)), "population": samples.seen[group],
                     "sampled": len(samples.rows[group])} for group in sorted(samples.rows)],
            review_sample_rows=samples.write(output / "review_samples.jsonl"),
            diagnostic_sample_rows=diagnostics.write(output / "flagged_samples.jsonl"),
        )
        write_report(output / "review.md", manifest)
        sink.publish()
        manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        if sink is not None and not closed:
            try:
                sink.close()
            except Exception:
                pass
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        manifest_file.write_text(json_text(manifest) + "\n", encoding="utf-8")
    print(f"Done: {manifest['training_rows']:,} training rows; {manifest['flagged_rows']:,} flagged.\n"
          f"Report: {output / 'review.md'}\nSamples: {output / 'review_samples.jsonl'}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="barq review", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing data/processed/prepare/<run-id> directory")
    parser.add_argument("--manifest", type=Path, help="Prepare manifest when inputs were moved; retain the run directory name")
    parser.add_argument("--per-group", type=int, default=100)
    parser.add_argument("--flagged-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Workspace root for reports/review/<run-id>")
    args = parser.parse_args(argv)
    try:
        run(args.input, manifest_path=args.manifest, per_group=args.per_group,
            flagged_per_group=args.flagged_per_group, seed=args.seed, output_root=args.output)
    except KeyboardInterrupt:
        print("Review interrupted. Preparation outputs are unchanged.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
