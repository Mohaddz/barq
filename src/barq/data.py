"""Phase 0: bounded previews and disk-backed preparation of pinned datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import fnmatch
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from barq.rules import benchmark_key, benchmark_texts, fingerprints, validate_example


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("datasets"):
        raise ValueError("Config must contain a nonempty datasets list.")
    for key in ("audit_rows", "sample_per_group", "batch_size"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer.")
    fraction = config.get("validation_fraction")
    if not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be between 0 (inclusive) and 1.")
    names = set()
    for source in config["datasets"]:
        name = source.get("name", "")
        if not re.fullmatch(r"[a-z0-9_-]+", name) or name in names:
            raise ValueError("Dataset names must be unique simple lowercase names.")
        names.add(name)
        if not re.fullmatch(r"[a-f0-9]{40}", source.get("revision", "")):
            raise ValueError(f"{name}: pin revision to a full 40-character commit SHA.")
        if source.get("adapter") not in {"mix", "aisa"}:
            raise ValueError(f"{name}: unknown adapter.")
        if source.get("config") != "default":
            raise ValueError("Phase 0 supports these datasets' default configuration.")
        if not isinstance(source.get("repo_id"), str) or "/" not in source["repo_id"]:
            raise ValueError(f"{name}: repo_id must be a Hugging Face dataset ID.")
        if type(source.get("preserve_splits")) is not bool:
            raise ValueError(f"{name}: preserve_splits must be true or false.")
        splits = source.get("splits")
        if not isinstance(splits, dict) or not splits:
            raise ValueError(f"{name}: splits must map names to expected row counts.")
        for split, count in splits.items():
            if split not in {"train", "dev", "validation", "test"}:
                raise ValueError(f"{name}: unsupported split {split!r}.")
            if type(count) is not int or count < 1:
                raise ValueError(f"{name}/{split}: expected count must be positive.")
        if not set(source.get("unlabeled_splits", [])).issubset(splits):
            raise ValueError(f"{name}: unlabeled_splits must occur in splits.")
        if source.get("unlabeled_splits") and not source["preserve_splits"]:
            raise ValueError("Input-only splits must preserve their official split.")
        if "train" in source.get("unlabeled_splits", []):
            raise ValueError("Input-only data cannot be a training split.")
    return config


def viewer_page(source, split, offset, length):
    """The viewer does not select revisions; verify its response before using it."""
    query = urlencode({"dataset": source["repo_id"], "config": source["config"],
                       "split": split, "offset": offset, "length": length})
    request = Request("https://datasets-server.huggingface.co/rows?" + query,
                      headers={"User-Agent": "barq-phase0/0.1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=40) as response:
                revision = response.headers.get("x-revision")
                if revision != source["revision"]:
                    raise ValueError(
                        f"Viewer revision for {source['name']}/{split} is {revision!r}; "
                        "it does not match the config pin. Audit cannot use that preview. "
                        "Review/update the pin and counts, or use prepare for the pinned data.")
                page = json.load(response)
            if page.get("num_rows_total") != source["splits"][split]:
                raise ValueError(f"{source['name']}/{split}: row count differs from config.")
            if len(page.get("rows", [])) != length:
                raise ValueError(f"{source['name']}/{split}: incomplete viewer page.")
            return page["rows"]
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 2:
                raise RuntimeError("Sample request failed; no full-download fallback was used.") from error
            time.sleep(2 ** attempt)


def audit_rows(source, split, limit, seed):
    """Read disjoint small pages spread through a split, without scanning Parquet."""
    total = source["splits"][split]
    budget = min(limit, total)
    pages = (budget + 24) // 25
    rng = random.Random(digest(f"{seed}:{source['repo_id']}:{split}"))
    for page_number in range(pages):
        lo, hi = page_number * total // pages, (page_number + 1) * total // pages
        count = (page_number + 1) * budget // pages - page_number * budget // pages
        offset = rng.randint(lo, hi - count)
        for item in viewer_page(source, split, offset, count):
            if item.get("truncated_cells"):
                raise ValueError("Viewer truncated a sampled row; use prepare to inspect originals.")
            yield item["row_idx"], item["row"]


def download_source(source, raw_root):
    """Cache unchanged original Parquet files; do not make a second Arrow copy."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info(source["repo_id"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise ValueError("Hugging Face did not resolve the requested revision.")
    files = {}
    for split in source["splits"]:
        files[split] = sorted(
            item.rfilename for item in info.siblings
            if fnmatch.fnmatch(item.rfilename, f"data/{split}-*.parquet")
            or item.rfilename == f"data/{split}.parquet"
        )
        if not files[split]:
            raise ValueError(f"No original Parquet files found for {source['name']}/{split}.")
    destination = raw_root / source["name"] / source["revision"]
    snapshot_download(repo_id=source["repo_id"], repo_type="dataset",
                      revision=source["revision"], local_dir=destination,
                      allow_patterns=[file for split_files in files.values() for file in split_files])
    return {split: [str(destination / file) for file in paths] for split, paths in files.items()}


def prepare_rows(files, split, raw_root):
    from datasets import load_dataset

    rows = load_dataset("parquet", data_files={split: files[split]}, split=split,
                        streaming=True, cache_dir=str(raw_root / ".metadata"))
    yield from enumerate(rows)


def clean_schema(schema):
    """Remove AISA's known Arrow union padding, retaining semantic null literals."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    optional_keywords = {"type", "description", "required", "enum", "properties", "items",
                         "additionalProperties", "format", "minimum", "maximum", "title",
                         "minItems", "maxItems", "minLength", "maxLength", "pattern",
                         "anyOf", "oneOf", "allOf", "$ref", "$defs", "definitions"}
    for key, value in schema.items():
        if value is None and key in optional_keywords:
            continue
        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            result[key] = {name: clean_schema(child) for name, child in value.items()
                           if child is not None}
        elif key in {"items", "additionalProperties"}:
            result[key] = clean_schema(value)
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(value, list):
            result[key] = [clean_schema(child) for child in value]
        else:
            result[key] = value
    return result


def permits_null(schema):
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return False
    if "enum" in schema:
        return not isinstance(schema["enum"], list) or None in schema["enum"]
    if "const" in schema:
        return schema["const"] is None
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            return any(permits_null(child) for child in schema[key])
    kind = schema.get("type")
    return kind is None or kind == "null" or isinstance(kind, list) and "null" in kind


def adapt_row(raw, source):
    messages = deepcopy(raw.get("messages"))
    if source["adapter"] == "mix":
        return {"messages": messages, "tools": [],
                "source": raw.get("dataset_name"), "task": raw.get("task_type"),
                "metadata": {}, "adapter_notes": []}

    tools = deepcopy(raw.get("tools_sampled"))
    notes = []
    if tools is None:
        # Never substitute the full tool registry for missing candidate tools.
        tools = []
        notes.append("missing_candidate_tools")
    schemas = {}
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            continue
        before = function.get("parameters")
        after = clean_schema(before)
        if "parameters" in function:
            function["parameters"] = after
        if before != after:
            notes.append("schema_padding_removed")
        if isinstance(after, dict) and isinstance(function.get("name"), str):
            schemas[function.get("name")] = after
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        for key in ("tool_calls", "think", "_think_for_train"):
            if message.get(key) is None:
                message.pop(key, None)
        for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls", []), list) else []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            function = call["function"]
            args = function.get("arguments")
            # Only Arrow dictionaries receive padding cleanup. JSON strings are originals.
            if not isinstance(args, dict) or not isinstance(function.get("name"), str) or function["name"] not in schemas:
                continue
            schema = schemas[function["name"]]
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                continue
            cleaned = {key: value for key, value in args.items()
                       if value is not None or key in required
                       or key in properties and permits_null(properties[key])}
            if cleaned != args:
                notes.append("argument_padding_removed")
                function["arguments"] = cleaned
    metadata = {key: raw.get(key) for key in
                ("dialect", "requires_function", "tool_called", "negative_category")}
    return {"messages": messages, "tools": tools, "source": source["repo_id"],
            "task": "tool_use", "metadata": metadata, "adapter_notes": sorted(set(notes))}


def load_benchmarks(config, config_dir, connection):
    connection.execute("CREATE TABLE benchmarks (hash TEXT, name TEXT, PRIMARY KEY(hash, name))")
    results = []
    for item in config.get("benchmarks", []):
        name = item["name"]
        if item.get("path") is None:
            results.append({"name": name, "status": "not_checked", "reason": "No reference file configured"})
            continue
        path = (config_dir / item["path"]).resolve()
        hasher = hashlib.sha256()
        count = 0
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                hasher.update(line)
                if not line.strip():
                    continue
                item_row = json.loads(line)
                if not isinstance(item_row, dict) or not isinstance(item_row.get("variants", []), list):
                    raise ValueError(f"{path}:{line_number}: expected prompt and optional variants list.")
                texts = [item_row.get("prompt"), *item_row.get("variants", [])]
                if any(not isinstance(text, str) or not text.strip() for text in texts):
                    raise ValueError(f"{path}:{line_number}: benchmark prompts must be nonempty strings.")
                for text in texts:
                    connection.execute("INSERT OR IGNORE INTO benchmarks VALUES (?, ?)",
                                       (benchmark_key(text), name))
                count += 1
        if count == 0:
            raise ValueError(f"Benchmark file is empty: {path}")
        results.append({"name": name, "status": "checked_exact", "path": str(path),
                        "sha256": hasher.hexdigest(), "reference_rows": count})
    connection.commit()
    return results


COMMON_FIELDS = [(key, pa.string()) for key in
                 ("id", "dataset", "revision", "source", "task", "original_split", "split")]
COMMON_FIELDS += [("row_index", pa.int64()), ("is_labeled", pa.bool_())]
CANDIDATE_SCHEMA = pa.schema(COMMON_FIELDS + [(key, pa.string()) for key in
    ("example_hash", "input_hash", "messages_json", "tools_json", "metadata_json")])
DECISION_SCHEMA = pa.schema(COMMON_FIELDS + [(key, pa.string()) for key in
    ("decision", "reasons_json", "adapter_notes_json", "duplicate_of", "benchmark_matches_json")])


class ParquetSink:
    def __init__(self, path, schema, batch_size):
        self.path = path
        self.partial = path.with_suffix(".parquet.partial")
        self.schema = schema
        self.batch_size = batch_size
        self.rows = []
        self.writer = pq.ParquetWriter(self.partial, schema, compression="zstd")

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self):
        self.flush()
        self.writer.close()

    def publish(self):
        self.partial.replace(self.path)


def jobs_for(sources):
    jobs = []
    for source in sources:
        for split in source["splits"]:
            output_split = "validation" if split == "dev" else split
            priority = {"test": 0, "validation": 1}.get(output_split, 2) if source["preserve_splits"] else 3
            jobs.append((priority, source["name"], split, source))
    return sorted(jobs, key=lambda item: item[:3])


def assign_split(connection, group, source, original_split, config):
    existing = connection.execute("SELECT split, protected FROM groups WHERE hash=?", (group,)).fetchone()
    official = source["preserve_splits"]
    split = "validation" if original_split == "dev" else original_split
    if official and split != "train":
        if not existing:
            connection.execute("INSERT INTO groups VALUES (?, ?, 1)", (group, split))
        return split, False
    if existing:
        return existing[0], bool(existing[1])
    if not official:
        fraction = int(digest(f"{config['seed']}:{group}")[:16], 16) / 2**64
        split = "validation" if fraction < config["validation_fraction"] else "train"
    connection.execute("INSERT INTO groups VALUES (?, ?, 0)", (group, split))
    return split, False


def write_report(path, manifest, counts, reasons, groups, dialects, observed, samples):
    lines = ["# Phase 0 data audit", "", f"Run: `{manifest['run_id']}` — **{manifest['mode']}**", "",
             f"Rows inspected: **{sum(counts.values()):,}**", "",
             "These are candidates after structural checks, not fact-checked or training-ready data.", ""]
    if manifest["mode"] == "audit":
        lines += ["This preview uses small, seeded pages spread across each selected split. It is not a",
                  "representative quality estimate. Counts and duplicate findings cover the sample only.",
                  "No full dataset files were downloaded. Sampling may miss rare sources/tasks.", ""]
    lines += ["## Decisions", "", "| Decision | Rows |", "|---|---:|"]
    lines += [f"| {key} | {value:,} |" for key, value in sorted(counts.items())]
    lines += ["", "## Source/task coverage", "", "| Source | Task | Inspected | Kept |", "|---|---|---:|---:|"]
    for (source, task), values in sorted(groups.items()):
        lines.append(f"| {source} | {task} | {values['seen']:,} | {values['keep']:,} |")
    for source in manifest["config"]["datasets"]:
        if source["name"] not in manifest["selected_datasets"]:
            continue
        missing = sorted(set(source.get("expected_sources", [])) - observed[source["name"]])
        if missing:
            lines += ["", f"Unobserved sources in {source['name']}: " + ", ".join(missing)]
    lines += ["", "## Reasons", ""]
    lines += [f"- `{key}`: {value:,}" for key, value in sorted(reasons.items())] or ["- No rejected rows."]
    if dialects:
        lines += ["", "## AISA dialect labels (inspected rows)", ""]
        lines += [f"- {key}: {value:,}" for key, value in sorted(dialects.items())]
    lines += ["", "## Benchmark checks", ""]
    for item in manifest["benchmarks"]:
        lines.append(f"- **{item['name']}**: {item['status']}" +
                     (f" ({item['reference_rows']:,} reference rows)" if "reference_rows" in item else " — no references supplied"))
    if not manifest["benchmarks"]:
        lines.append("- NOT CHECKED: no benchmark references configured.")
    lines += ["", "Only NFC/whitespace-normalized exact user-text matches are checked. Translations,",
              "paraphrases, answer-only overlap and near duplicates remain unchecked. Source licenses",
              "and factual/religious correctness require review before training.", "",
              "AISA dev remains validation. Its blind test inputs are in holdout.parquet; they are",
              "not no-call negatives and cannot be scored locally without gold labels. Official",
              "evaluation rows are retained even if repeated; matching training inputs are excluded.",
              "This protection covers only selected datasets and rows inspected in this run.", "",
              "No source rebalancing or teacher rewriting has been applied.", "",
              "## Review samples", "", "Full sampled records are in samples.jsonl beside this report.", ""]
    for group_samples in samples.values():
        for sample in group_samples:
            lines += [f"- `{sample['id'][:12]}` | {sample['source']} / {sample['task']} | {sample['decision']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config, config_path, mode, root, limit=None, selected=None):
    sources = [source for source in config["datasets"] if selected is None or source["name"] == selected]
    if not sources:
        raise ValueError(f"Unknown dataset {selected!r}.")
    if mode == "prepare" and limit is not None:
        raise ValueError("--limit is only for audit; prepare always processes all selected rows.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / "data" / "processed" / mode / run_id
    report_dir = root / "reports" / mode / run_id
    output.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    manifest = {"schema_version": 1, "run_id": run_id, "mode": mode, "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "selected_datasets": [source["name"] for source in sources],
                "sample_limit_per_split": (limit or config["audit_rows"]) if mode == "audit" else None,
                "config_sha256": digest(json_text(config)), "config_path": str(config_path),
                "python": sys.version.split()[0],
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes() + Path(__file__).with_name("rules.py").read_bytes()
                ).hexdigest(),
                "packages": {name: version(name) for name in ("barq", "datasets", "huggingface-hub", "pyarrow", "pyyaml")},
                "benchmarks": [], "source_files": {}, "output": str(output)}
    manifest_path = report_dir / "manifest.json"
    manifest_path.write_text(json_text(manifest) + "\n", encoding="utf-8")
    sinks = []
    counts, reasons, dialects = Counter(), Counter(), Counter()
    groups = defaultdict(Counter)
    observed = defaultdict(set)
    samples, sample_seen = defaultdict(list), Counter()
    sample_rng = random.Random(config["seed"])
    downloads, split_counts = {}, {}
    try:
        with closing(sqlite3.connect(output / "index.sqlite3")) as connection:
            connection.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA cache_size=-32768;
                CREATE TABLE examples (hash TEXT PRIMARY KEY, id TEXT NOT NULL);
                CREATE TABLE groups (hash TEXT PRIMARY KEY, split TEXT NOT NULL, protected INTEGER NOT NULL);
            """)
            manifest["benchmarks"] = load_benchmarks(config, config_path.parent, connection)
            for name, schema in (("candidates", CANDIDATE_SCHEMA), ("holdout", CANDIDATE_SCHEMA), ("decisions", DECISION_SCHEMA)):
                sinks.append(ParquetSink(output / f"{name}.parquet", schema, config["batch_size"]))
            candidate_sink, holdout_sink, decision_sink = sinks
            for _, _, original_split, source in jobs_for(sources):
                print(f"{mode}: {source['name']}/{original_split}", flush=True)
                if mode == "audit":
                    rows = audit_rows(source, original_split, limit or config["audit_rows"], config["seed"])
                else:
                    if source["name"] not in downloads:
                        downloads[source["name"]] = download_source(source, root / "data" / "raw")
                        manifest["source_files"][source["name"]] = downloads[source["name"]]
                    rows = prepare_rows(downloads[source["name"]], original_split, root / "data" / "raw")
                total = 0
                for row_index, raw in rows:
                    total += 1
                    item = adapt_row(raw, source)
                    labeled = original_split not in source.get("unlabeled_splits", [])
                    row_id = digest(f"{source['repo_id']}@{source['revision']}:{source['config']}:{original_split}:{row_index}")
                    source_name = item["source"] if isinstance(item["source"], str) else "unknown"
                    task = item["task"] if isinstance(item["task"], str) else "unknown"
                    common = {"id": row_id, "dataset": source["name"], "revision": source["revision"],
                              "source": source_name, "task": task, "original_split": original_split,
                              "split": "validation" if original_split == "dev" else original_split,
                              "row_index": row_index, "is_labeled": labeled}
                    # A bad gold target must not erase an official evaluation boundary.
                    if source["preserve_splits"] and original_split != "train":
                        if isinstance(item["messages"], list) and isinstance(item["tools"], list):
                            try:
                                _, protected_group = fingerprints(item["messages"], item["tools"])
                                assign_split(connection, protected_group, source, original_split, config)
                            except (ValueError, TypeError):
                                pass
                    problems = validate_example(item["messages"], item["tools"], allow_missing_target=not labeled)
                    if source_name == "unknown" or task == "unknown":
                        problems.append("missing_source_or_task")
                    if "missing_candidate_tools" in item["adapter_notes"]:
                        problems.append("missing_candidate_tools")
                    decision, duplicate_of, matches = "quarantine", None, []
                    full_hash = input_hash = None
                    if not problems:
                        full_hash, input_hash = fingerprints(item["messages"], item["tools"])
                        common["split"], protected = assign_split(connection, input_hash, source, original_split, config)
                        matches = sorted({name for text in benchmark_texts(item["messages"])
                                          for (name,) in connection.execute(
                                              "SELECT name FROM benchmarks WHERE hash=?", (benchmark_key(text),))})
                        official_eval = source["preserve_splits"] and original_split != "train"
                        previous = connection.execute("SELECT id FROM examples WHERE hash=?", (full_hash,)).fetchone()
                        if protected:
                            decision, problems = "exclude", ["official_holdout_overlap"]
                        elif matches and not official_eval:
                            decision, problems = "exclude", ["benchmark_overlap"]
                        elif previous and not official_eval:
                            decision, problems, duplicate_of = "exclude", ["exact_duplicate"], previous[0]
                        else:
                            decision = "keep" if labeled else "holdout"
                            connection.execute("INSERT OR IGNORE INTO examples VALUES (?, ?)", (full_hash, row_id))
                            candidate = {**common, "example_hash": full_hash, "input_hash": input_hash,
                                         "messages_json": json_text(item["messages"]), "tools_json": json_text(item["tools"]),
                                         "metadata_json": json_text(item["metadata"])}
                            (candidate_sink if labeled else holdout_sink).append(candidate)
                    decision_sink.append({**common, "decision": decision, "reasons_json": json_text(problems),
                                          "adapter_notes_json": json_text(item["adapter_notes"]),
                                          "duplicate_of": duplicate_of, "benchmark_matches_json": json_text(matches)})
                    counts[decision] += 1
                    reasons.update(problems)
                    group = (source_name, task)
                    groups[group]["seen"] += 1
                    groups[group][decision] += 1
                    observed[source["name"]].add(source_name)
                    if source["adapter"] == "aisa":
                        dialects[str(item["metadata"].get("dialect") or "unknown")] += 1
                    # Bounded reservoir per source/task/decision for manual inspection.
                    sample_key = (source_name, task, decision)
                    sample_seen[sample_key] += 1
                    sample = {**common, "decision": decision, "reasons": problems,
                              "messages": item["messages"], "tools": item["tools"], "metadata": item["metadata"]}
                    if len(samples[sample_key]) < config["sample_per_group"]:
                        samples[sample_key].append(sample)
                    else:
                        position = sample_rng.randrange(sample_seen[sample_key])
                        if position < config["sample_per_group"]:
                            samples[sample_key][position] = sample
                    if total % config["batch_size"] == 0:
                        connection.commit()
                    if total % 10000 == 0:
                        print(f"  inspected {total:,} rows", flush=True)
                expected = source["splits"][original_split] if mode == "prepare" else min(limit or config["audit_rows"], source["splits"][original_split])
                if total != expected:
                    raise ValueError(f"{source['name']}/{original_split}: expected {expected:,} rows, read {total:,}.")
                split_counts[f"{source['name']}/{original_split}"] = total
                connection.commit()
            for sink in sinks:
                sink.close()
            sinks_closed = True
            write_report(report_dir / "audit.md", manifest, counts, reasons, groups, dialects, observed, samples)
            with (report_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
                for group_samples in samples.values():
                    for sample in group_samples:
                        stream.write(json_text(sample) + "\n")
            for sink in sinks:
                sink.publish()
            manifest.update(status="complete", counts=dict(counts), inspected_by_split=split_counts,
                            finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        if not locals().get("sinks_closed"):
            for sink in sinks:
                try:
                    sink.close()
                except Exception:
                    pass
        manifest.update(status="failed", error=str(error), counts=dict(counts), inspected_by_split=split_counts)
        raise
    finally:
        manifest_path.write_text(json_text(manifest) + "\n", encoding="utf-8")
    print(f"Done: {dict(counts)}\nReport: {report_dir / 'audit.md'}\nData: {output}", flush=True)
    return output, report_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("audit", "prepare"), default="audit")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--limit", type=int, help="Audit rows per dataset split; never used for prepare")
    parser.add_argument("--dataset", help="Select one configured dataset, e.g. mix or aisa")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="Root for data/ and reports/")
    args = parser.parse_args()
    try:
        config_path = args.config.resolve()
        run(read_config(config_path), config_path, args.mode, args.output.resolve(), args.limit, args.dataset)
    except KeyboardInterrupt:
        print("Interrupted. Completed downloads remain cached; rerun starts a fresh processing run.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
