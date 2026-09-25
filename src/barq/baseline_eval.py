"""Small, zero-shot MCQ baseline for the pinned Nemotron model."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
RENDERER = "nemotron3_ultra"
PROMPT_VERSION = "barq-mcq-ar-v1"
SYSTEM_PROMPT = "أجب عن سؤال الاختيار من متعدد. أخرج حرف الخيار الصحيح فقط، مثل A أو B أو C أو D أو E."
PRICES_PER_MILLION = {"prefill": 0.195, "sample": 0.495}
DEFAULT_LIMIT_PER_BENCHMARK = 100

BENCHMARKS = (
    {"key": "arabicmmlu", "repo": "MBZUAI/ArabicMMLU", "subset": "All", "split": "test", "adapter": "arabicmmlu"},
    {"key": "alghafa_exams", "repo": "OALL/AlGhafa-Arabic-LLM-Benchmark-Native", "subset": "mcq_exams_test_ar", "split": "test", "adapter": "alghafa"},
    {"key": "arabic_exams", "repo": "OALL/Arabic_EXAMS", "subset": "default", "split": "test", "adapter": "arabic_exams"},
)


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _options(items):
    choices = []
    for letter, value in items:
        text = _text(value)
        if text:
            choices.append((letter, text))
    if len(choices) < 2 or len({letter for letter, _ in choices}) != len(choices):
        return []
    return choices


def parse_row(adapter, row, *, subset="", row_index=0):
    """Adapt one public benchmark test row; malformed rows return None."""
    if adapter == "arabicmmlu":
        question = _text(row.get("Question"))
        context = _text(row.get("Context"))
        raw_key = _text(row.get("Answer Key")).upper()
        if raw_key not in "ABCDE" or len(raw_key) != 1:
            return None
        choices = _options((chr(65 + i), row.get(f"Option {i + 1}")) for i in range(5))
        correct = raw_key
        identifier = row.get("ID", row.get("id", row_index))
        if row.get("is_few_shot") not in (None, 0, False, "0"):
            return None
    elif adapter == "alghafa":
        question = _text(row.get("query"))
        context = ""
        choices = _options((chr(65 + i), row.get(f"sol{i + 1}")) for i in range(5))
        raw_label = row.get("label")
        try:
            index = int(raw_label)
        except (TypeError, ValueError):
            return None
        correct = chr(65 + index) if 0 <= index < 5 else ""
        identifier = row.get("id", row_index)
    elif adapter == "arabic_exams":
        question = _text(row.get("question"))
        context = ""
        choices = _options((letter, row.get(letter)) for letter in "ABCD")
        correct = _text(row.get("answer")).upper()
        identifier = row.get("id", row_index)
    else:
        raise ValueError(f"Unknown MCQ adapter: {adapter}")

    valid_letters = {letter for letter, _ in choices}
    if not question or len(choices) < 2 or correct not in valid_letters:
        return None
    if context:
        question = f"السياق:\n{context}\n\nالسؤال:\n{question}"
    formatted = "\n".join(f"{letter}. {text}" for letter, text in choices)
    user_prompt = f"{question}\n\nالخيارات:\n{formatted}"
    stable_id = f"{subset}:{identifier}"
    return {"id": stable_id, "subset": subset, "question": user_prompt,
            "choices": [letter for letter, _ in choices], "gold": correct}


def messages_for(item):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]}]


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_LINE = re.compile(
    r"^(?:(?:answer|option|الإجابة|الجواب|الخيار)\s*[:：]?\s*)?\(?([A-E])\)?[.!،]?$",
    re.IGNORECASE,
)
_EXPLICIT = re.compile(r"(?:answer|option|الإجابة|الجواب|الخيار)\s*[:：]?\s*\(?([A-E])\)?\b", re.IGNORECASE)


def extract_answer(text):
    """Accept a standalone final choice or one explicit answer label; otherwise fail closed."""
    text = _THINK.sub("", text or "").strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = _ANSWER_LINE.fullmatch(lines[-1])
    if match:
        return match.group(1).upper()
    explicit = _EXPLICIT.findall(text)
    return explicit[-1].upper() if len(explicit) == 1 else None


def select_deterministic(rows, limit, seed=42):
    """Select a reproducible, spread-out sample without changing benchmark labels."""
    if limit is None or len(rows) <= limit:
        return rows
    ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest())
    return ranked[:limit]


def summarize(results):
    grouped = defaultdict(list)
    subjects = defaultdict(lambda: defaultdict(list))
    for row in results:
        grouped[row["benchmark"]].append(row)
        subjects[row["benchmark"]][row.get("subset", "default")].append(row)
    summary = {}
    for name, rows in sorted(grouped.items()):
        correct = sum(row["correct"] is True for row in rows)
        summary[name] = {"n": len(rows), "correct": correct,
                         "accuracy": correct / len(rows) if rows else None,
                         "unparsed": sum(row["prediction"] is None for row in rows),
                         "subsets": {subset: {"n": len(part),
                                              "correct": sum(row["correct"] is True for row in part),
                                              "accuracy": sum(row["correct"] is True for row in part) / len(part),
                                              "unparsed": sum(row["prediction"] is None for row in part)}
                                     for subset, part in sorted(subjects[name].items())}}
    return summary


def build_report(*, status, provenance, results, limit_per_benchmark, sampling=None):
    summary = summarize(results)
    for name, metrics in summary.items():
        source = next((item for item in provenance
                       if item.get("benchmark") == name and item.get("status") == "loaded"), None)
        metrics["denominator"] = source.get("eligible_rows") if source else None
        metrics["sampled"] = bool(source and source.get("eligible_rows") is not None
                                   and metrics["n"] < source["eligible_rows"])
    return {
        "schema_version": 1,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "renderer": RENDERER,
        "prompt": {"version": PROMPT_VERSION, "system": SYSTEM_PROMPT},
        "sampling": sampling or {"temperature": 0.0, "max_tokens": 128, "stop_sequences": "renderer defaults"},
        "split_policy": "official test split only; no train/validation/few-shot examples; every source revision pinned",
        "limit_per_benchmark": limit_per_benchmark,
        "provenance": provenance,
        "summary": summary,
        "rows": results,
    }


def validate_report(report):
    required = {"schema_version", "status", "created_at", "model", "renderer", "prompt",
                "sampling", "split_policy", "provenance", "summary", "rows"}
    missing = required - report.keys()
    if missing:
        raise ValueError(f"Report is missing fields: {sorted(missing)}")
    if report["schema_version"] != 1 or report["model"] != MODEL_ID or report["renderer"] != RENDERER:
        raise ValueError("Unexpected baseline report identity")
    for row in report["rows"]:
        if not {"benchmark", "id", "gold", "prediction", "correct"} <= row.keys():
            raise ValueError("Malformed baseline result row")
        if row["correct"] is not None and row["correct"] != (row["gold"] == row["prediction"]):
            raise ValueError("Baseline row score does not match its answer")
    return True


def dry_run_report():
    fixtures = [
        ("arabicmmlu", {"ID": 17, "Question": "ما عاصمة المملكة العربية السعودية؟", "Context": None,
                        "Option 1": "جدة", "Option 2": "الرياض", "Option 3": "مكة", "Option 4": "الدمام",
                        "Option 5": None, "Answer Key": "B", "is_few_shot": 0}, "B"),
        ("alghafa", {"query": "كم عدد أيام الأسبوع؟", "sol1": "خمسة", "sol2": "ستة",
                      "sol3": "سبعة", "sol4": "ثمانية", "label": "2"}, "الإجابة: C"),
        ("arabic_exams", {"id": "science-1", "question": "ما نواتج هضم الدهون؟", "A": "أحماض دهنية وجلسرول",
                           "B": "أحماض أمينية", "C": "سكريات", "D": "غازات", "answer": "A"},
         "<think>قد تكون B، لكن الاختيار النهائي هو A.</think>\nA"),
    ]
    results = []
    for adapter, row, completion in fixtures:
        item = parse_row(adapter, row, subset="fixture", row_index=0)
        if item is None:
            raise ValueError(f"Dry-run fixture failed to parse: {adapter}")
        messages = messages_for(item)
        if len(messages) != 2 or messages[0]["role"] != "system" or "الخيارات:" not in messages[1]["content"]:
            raise ValueError("Dry-run prompt formatting failed")
        prediction = extract_answer(completion)
        results.append({"benchmark": adapter, "subset": "fixture", "id": item["id"], "gold": item["gold"],
                        "prediction": prediction, "correct": prediction == item["gold"]})
    if extract_answer("A or B") is not None or extract_answer("reasoning mentions A\nfinal text unclear") is not None:
        raise ValueError("Dry-run answer extraction accepted an ambiguous completion")
    report = build_report(status="dry_run", provenance=[{"repo": "synthetic-fixture", "revision": None,
                                                        "subset": "fixture", "split": "fixture", "rows_seen": 3}],
                          results=results, limit_per_benchmark=None,
                          sampling={"provider_called": False, "temperature": 0, "max_tokens": 128})
    validate_report(report)
    return report


def build_preflight_report(provenance, records, *, limit_per_benchmark):
    checks = Counter()
    by_benchmark = Counter()
    for item in records:
        messages = messages_for(item)
        letters = item["choices"]
        if (len(messages) == 2 and messages[0]["role"] == "system"
                and messages[1]["role"] == "user"
                and all(f"{letter}. " in messages[1]["content"] for letter in letters)
                and item["gold"] in letters):
            checks["valid_prompt_and_answer"] += 1
        else:
            checks["malformed_prompt_or_answer"] += 1
        by_benchmark[item["benchmark"]] += 1
    if checks["malformed_prompt_or_answer"]:
        raise ValueError("Source preflight found malformed parsed rows")
    sources = {}
    for entry in provenance:
        if entry.get("status") != "loaded":
            continue
        sources[entry["benchmark"]] = {
            "repo": entry["repo"], "revision": entry["revision"],
            "subsets": entry["subsets"], "split": entry["split"],
            "rows_seen": entry["rows_seen"], "valid_rows": entry["valid_rows"],
            "malformed_rows": entry["rows_seen"] - entry["valid_rows"],
            "exact_duplicates_removed": entry["exact_duplicates_removed"],
            "conflicting_duplicate_questions": entry["conflicting_duplicate_questions"],
            "conflicting_duplicate_rows_removed": entry["conflicting_duplicate_rows_removed"],
            "eligible_unique_rows": entry["eligible_rows"],
            "sampled_rows": entry["evaluated_rows"],
        }
    return {"schema_version": 1, "status": "source_preflight_complete",
            "model": MODEL_ID, "renderer": RENDERER,
            "split_policy": "official test split only; no train/validation/few-shot examples; revision pinned",
            "limit_per_benchmark": limit_per_benchmark,
            "tinker_calls": 0, "rendering": "prompt text/schema checked; Tinker renderer not invoked",
            "checks": dict(checks), "sources": sources,
            "sampled_rows_by_benchmark": dict(by_benchmark)}


def run_preflight(*, output, limit_per_benchmark=DEFAULT_LIMIT_PER_BENCHMARK):
    """Load and validate official test schemas on Modal without invoking Tinker."""
    import os
    if not os.environ.get("MODAL_TASK_ID"):
        raise RuntimeError("Source preflight must run on Modal to keep benchmark files off the local machine.")
    provenance, records = _load_public_test_rows(limit_per_benchmark)
    report = build_preflight_report(provenance, records, limit_per_benchmark=limit_per_benchmark)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    return {"status": report["status"], "output": str(output), "tinker_calls": 0,
            "sources": report["sources"], "checks": report["checks"]}


def _load_public_test_rows(limit_per_benchmark):
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    api = HfApi(token=False)
    all_records = []
    provenance = []
    for benchmark in BENCHMARKS:
        info = api.dataset_info(benchmark["repo"])
        revision = info.sha
        configs = get_dataset_config_names(benchmark["repo"], revision=revision)
        selected_configs = ["default" if benchmark["subset"] == "default" and "default" in configs
                            else benchmark["subset"]]
        total_seen = 0
        valid_seen = 0
        data_records = []
        for config in selected_configs:
            if config not in configs:
                raise ValueError(f"Configured benchmark subset not found: {benchmark['repo']}[{config}]")
            splits = get_dataset_split_names(benchmark["repo"], config_name=config, revision=revision)
            if benchmark["split"] not in splits:
                # Record every missing test config; never substitute validation/train.
                provenance.append({"benchmark": benchmark["key"], "repo": benchmark["repo"],
                                  "revision": revision, "subset": config, "split": benchmark["split"],
                                  "status": "missing_test_split", "rows_seen": 0})
                continue
            stream = load_dataset(benchmark["repo"], config, split=benchmark["split"],
                                  streaming=True, revision=revision, token=False)
            for index, raw in enumerate(stream):
                total_seen += 1
                item = parse_row(benchmark["adapter"], raw, subset=config, row_index=index)
                if item is not None:
                    valid_seen += 1
                    item["benchmark"] = benchmark["key"]
                    data_records.append(item)
        if not data_records:
            raise ValueError(f"No valid official test rows found for {benchmark['key']}")
        grouped_records = defaultdict(list)
        for item in data_records:
            key = " ".join(item["question"].split())
            grouped_records[key].append(item)
        unique_records = []
        duplicates_removed = 0
        conflicting_questions = 0
        conflicting_rows = 0
        for group in grouped_records.values():
            labels = {item["gold"] for item in group}
            if len(labels) > 1:
                conflicting_questions += 1
                conflicting_rows += len(group)
                continue
            unique_records.append(group[0])
            duplicates_removed += len(group) - 1
        eligible_count = len(unique_records)
        data_records = select_deterministic(unique_records, limit_per_benchmark)
        provenance.append({"benchmark": benchmark["key"], "repo": benchmark["repo"],
                           "revision": revision, "subsets": selected_configs, "split": benchmark["split"],
                           "status": "loaded", "rows_seen": total_seen, "valid_rows": valid_seen,
                           "exact_duplicates_removed": duplicates_removed,
                           "conflicting_duplicate_questions": conflicting_questions,
                           "conflicting_duplicate_rows_removed": conflicting_rows,
                           "eligible_rows": eligible_count,
                           "evaluated_rows": len(data_records)})
        all_records.extend(data_records)
    return provenance, all_records


def _row_key(row):
    identity = [row.get("benchmark"), row.get("subset"), row.get("id")]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _token_cost(tokens):
    return round(tokens.get("prompt", 0) * PRICES_PER_MILLION["prefill"] / 1e6
                 + tokens.get("completion", 0) * PRICES_PER_MILLION["sample"] / 1e6, 6)


def run_live(*, output, limit_per_benchmark=DEFAULT_LIMIT_PER_BENCHMARK,
             run_id="manual", checkpoint=lambda: None):
    """Fetch official test split streams on Modal and sample through Tinker."""
    import os
    if not os.environ.get("MODAL_TASK_ID"):
        raise RuntimeError("Live baseline runs must execute on Modal; use the Modal launcher.")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is missing from Modal secret 'tinker-secret'.")

    import tinker
    from tinker_cookbook import renderers, tokenizer_utils

    provenance, records = _load_public_test_rows(limit_per_benchmark)
    input_fingerprint = hashlib.sha256(json.dumps(
        {"model": MODEL_ID, "renderer": RENDERER, "prompt": PROMPT_VERSION,
         "limit_per_benchmark": limit_per_benchmark, "provenance": provenance,
         "rows": [_row_key(row) for row in records]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = renderers.get_renderer(RENDERER, tokenizer, model_name=MODEL_ID)
    client = tinker.ServiceClient().create_sampling_client(base_model=MODEL_ID)
    sampling_params = tinker.types.SamplingParams(temperature=0.0, max_tokens=128,
                                                 stop=renderer.get_stop_sequences())
    results = []
    token_counts = Counter()
    output = Path(output)
    predictions_path = output / "predictions.jsonl"
    progress_path = output / "progress.json"
    if output.exists():
        final_path = output / "manifest.json"
        if final_path.is_file():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            if final.get("run_id") != run_id or final.get("input_fingerprint") != input_fingerprint:
                raise ValueError("Existing completed run ID has different inputs; choose a new run ID.")
            return {"status": final["status"], "output": str(output),
                    "evaluated": sum(entry["n"] for entry in final["summary"].values()),
                    "summary": final["summary"], "estimated_usd": final["sampling"]["estimated_usd"]}
        if not progress_path.is_file() or not predictions_path.is_file():
            raise ValueError("Run directory already exists without resumable progress; choose a new run ID.")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        identity = {"run_id": run_id, "input_fingerprint": input_fingerprint}
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError("Run ID is already bound to different benchmark inputs.")
        committed = set(progress.get("committed_row_keys", []))
        planned = {_row_key(row) for row in records}
        if not committed <= planned:
            raise ValueError("Saved progress contains IDs outside the current selected test rows.")
        with predictions_path.open(encoding="utf-8") as old_stream:
            for line in old_stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _row_key(row) not in committed:
                    continue
                results.append(row)
                token_counts["prompt"] += row["prompt_tokens"]
                token_counts["completion"] += row["completion_tokens"]
        if {_row_key(row) for row in results} != committed:
            raise ValueError("Committed progress and durable prediction rows do not match; stopping to prevent repeat calls.")
        with predictions_path.open("w", encoding="utf-8") as durable_stream:
            for row in results:
                durable_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        output.mkdir(parents=True)
        committed = set()
    predictions_path.touch(exist_ok=True)

    def save_progress():
        progress = {"schema_version": 1, "status": "running", "run_id": run_id,
                    "input_fingerprint": input_fingerprint, "model": MODEL_ID,
                    "renderer": RENDERER, "prompt_version": PROMPT_VERSION,
                    "limit_per_benchmark": limit_per_benchmark,
                    "provenance": provenance, "committed_row_keys": sorted(committed),
                    "completed_rows": len(results), "tokens": dict(token_counts),
                    "estimated_usd": _token_cost(token_counts), "summary": summarize(results)}
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checkpoint()

    if not progress_path.exists():
        save_progress()
    pending_new = 0
    with predictions_path.open("a", encoding="utf-8") as stream:
        for index, item in enumerate(records, 1):
            key = _row_key(item)
            if key in committed:
                continue
            prompt = renderer.build_generation_prompt(messages_for(item))
            response = client.sample(prompt=prompt, num_samples=1, sampling_params=sampling_params).result()
            sequence = response.sequences[0]
            completion = tokenizer.decode(sequence.tokens)
            prediction = extract_answer(completion)
            row = {"benchmark": item["benchmark"], "subset": item["subset"],
                   "id": item["id"], "gold": item["gold"],
                   "prediction": prediction, "correct": prediction == item["gold"],
                   "completion": completion, "prompt_tokens": prompt.length,
                   "completion_tokens": len(sequence.tokens)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
            committed.add(key)
            pending_new += 1
            token_counts["prompt"] += prompt.length
            token_counts["completion"] += len(sequence.tokens)
            if pending_new >= 25 or len(committed) == len(records):
                stream.flush()
                save_progress()
                pending_new = 0
            if len(committed) % 25 == 0:
                print(json.dumps({"evaluated": index, "total": len(records)}), flush=True)
    report = build_report(status="complete", provenance=provenance, results=results,
                          limit_per_benchmark=limit_per_benchmark,
                          sampling={"temperature": 0.0, "max_tokens": 128,
                                    "stop_sequences": "renderer defaults", "tokens": dict(token_counts),
                                    "estimated_usd": _token_cost(token_counts),
                                    "listed_prices_per_million": PRICES_PER_MILLION})
    report["run_id"] = run_id
    report["input_fingerprint"] = input_fingerprint
    validate_report(report)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    checkpoint()
    return {"status": report["status"], "output": str(output), "evaluated": len(results),
            "summary": report["summary"], "estimated_usd": report["sampling"]["estimated_usd"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dry-run or launch the Barq Nemotron Arabic MCQ baseline.")
    parser.add_argument("--dry-run", action="store_true", help="validate synthetic fixtures without network/API calls")
    parser.add_argument("--out", type=Path, default=Path("reports/baseline-eval/dry-run.json"))
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("Use --dry-run locally; live inference is launched only by scripts/modal_baseline_eval.py.")
    report = dry_run_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]),
                      "accuracy": summarize(report["rows"]), "report": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
