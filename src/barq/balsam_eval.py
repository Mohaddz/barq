"""Local-file loader and scoring helpers for the saved BALSAM v2 dev subset.

Source archives must be staged on the Modal volume before live evaluation. The module
returns no dataset text in summaries; source rows and predictions stay on that volume.
"""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import unicodedata
import zipfile
from hashlib import sha256

from barq.baseline_eval import MODEL_ID, RENDERER

BENCHMARK = "balsam_v2_saved_dev_subset"
DEFAULT_ARCHIVES = tuple(f"/barq/benchmarks/balsam-v2-saved/dev({i}).zip" for i in range(6, 10))
SFT_DEFAULT = Path("/barq/recuration/dc8a7acc2f2898bc/refinements/20260924T124044035552Z/source-fixes-20260924T134115Z/sft-candidate-20260925T183854Z")
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
MAX_EXPANDED_BYTES = 256 * 1024 * 1024


def _strings(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return [text for item in value if (text := _strings(item))]
    return ""


def _rows_from_file(raw):
    document = json.loads(raw.decode("utf-8-sig"))
    # KSAA export envelope: metadata.json.data.dev; simpler tasks use top-level data.dev.
    payload = document.get("json") if isinstance(document.get("json"), dict) else document
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("dev") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Expected a BALSAM v2 JSON envelope with data.dev rows")
    raw_metric = document.get("metric") or payload.get("metric")
    if not raw_metric:
        metric_list = payload.get("metric_list", [])
        if isinstance(metric_list, list) and metric_list:
            raw_metric = metric_list[0].get("metric") if isinstance(metric_list[0], dict) else metric_list[0]
    metric = str(raw_metric or "").lower()
    task = str(document.get("task") or payload.get("task") or document.get("category") or "unknown")
    task_name = str(payload.get("name") or document.get("name") or task)
    return task, task_name, metric, rows


def load_archives(paths=DEFAULT_ARCHIVES):
    """Read saved dev ZIPs and yield normalized prompt/reference records."""
    records, provenance = [], []
    for archive_path in map(Path, paths):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Saved BALSAM archive missing on Modal volume: {archive_path}")
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError(f"BALSAM archive exceeds size cap: {archive_path.name}")
        file_rows = 0
        with zipfile.ZipFile(archive_path) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            if sum(item.file_size for item in infos) > MAX_EXPANDED_BYTES:
                raise ValueError(f"Expanded BALSAM archive exceeds size cap: {archive_path.name}")
            for info in infos:
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise ValueError(f"Unsafe path in BALSAM archive: {archive_path.name}")
                if member.suffix.lower() != ".json":
                    continue
                task, task_name, metric, rows = _rows_from_file(archive.read(info))
                for index, row in enumerate(rows):
                    if not isinstance(row, dict) or not _strings(row.get("output")):
                        continue
                    instruction = _strings(row.get("instruction"))
                    inputs = _strings(row.get("input"))
                    if isinstance(inputs, list):
                        inputs = "\n".join(inputs)
                    if not instruction and not inputs:
                        continue
                    choices = _strings(row.get("mcq"))
                    if not isinstance(choices, list):
                        choices = [choices] if choices else []
                    prompt = "\n\n".join(part for part in (instruction, inputs) if part)
                    if choices:
                        prompt += "\n\nالخيارات:\n" + "\n".join(
                            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
                        )
                        prompt += "\nأجب بنص الخيار الصحيح فقط."
                    else:
                        prompt += "\nاتبع التعليمات وأكمل المهمة."
                    key = f"{archive_path.stem}:{info.filename}:{row.get('id', index)}"
                    references = _strings(row["output"])
                    if not isinstance(references, list):
                        references = [references]
                    records.append({"id": key, "task": task, "task_name": task_name,
                                    "metric": metric, "prompt": prompt,
                                    "references": references,
                                    "choices": choices if isinstance(choices, list) else []})
                    file_rows += 1
        provenance.append({"archive": archive_path.name, "rows": file_rows})
    if not records:
        raise ValueError("Saved BALSAM archives contained no evaluable dev rows")
    return records, provenance


def _normalize(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _accuracy(prediction, references):
    pred = _normalize(prediction)
    if not pred:
        return 0.0
    best = 0.0
    for ref in references:
        target = _normalize(ref)
        if not target:
            continue
        ratio = SequenceMatcher(None, target, pred).ratio()
        # KSAA's fuzzy accuracy uses RapidFuzz ratio or partial_ratio >= .85.
        partial = max((SequenceMatcher(None, target, pred[i:i+len(target)]).ratio()
                       for i in range(max(1, len(pred) - len(target) + 1))), default=0)
        best = max(best, ratio, partial)
    return float(best >= .85)


def _tokens(text):
    return text.split()


def _bleu(prediction, references):
    """Smoothed sentence BLEU with whitespace tokenization, matching KSAA's setup."""
    hyp = _tokens(prediction)
    refs = [_tokens(ref) for ref in references if ref]
    if not hyp or not refs:
        return 0.0
    closest = min(refs, key=lambda ref: (abs(len(ref) - len(hyp)), len(ref)))
    bp = min(1.0, math.exp(1 - len(closest) / len(hyp))) if len(hyp) else 0.0
    logs = []
    for n in range(1, 5):
        hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1)))
        max_ref = Counter()
        for ref in refs:
            counts = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1)))
            for gram, count in counts.items():
                max_ref[gram] = max(max_ref[gram], count)
        clipped = sum(min(count, max_ref[gram]) for gram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        # Effective-order smoothing for short Arabic answers.
        logs.append(math.log((clipped + 1) / (total + 1)))
    return bp * math.exp(sum(logs) / 4)


def _rouge(prediction, references):
    """Return max-reference ROUGE-1/2/L F1 using whitespace tokenization."""
    hyp = _tokens(prediction)
    if not hyp:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref_text in references:
        ref = _tokens(ref_text)
        scores = {}
        for n, name in ((1, "rouge1"), (2, "rouge2")):
            a = Counter(tuple(hyp[i:i+n]) for i in range(max(0, len(hyp)-n+1)))
            b = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1)))
            overlap = sum((a & b).values())
            p = overlap / sum(a.values()) if a else 0.0
            r = overlap / sum(b.values()) if b else 0.0
            scores[name] = 2*p*r/(p+r) if p+r else 0.0
        # Longest common subsequence, calculated with a compact row.
        prev = [0] * (len(ref) + 1)
        for token in hyp:
            curr = [0]
            for j, other in enumerate(ref, 1):
                curr.append(prev[j-1] + 1 if token == other else max(prev[j], curr[-1]))
            prev = curr
        lcs = prev[-1]
        p = lcs / len(hyp)
        r = lcs / len(ref) if ref else 0.0
        scores["rougeL"] = 2*p*r/(p+r) if p+r else 0.0
        best = {key: max(best[key], scores[key]) for key in best}
    return best


def score(prediction, references, metric):
    metric = metric.lower()
    if metric == "accuracy":
        return {"accuracy": _accuracy(prediction, references)}
    if metric == "bleu":
        return {"bleu": _bleu(prediction, references)}
    if metric == "rouge":
        return _rouge(prediction, references)
    raise ValueError(f"Unsupported BALSAM metric: {metric}")


def summarize(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row["task"], row["metric"])].append(row)
    summary = {}
    for (task, metric), rows in sorted(grouped.items()):
        metrics = defaultdict(list)
        for row in rows:
            for name, value in row["scores"].items():
                metrics[name].append(value)
        summary[f"{task}:{metric}"] = {"task": task, "metric": metric, "n": len(rows),
            **{name: sum(values) / len(values) for name, values in metrics.items() if values}}
    return summary


def check_overlap(records, sft_root):
    """Return counts only; checks prompts and references against SFT messages."""
    index = _OverlapIndex()
    for label, text in _sft_texts(Path(sft_root)):
        index.add(text, label)
    counts = Counter()
    for row in records:
        for value in [row["prompt"], *row["references"]]:
            hit = index.match(value)
            if hit:
                counts[hit["kind"]] += 1
    return {"matches": dict(counts), "checked_rows": len(records),
            "method": "NFC/whitespace exact and 5-word-shingle Jaccard >= 0.85 against SFT user and assistant messages"}


def _normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def _shingles(text):
    words = re.findall(r"\w+", _normalized(text).lower())
    if len(words) < 8:
        return set()
    return {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}


class _OverlapIndex:
    """Small copy of Barq's exact and 5-word-shingle index without data-pipeline imports."""
    def __init__(self):
        self.exact = {}
        self.sets = []
        self.postings = defaultdict(set)
        self.labels = []

    def add(self, text, label):
        text = _normalized(text)
        if not text or text in self.exact:
            return
        self.exact[text] = label
        shingles = _shingles(text)
        index = len(self.sets)
        self.sets.append(shingles)
        self.labels.append(label)
        for shingle in shingles:
            self.postings[shingle].add(index)

    def match(self, text):
        text = _normalized(text)
        if text in self.exact:
            return {"kind": "exact", "reference": self.exact[text]}
        shingles = _shingles(text)
        if not shingles:
            return None
        candidates = Counter(index for shingle in shingles
                             for index in self.postings.get(shingle, ()))
        for index, shared in candidates.items():
            score = shared / (len(shingles) + len(self.sets[index]) - shared)
            if score >= 0.85:
                return {"kind": "near", "reference": self.labels[index], "jaccard": score}
        return None


def _sft_texts(sft_root):
    for filename in ("train.jsonl", "validation.jsonl"):
        path = Path(sft_root) / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for index, message in enumerate(row.get("messages", [])):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        yield f"{filename}:{row.get('id', 'unknown')}:{index}", content


def run_live(*, archives=DEFAULT_ARCHIVES, output, sft_root, checkpoint=lambda: None):
    """Evaluate the saved BALSAM dev subset with the configured Tinker model."""
    import os
    if not os.environ.get("MODAL_TASK_ID"):
        raise RuntimeError("Saved BALSAM evaluation must execute on Modal.")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is missing from Modal secret 'tinker-secret'.")

    import tinker
    from tinker_cookbook import renderers, tokenizer_utils
    from barq.baseline_eval import _token_cost
    records, provenance = load_archives(archives)
    index = _OverlapIndex()
    for label, text in _sft_texts(Path(sft_root)):
        index.add(text, label)
    clean_records, overlap = [], Counter()
    for row in records:
        hits = [index.match(value) for value in [row["prompt"], *row["references"]]]
        hits = [hit for hit in hits if hit]
        if hits:
            overlap.update(hit["kind"] for hit in hits)
        else:
            clean_records.append(row)
    if not clean_records:
        raise ValueError("All saved BALSAM rows overlap the SFT candidate; no rows remain to evaluate.")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_ID)
    renderer = renderers.get_renderer(RENDERER, tokenizer, model_name=MODEL_ID)
    client = tinker.ServiceClient().create_sampling_client(base_model=MODEL_ID)
    sampling_params = tinker.types.SamplingParams(temperature=0.0, max_tokens=512,
                                                 stop=renderer.get_stop_sequences())
    fingerprint = sha256(json.dumps(
        {"model": MODEL_ID, "benchmark": BENCHMARK,
         "rows": [{"id": row["id"], "prompt": row["prompt"],
                   "references": row["references"]} for row in clean_records]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    results, tokens = [], Counter()
    predictions_path = output / "predictions.jsonl"
    progress_path = output / "progress.json"
    final_path = output / "manifest.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if final.get("input_fingerprint") != fingerprint:
            raise ValueError("Completed BALSAM run directory belongs to different inputs; choose a new run ID.")
        return {"status": final["status"], "benchmark": BENCHMARK,
                "output": str(output), "eligible_rows": final["eligible_rows"],
                "evaluated_rows": final["evaluated_rows"],
                "excluded_overlap_rows": final["overlap"]["excluded_rows"],
                "summary": final["summary"], "estimated_usd": final["sampling"]["estimated_usd"]}
    committed = set()
    if predictions_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("input_fingerprint") != fingerprint:
            raise ValueError("Saved BALSAM progress belongs to different inputs; choose a new run ID.")
        planned = {row["id"] for row in clean_records}
        committed = set(progress.get("committed_ids", []))
        if not committed <= planned:
            raise ValueError("Saved progress contains IDs outside the current BALSAM inputs.")
        with predictions_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id") not in committed:
                    continue
                results.append(row)
                tokens["prompt"] += row["prompt_tokens"]
                tokens["completion"] += row["completion_tokens"]
        if {row["id"] for row in results} != committed:
            raise ValueError("Committed BALSAM progress does not match durable prediction rows.")
        with predictions_path.open("w", encoding="utf-8") as stream:
            for row in results:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        if predictions_path.exists() or progress_path.exists():
            raise ValueError("BALSAM run directory has incomplete progress files; choose a new run ID.")
        predictions_path.touch()

    def save_progress():
        progress = {"status": "running", "input_fingerprint": fingerprint,
                    "model": MODEL_ID, "committed_ids": sorted(committed),
                    "completed_rows": len(results), "tokens": dict(tokens)}
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        checkpoint()

    if not progress_path.exists():
        save_progress()
    pending = 0
    with predictions_path.open("a", encoding="utf-8") as stream:
        for position, item in enumerate(clean_records, 1):
            if item["id"] in committed:
                continue
            system = ("اختر الخيار الأنسب، وأخرج نص الخيار الصحيح فقط."
                      if item["choices"] else "اتبع تعليمات المهمة وأجب مباشرة باللغة المطلوبة.")
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": item["prompt"]}]
            prompt = renderer.build_generation_prompt(messages)
            response = client.sample(prompt=prompt, num_samples=1,
                                     sampling_params=sampling_params).result()
            sequence = response.sequences[0]
            completion = tokenizer.decode(sequence.tokens).strip()
            scores = score(completion, item["references"], item["metric"])
            row = {"benchmark": BENCHMARK, "task": item["task"],
                   "task_name": item["task_name"], "metric": item["metric"],
                   "id": item["id"], "prediction": completion,
                   "references": item["references"], "scores": scores,
                   "prompt_tokens": prompt.length,
                   "completion_tokens": len(sequence.tokens)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)
            committed.add(item["id"])
            pending += 1
            tokens["prompt"] += prompt.length
            tokens["completion"] += len(sequence.tokens)
            if pending >= 25 or len(committed) == len(clean_records):
                stream.flush()
                save_progress()
                pending = 0
                print(json.dumps({"evaluated": len(committed), "total": len(clean_records)}), flush=True)
    report = {"schema_version": 1, "status": "complete",
              "benchmark": BENCHMARK, "label": "saved BALSAM v2 dev subset",
              "model": MODEL_ID, "renderer": RENDERER,
              "split": "saved dev archives dev(6).zip through dev(9).zip",
              "scoring_note": "Local BLEU, ROUGE, and fuzzy-accuracy approximations follow the vendored KSAA metric setup; exact scores may differ because this path does not call KSAA's evaluate/RapidFuzz implementations.",
              "provenance": provenance,
              "input_fingerprint": fingerprint,
              "overlap": {"excluded_rows": len(records) - len(clean_records),
                          "match_kinds": dict(overlap),
                          "method": "NFC/whitespace exact and 5-word-shingle Jaccard >= 0.85 against SFT user and assistant messages"},
              "eligible_rows": len(records), "evaluated_rows": len(results),
              "summary": summarize(results), "sampling": {"temperature": 0.0,
                  "max_tokens": 512, "tokens": dict(tokens),
                  "estimated_usd": _token_cost(tokens)},
              "rows": results}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    checkpoint()
    return {"status": report["status"], "benchmark": BENCHMARK,
            "output": str(output), "eligible_rows": len(records),
            "evaluated_rows": len(results), "excluded_overlap_rows": len(records)-len(clean_records),
            "summary": report["summary"], "estimated_usd": report["sampling"]["estimated_usd"]}
