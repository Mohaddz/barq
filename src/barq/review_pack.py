"""Small blind human-review packs; no downloads or model calls."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import tarfile

from barq.quality import digest, json_text, load_samples, now, read_jsonl, select_samples

KEYS = ("id", "example_hash", "input_hash")
LABELS = ("usable", "flawed", "unsure")
MAX_BYTES = 50 * 1024 * 1024


def history_rows(path):
    """Read only sample members, without extracting archives or running their code."""
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Use small history files (at most 50 MiB).")
    if not path.name.endswith(".tar.gz"):
        return read_jsonl(path)[0]
    rows, total = [], 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).name not in {"sample.jsonl", "examples.jsonl"}:
                continue
            total += member.size
            if total > MAX_BYTES:
                raise ValueError("History archive samples exceed 50 MiB.")
            with archive.extractfile(member) as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
    if not rows:
        raise ValueError(f"No sample history found in {path}.")
    return rows


def write_rows(path, rows):
    path.write_text("".join(json_text(row) + "\n" for row in rows), encoding="utf-8")


def build(input_path, exclusions, output, *, limit=100, seed=44):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Pack size must be between 1 and 1000.")
    if not exclusions:
        raise ValueError("Supply prior sample/assessment files to establish freshness.")
    rows, source_hash, skipped = load_samples(input_path)
    seen = {key: set() for key in KEYS}
    history = []
    for path in exclusions:
        previous = history_rows(path)
        for row in previous:
            if not isinstance(row, dict) or not any(row.get(key) for key in KEYS):
                raise ValueError(f"Invalid history record in {path}.")
            for key in KEYS:
                if row.get(key):
                    seen[key].add(row[key])
        history.append({"path": str(path), "sha256": digest(path.read_bytes()), "rows": len(previous)})
    eligible = [row for row in rows if not any(row[key] in seen[key] for key in KEYS)]
    # Duplicate prompts count once, even when they have several acceptable targets.
    unique, input_seen = [], set()
    for row in sorted(eligible, key=lambda row: digest([seed, row["id"]])):
        if row["input_hash"] not in input_seen:
            unique.append(row)
            input_seen.add(row["input_hash"])
    selected = select_samples(unique, {}, limit, seed)
    if len(selected) != limit:
        raise ValueError(f"Only {len(selected)} fresh examples available; requested {limit}.")
    selected.sort(key=lambda row: digest([seed, "display", row["id"]]))
    # Originals stay in examples.jsonl for provenance; the page receives a strict allowlist.
    visible = [{**{key: row[key] for key in KEYS},
                "messages": [{key: message[key] for key in
                              ("role", "content", "tool_calls", "tool_call_id", "name") if key in message}
                             for message in row["messages"]], "tools": row.get("tools", [])}
               for row in selected]
    pack_id = digest({"examples": visible, "seed": seed})[:20]
    manifest = {"schema_version": 1, "pack_id": pack_id, "created_at": now(), "rows": limit,
                "input": str(input_path), "input_sha256": source_hash, "exclusions": history,
                "available_training_rows": len(rows), "excluded_by_history": len(rows) - len(eligible),
                "unique_eligible_inputs": len(unique), "skipped": skipped, "seed": seed,
                "groups": [{"source": a, "task": b, "rows": count} for (a, b), count in
                           sorted(Counter((r["source"], r["task"]) for r in selected).items())],
                "scope": "Balanced existing review pool; not a corpus quality estimate. Fresh by stored hashes, not proven semantic-family disjoint."}
    payload = json_text({"pack_id": pack_id, "rows": visible}).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = Path(__file__).with_name("review.html").read_text(encoding="utf-8").replace("__PACK_JSON__", payload)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "examples.jsonl", selected)
    manifest["examples_sha256"] = digest((output / "examples.jsonl").read_bytes())
    (output / "review.html").write_text(html, encoding="utf-8")
    manifest["review_html_sha256"] = digest((output / "review.html").read_bytes())
    manifest["status"] = "complete"
    (output / "manifest.json").write_text(json_text(manifest) + "\n", encoding="utf-8")
    return manifest


def pack_rows(pack):
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    rows, sha = read_jsonl(pack / "examples.jsonl")
    if manifest.get("status") != "complete" or sha != manifest["examples_sha256"] or len(rows) != manifest["rows"]:
        raise ValueError("Pack is incomplete or its examples changed.")
    return manifest, {row["id"]: row for row in rows}


def validate_labels(labels, manifest, examples):
    seen = set()
    for row in labels:
        row_id = row.get("id")
        if row_id not in examples or row_id in seen or row.get("pack_id") != manifest["pack_id"]:
            raise ValueError("Labels contain a duplicate, unknown ID or different pack.")
        if any(row.get(key) != examples[row_id][key] for key in KEYS) or row.get("schema_version") != 1:
            raise ValueError("Label identity does not match this example.")
        if row.get("label") not in LABELS or not isinstance(row.get("note", ""), str):
            raise ValueError("Invalid human label or note.")
        seen.add(row_id)
    if seen != set(examples):
        raise ValueError("Finish all examples before freezing labels.")


def freeze(pack, labels_path):
    manifest, examples = pack_rows(pack)
    labels, _ = read_jsonl(labels_path)
    validate_labels(labels, manifest, examples)
    # Exclusive creation prevents later edits from silently changing the reference set.
    with (pack / "human_labels.jsonl").open("x", encoding="utf-8") as stream:
        stream.write("".join(json_text(row) + "\n" for row in labels))
    receipt = {"frozen_at": now(), "sha256": digest((pack / "human_labels.jsonl").read_bytes())}
    (pack / "labels_frozen.json").write_text(json_text(receipt) + "\n", encoding="utf-8")
    return receipt


def compare(pack, judgments_path):
    manifest, examples = pack_rows(pack)
    labels, sha = read_jsonl(pack / "human_labels.jsonl")
    if sha != json.loads((pack / "labels_frozen.json").read_text())["sha256"]:
        raise ValueError("Frozen human labels changed.")
    validate_labels(labels, manifest, examples)
    judgments, _ = read_jsonl(judgments_path)
    model = {}
    mapping = {"keep": "usable", "repair": "flawed", "drop": "flawed", "review": "unsure"}
    for row in judgments:
        row_id = row.get("id")
        if row_id not in examples or row_id in model or any(row.get(k) != examples[row_id][k] for k in KEYS):
            raise ValueError("Judgments contain duplicate, unknown or mismatched examples.")
        model[row_id] = mapping[row["judgment"]["decision"]] if row.get("status") == "complete" else "unjudged"
    counts = defaultdict(Counter)
    comparisons = []
    for label in labels:
        row = examples[label["id"]]
        predicted = model.get(row["id"], "unjudged")
        bucket = counts[row["task"]]
        bucket["rows"] += 1
        bucket["unjudged"] += predicted == "unjudged"
        bucket["agreement"] += predicted == label["label"]
        bucket["flawed_kept"] += label["label"] == "flawed" and predicted == "usable"
        bucket["usable_rejected"] += label["label"] == "usable" and predicted == "flawed"
        comparisons.append({"id": row["id"], "task": row["task"], "human": label["label"], "judge": predicted})
    write_rows(pack / "comparisons.jsonl", comparisons)
    lines = ["# Human review vs Muse", "", "Agreement with this small human reference set is not corpus-wide accuracy.",
             "Flawed combines repair/drop; usable does not certify unsupported facts. Unsure is not a confirmed error.", "",
             "| Task | Examples | Agree | Human-flawed kept | Human-usable rejected | Unjudged |",
             "|---|---:|---:|---:|---:|---:|"]
    for task, count in sorted(counts.items()):
        safe_task = task.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {safe_task} | " + " | ".join(str(count[k]) for k in
                     ("rows", "agreement", "flawed_kept", "usable_rejected", "unjudged")) + " |")
    (pack / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze", help="Freeze a completed browser export before running the judge")
    freeze_parser.add_argument("--pack", type=Path, required=True)
    freeze_parser.add_argument("--labels", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--pack", type=Path, required=True)
    compare_parser.add_argument("--judgments", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.pack, args.labels) if args.command == "freeze" else compare(args.pack, args.judgments)
    print(json_text(result))


if __name__ == "__main__":
    main()
