"""Credential-free schema and SFT-overlap preflight for saved BALSAM dev ZIPs."""
import json
from pathlib import Path
import sys

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
app = modal.App("barq-balsam-saved-preflight")
volume = modal.Volume.from_name("barq-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(REPO / "src/barq", "/app/barq", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/app"})
    .workdir("/app")
)


@app.function(image=image, volumes={"/barq": volume}, cpu=2, memory=8192,
              timeout=30 * 60, max_containers=1, retries=0)
def check():
    from collections import Counter
    from barq.balsam_eval import DEFAULT_ARCHIVES, SFT_DEFAULT, _OverlapIndex, _sft_texts, load_archives

    volume.reload()
    records, provenance = load_archives(DEFAULT_ARCHIVES)
    row_counts = Counter((row["task"], row["metric"]) for row in records)
    index = _OverlapIndex()
    for label, text in _sft_texts(Path(SFT_DEFAULT)):
        index.add(text, label)
    overlap = Counter()
    eligible_counts = Counter()
    excluded = 0
    for row in records:
        hits = [index.match(value) for value in [row["prompt"], *row["references"]]]
        hits = [hit for hit in hits if hit]
        if hits:
            excluded += 1
            overlap.update(hit["kind"] for hit in hits)
        else:
            eligible_counts[(row["task"], row["metric"])] += 1
    report = {"status": "preflight_complete", "benchmark": "saved BALSAM v2 dev subset",
              "archives": provenance, "total_rows": len(records),
              "rows_by_task_metric": {f"{task}:{metric}": count
                                      for (task, metric), count in sorted(row_counts.items())},
              "eligible_by_task_metric": {f"{task}:{metric}": count
                                          for (task, metric), count in sorted(eligible_counts.items())},
              "eligible_after_overlap": len(records) - excluded,
              "overlap_excluded_rows": excluded, "overlap_match_kinds": dict(overlap),
              "live_tinker_calls": 0,
              "live_command": "uv run --locked --group modal modal run scripts/modal_balsam_saved_eval.py --live"}
    output = Path("/barq/reports/balsam-saved-eval/preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    print(json.dumps(check.remote(), ensure_ascii=False, indent=2))
