import json
import zipfile

from barq.balsam_eval import load_archives, score, summarize


def _archive(path, name, payload):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"dev/{name}.json", json.dumps(payload, ensure_ascii=False))


def test_loads_nested_balsam_envelopes_and_normalizes_references(tmp_path):
    nested = tmp_path / "dev(6).zip"
    _archive(nested, "translation", {
        "task": "Machine Translation", "metric": "bleu",
        "json": {"name": "synthetic", "data": {"dev": [
            {"id": 1, "input": "Hello", "output": "مرحبا"},
            {"id": 2, "instruction": "ترجم", "input": "Hello",
             "output": ["مرحبا", "أهلا"]},
        ]}},
    })
    records, provenance = load_archives([nested])
    assert len(records) == 2
    assert records[0]["metric"] == "bleu"
    assert records[0]["references"] == ["مرحبا"]
    assert records[1]["references"] == ["مرحبا", "أهلا"]
    assert provenance == [{"archive": "dev(6).zip", "rows": 2}]


def test_scores_and_task_summary_keep_metric_groups_separate():
    bleu = score("hello there", ["hello there"], "bleu")
    rouge = score("hello there", ["hello there"], "rouge")
    accuracy = score("A", ["A"], "accuracy")
    assert bleu["bleu"] > 0.99
    assert rouge["rouge1"] == 1.0
    assert rouge["rougeL"] == 1.0
    assert accuracy["accuracy"] == 1.0
    assert summarize([
        {"task": "translation", "metric": "bleu", "scores": bleu},
        {"task": "composition", "metric": "rouge", "scores": rouge},
    ])["translation:bleu"]["n"] == 1
