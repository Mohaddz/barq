"""Check official Arabic MCQ test sources on Modal without model calls.

    modal run --detach scripts/modal_baseline_preflight.py --limit-per-benchmark 100
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re

import modal

app = modal.App("barq-baseline-source-preflight")
volume = modal.Volume.from_name("barq-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets==5.0.1", "huggingface-hub==1.30.0")
    .add_local_dir(Path(__file__).resolve().parents[1] / "src/barq", "/app/barq", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/app", "HF_HOME": "/barq/hf-cache"})
    .workdir("/app")
)


def safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) or ".." in value:
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    return value


@app.function(image=image, volumes={"/barq": volume}, cpu=2, memory=8192,
              timeout=2 * 60 * 60, max_containers=1, retries=0)
def preflight(limit_per_benchmark: int = 100, run_id: str = "manual"):
    from barq.baseline_eval import run_preflight

    volume.reload()
    output = Path("/barq/reports/baseline-eval/source-preflight") / run_id
    result = run_preflight(output=output, limit_per_benchmark=limit_per_benchmark)
    volume.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(limit_per_benchmark: int = 100, run_id: str = ""):
    if limit_per_benchmark < 1:
        raise ValueError("limit-per-benchmark must be positive")
    chosen_run_id = safe_run_id(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    call = preflight.spawn(limit_per_benchmark, chosen_run_id)
    print(json.dumps({"status": "submitted", "app_id": app.app_id,
                      "call_id": call.object_id, "run_id": chosen_run_id,
                      "tinker_calls": 0, "volume": "barq-data"}, indent=2))
