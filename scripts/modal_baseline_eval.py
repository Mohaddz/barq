"""Run the baseline remotely (requires Modal secret `tinker-secret`).

    uv run --locked --group modal modal run --detach scripts/modal_baseline_eval.py --live

Resume an interrupted run with its original `--run-id`.

Tinker calls only happen when `--live` is explicitly passed.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
app = modal.App("barq-baseline-eval")
volume = modal.Volume.from_name("barq-data")
source_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "datasets==5.0.1",
        "huggingface-hub==1.30.0",
    )
    .add_local_dir(REPO / "src/barq", "/app/barq", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/app", "HF_HOME": "/barq/hf-cache"})
    .workdir("/app")
)
image = source_image.pip_install("tinker==0.30.1", "tinker-cookbook==0.5.5")


def _safe_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) or ".." in value:
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    return value


@app.function(
    image=image,
    volumes={"/barq": volume},
    secrets=[modal.Secret.from_name("tinker-secret")],
    cpu=2,
    memory=8192,
    timeout=24 * 60 * 60,
    max_containers=1,
    retries=0,
)
def evaluate(limit_per_benchmark: int = 100, run_id: str = "manual"):
    from barq.baseline_eval import run_live

    volume.reload()
    output = Path("/barq/reports/baseline-eval") / run_id
    result = run_live(output=output, limit_per_benchmark=limit_per_benchmark,
                      run_id=run_id, checkpoint=volume.commit)
    volume.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(live: bool = False, limit_per_benchmark: int = 100, run_id: str = ""):
    if not live:
        raise ValueError("Use `python -m barq.baseline_eval --dry-run` locally, or pass --live after Tinker setup.")
    if limit_per_benchmark < 1:
        raise ValueError("limit-per-benchmark must be positive")
    chosen_run_id = _safe_run_id(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    call = evaluate.spawn(limit_per_benchmark, chosen_run_id)
    print(json.dumps({"status": "submitted", "app_id": app.app_id,
                      "call_id": call.object_id, "volume": "barq-data",
                      "run_id": chosen_run_id,
                      "limit_per_benchmark": limit_per_benchmark,
                      "results": "/barq/reports/baseline-eval"}, indent=2))
