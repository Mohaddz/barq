"""Run the saved BALSAM v2 dev subset on Modal (requires tinker-secret)."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
app = modal.App("barq-balsam-saved-eval")
volume = modal.Volume.from_name("barq-data")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("tinker==0.30.1", "tinker-cookbook==0.5.5")
    .add_local_dir(REPO / "src/barq", "/app/barq", copy=True,
                   ignore=["**/__pycache__/**", "**/*.pyc"])
    .env({"PYTHONPATH": "/app"})
    .workdir("/app")
)


def _safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) or ".." in value:
        raise ValueError("run-id may contain only letters, numbers, dot, underscore, and hyphen")
    return value


@app.function(image=image, volumes={"/barq": volume},
              secrets=[modal.Secret.from_name("tinker-secret")], cpu=2,
              memory=8192, timeout=24 * 60 * 60, max_containers=1, retries=0)
def evaluate(run_id="manual"):
    from barq.balsam_eval import DEFAULT_ARCHIVES, SFT_DEFAULT, run_live

    volume.reload()
    output = Path("/barq/reports/balsam-saved-eval") / run_id
    result = run_live(archives=DEFAULT_ARCHIVES, output=output,
                      sft_root=SFT_DEFAULT, checkpoint=volume.commit)
    volume.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(live: bool = False, run_id: str = ""):
    if not live:
        raise ValueError("Stage archives, run modal_balsam_saved_preflight.py, then pass --live to evaluate.")
    chosen = _safe_id(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    call = evaluate.spawn(chosen)
    print(json.dumps({"status": "submitted", "app_id": app.app_id,
                      "call_id": call.object_id, "volume": "barq-data",
                      "run_id": chosen,
                      "results": "/barq/reports/balsam-saved-eval"}, indent=2))
