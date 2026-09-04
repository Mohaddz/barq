"""Run with: modal run --detach scripts/modal_pipeline.py

Only source/configuration files are uploaded. Datasets are fetched on Modal.
Run one job at a time against this named Volume.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import modal


REPO = Path(__file__).resolve().parents[1]
VOLUME_NAME = "barq-data"
app = modal.App("barq-data-pipeline")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=1)

# uv_sync uploads only pyproject.toml/uv.lock and installs their dependencies.
# Explicit source paths keep data/, reports/, .git/ and credentials out of the image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(REPO), frozen=False, uv_version="0.12.1", extra_options="--locked --no-dev")
    .add_local_file(REPO / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(REPO / "README.md", "/app/README.md", copy=True)
    .add_local_dir(REPO / "src", "/app/src", copy=True, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(REPO / "configs" / "data.yaml", "/app/configs/data.yaml", copy=True)
    .add_local_file(REPO / "configs" / "curation.yaml", "/app/configs/curation.yaml", copy=True)
    .run_commands(
        "/.uv/uv pip install --python /.uv/.venv/bin/python --no-deps /app",
        "ln -s /barq/data /app/data",
    )
    .env({"HF_HOME": "/barq/hf-cache"})
    .workdir("/app")
)


@app.function(image=image, volumes={"/barq": volume}, cpu=4, memory=16384,
              timeout=4 * 60 * 60, max_containers=1, retries=0)
def pipeline():
    from barq.persistent import run

    volume.reload()
    result = run(Path("/barq"), config_dir=Path("/app/configs"), commit=volume.commit, workers=4)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main():
    call = pipeline.spawn()
    receipt = {"app_id": app.app_id, "call_id": call.object_id, "volume": VOLUME_NAME,
               "status": "submitted", "submitted_at": datetime.now(timezone.utc).isoformat(),
               "logs_command": f"modal app logs {app.app_id} -f"}
    directory = REPO / "reports" / "modal-launch"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json")
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Submitted: {call.object_id}\nReceipt: {path}\nVolume: {VOLUME_NAME}")
    print(f"Logs: {receipt['logs_command']}")
