"""Run with: uv run --locked --group modal modal run --detach scripts/modal_pipeline.py

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
# Match the lockfile registry rather than the older builder's ambient package mirror.
# Explicit source paths keep data/, reports/, .git/ and credentials out of the image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(REPO), groups=["modal"], frozen=False, uv_version="0.12.1",
             extra_options="--locked --no-dev --default-index https://pypi.org/simple")
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


@app.function(image=image, volumes={"/barq": volume}, cpu=1, memory=2048, timeout=300, retries=0)
def review_pack():
    """Build a blind 100-example page using only the small saved review pool."""
    from barq.review_pack import build

    volume.reload()
    root = Path("/barq")
    source = root / "reports/quality-inputs/20260905/review_samples.jsonl"
    # Required history fails closed if a retained pilot archive is missing.
    required = [root / "reports/quality/luna-pilot-20260905-v4.tar.gz",
                root / "reports/quality/muse-tightened-20260905.tar.gz",
                root / "reports/quality-inputs/20260905/calibration.jsonl"]
    if not source.is_file() or any(not path.is_file() for path in required):
        raise ValueError("Saved review pool or prior pilot history is missing from barq-data.")
    exclusions = sorted(set(required + list((root / "reports/quality").glob("*.tar.gz"))
                            + list((root / "reports/quality").rglob("sample.jsonl"))
                            + list((root / "reports/review-packs").glob("*/examples.jsonl"))))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = root / "reports/review-packs" / run_id
    manifest = build(source, exclusions, output)
    volume.commit()
    result = {"status": manifest["status"], "pack_id": manifest["pack_id"], "rows": manifest["rows"],
              "volume": VOLUME_NAME, "path": str(output.relative_to(root)),
              "page": str((output / "review.html").relative_to(root))}
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(image=image, volumes={"/barq": volume}, cpu=1, memory=1024, timeout=180)
def check():
    """Verify image imports and persistent storage without fetching any datasets."""
    from importlib.metadata import version
    from barq.curate import read_config as read_curation
    from barq.data import read_config as read_data
    from barq.persistent import run  # Import the actual orchestration and its dependencies.

    read_data(Path("/app/configs/data.yaml"))
    read_curation(Path("/app/configs/curation.yaml"))
    volume.reload()
    probe = Path("/barq") / (".startup-check-" + uuid4().hex)
    probe.write_text("barq-volume-ok", encoding="utf-8")
    volume.commit()
    volume.reload()
    assert probe.read_text(encoding="utf-8") == "barq-volume-ok"
    probe.unlink()
    volume.commit()
    result = {"status": "ok", "volume": VOLUME_NAME,
              "packages": {name: version(name) for name in ("barq", "modal", "datasets", "pyarrow")}}
    print(json.dumps(result, indent=2), flush=True)
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
