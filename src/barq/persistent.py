"""Two durable stages, reusing only matching complete artifacts; no cloud dependency."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq

from barq.curate import CURATION_SCHEMA, run as curate_run
from barq.data import CANDIDATE_SCHEMA, digest, json_text, read_config, run as prepare_run
from barq.review import open_inputs


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _implementation(names):
    return hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in names)).hexdigest()


def _completed(root, stage, matches):
    candidates = []
    for path in (root / "reports" / stage).glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_bytes())
            if ((manifest["status"], manifest["mode"], manifest["schema_version"]) == ("complete", stage, 1)
                    and manifest["run_id"] == path.parent.name and isinstance(manifest["finished_at"], str)):
                candidates.append((manifest["finished_at"], path, manifest))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for _, path, manifest in sorted(candidates, reverse=True):
        output = root / "data" / "processed" / stage / path.parent.name
        try:
            if matches(output, path, manifest):
                return output, path.parent
        except (OSError, ValueError, KeyError, TypeError):
            continue  # An unverifiable cache entry never substitutes for a completed stage.
    return None


def run(root: Path, *, config_dir: Path, commit: Callable[[], None], workers: int = 4) -> dict:
    """Commit closed stages, preserve failures/cache, and publish latest only on success.

    Reuse validates recorded identities and Parquet footers, not every data byte. Run
    one writer per root; the hosting wrapper owns any concurrency/volume coordination.
    """
    root = root.resolve()
    try:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        root.mkdir(parents=True, exist_ok=True)
        config_dir = config_dir.resolve(strict=True)
        data_path, curation_path = config_dir / "data.yaml", config_dir / "curation.yaml"
        config = read_config(data_path)
        config_hash, curation_hash = digest(json_text(config)), _file_hash(curation_path)
        prepare_code = _implementation(("data.py", "rules.py"))
        curate_code = _implementation(("curate.py", "rules.py", "data.py", "review.py"))
        names = {item["name"] for item in config["datasets"]}
        benchmarks = {item["name"]: _file_hash((config_dir / item["path"]).resolve())
                      for item in config.get("benchmarks", []) if item.get("path") is not None}

        def matching_prepare(output, path, manifest):
            if (manifest.get("config_sha256") != config_hash
                    or digest(json_text(manifest.get("config"))) != config_hash
                    or manifest.get("implementation_sha256") != prepare_code
                    or not isinstance(manifest.get("selected_datasets"), list)
                    or len(manifest["selected_datasets"]) != len(names)
                    or set(manifest["selected_datasets"]) != names):
                return False
            if any({item["name"]: item.get("sha256") for item in manifest.get("benchmarks", [])}.get(name) != value
                   for name, value in benchmarks.items()):
                return False
            open_inputs(output, path)
            return all((path.parent / name).is_file() for name in ("audit.md", "samples.jsonl"))

        preparation = _completed(root, "prepare", matching_prepare)
        if preparation is None:
            preparation = prepare_run(config, data_path, "prepare", root, workers=workers)
            commit()
            if not matching_prepare(preparation[0], preparation[1] / "manifest.json",
                                    json.loads((preparation[1] / "manifest.json").read_bytes())):
                raise ValueError("New prepare artifacts do not match the requested pipeline.")
        else:
            print(f"prepare: reusing complete run {preparation[0].name}", flush=True)
        prepared, prepare_report = preparation
        _, prepare_manifest, prepare_info, prepare_hash, input_files = open_inputs(prepared, prepare_report / "manifest.json")

        def matching_curate(output, path, manifest):
            if (manifest.get("input_run_id") != prepare_info["run_id"]
                    or manifest.get("input_manifest_sha256") != prepare_hash
                    or manifest.get("config_sha256") != curation_hash
                    or manifest.get("implementation_sha256") != curate_code):
                return False
            counts = manifest["counts"]
            if (set(counts) != {"accept", "review", "repair", "exclude"}
                    or any(type(n) is not int or n < 0 for n in counts.values())
                    or type(manifest["training_rows"]) is not int or sum(counts.values()) != manifest["training_rows"]):
                return False
            skipped = manifest["skipped_candidate_rows"]
            if (not isinstance(skipped, dict) or any(type(n) is not int or n < 0 for n in skipped.values())
                    or manifest["training_rows"] + sum(skipped.values()) != input_files["candidates"]["rows"]):
                return False
            for name, info in input_files.items():
                if any(manifest["input_files"][name][key] != info[key] for key in ("rows", "size_bytes", "schema_sha256")):
                    return False
            for name, expected, schema in (("candidates", counts["accept"], CANDIDATE_SCHEMA),
                                           ("decisions", manifest["training_rows"], CURATION_SCHEMA)):
                with pq.ParquetFile(output / f"{name}.parquet") as parquet:
                    if parquet.metadata.num_rows != expected or not parquet.schema_arrow.equals(schema, check_metadata=False):
                        return False
            return all((path.parent / name).is_file() for name in ("curation.md", "review_samples.jsonl"))

        curation = _completed(root, "curate", matching_curate)
        if curation is None:
            curation = curate_run(prepared, config_path=curation_path,
                                 manifest_path=prepare_manifest, output_root=root)
            path = curation[1] / "manifest.json"
            manifest = json.loads(path.read_bytes())
            if ((manifest.get("status"), manifest.get("mode"), manifest.get("schema_version")) != ("complete", "curate", 1)
                    or manifest.get("run_id") != curation[0].name or not matching_curate(curation[0], path, manifest)):
                raise ValueError("New curation artifacts are not complete and consistent.")
        else:
            print(f"curate: reusing complete run {curation[0].name}", flush=True)
        commit()  # Curate has returned with closed files, or its persisted outputs were verified.
        result = {"prepare_run_id": prepared.name, "curate_run_id": curation[0].name,
                  "prepare_report": (prepare_report / "audit.md").relative_to(root).as_posix(),
                  "prepare_manifest": (prepare_report / "manifest.json").relative_to(root).as_posix(),
                  "curate_report": (curation[1] / "curation.md").relative_to(root).as_posix(),
                  "curate_manifest": (curation[1] / "manifest.json").relative_to(root).as_posix()}
        temporary = root / ".latest.json.tmp"
        temporary.write_text(json_text(result) + "\n", encoding="utf-8")
        temporary.replace(root / "latest.json")
        commit()
        return result
    except BaseException as error:
        try:
            commit()
        except BaseException as commit_error:
            error.add_note(f"Persisting the failed run also failed: {commit_error}")
        raise
