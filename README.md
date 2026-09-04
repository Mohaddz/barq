# Barq — Arabic data preparation

Phase 0 prepares and audits Arabic chat and tool-use data before training. It uses pinned copies of [arabic-sft-mix-2](https://huggingface.co/datasets/Mohaddz/arabic-sft-mix-2) and [AISA-ArabicFC](https://huggingface.co/datasets/TuwaiqAcademy/AISA-ArabicFC).

No training, RL, model judging, paid API calls or credentials are required. Audit and preparation need internet access to fetch public data. Review and curation use only prepared files already on disk.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on the other machine. The repository selects Python 3.12 with `.python-version` and locks dependencies in `uv.lock`. After cloning, run these commands from the repository root:

```sh
uv sync --locked
uv run --locked barq
```

The default command runs a small audit. Useful explicit commands:

```sh
uv run --locked barq audit --limit 100
uv run --locked barq audit --limit 100 --dataset mix
uv run --locked barq prepare
uv run --locked barq prepare --workers 4
```

`--dataset mix` or `--dataset aisa` selects one dataset. Both commands accept `--config configs/data.yaml` and an optional `--output PATH` for the workspace root; the default is the current working directory. This root contains the raw cache, processed data and reports. Benchmark paths still resolve relative to the config file.

## Audit and prepare

**Audit** requests a fixed budget of rows per configured split from the Hugging Face Dataset Viewer, using deterministic offsets spread through the split. `--limit 100` means 100 rows per split, not 100 per source. It checks the response's `x-revision` against the pinned revision and never falls back to downloading the complete dataset. A mismatched or unavailable viewer response stops the audit.

Audit samples are not representative source-level measurements. Rare sources can be missing, and duplicate/conflict checks cover only the rows inspected. The report shows observed coverage; it does not estimate full-dataset defect rates.

**Prepare** downloads the pinned Parquet files into the local raw cache, then processes them in batches. It uses SQLite for duplicate/group tracking and writes Parquet batches rather than keeping the dataset in memory. Allow more than 2 GB of disk space for raw data, plus space for indexes and derived outputs.

Local Parquet files are read directly through PyArrow. AISA's unused preformatted `text` and full `tools` registry are skipped during reading; the original files still contain them. Downloads already run concurrently across files.

`prepare --workers 4` uses four processes for adaptation, validation, hashing and JSON encoding. The default is one process, which avoids startup and communication costs on small inputs. Work is queued in bounded batches; more workers use more memory. Duplicate decisions, split assignments, sampling and output writing remain ordered in the parent process, so worker count does not change the resulting rows. No GPU or distributed framework is needed.

The terminal prints rows/second, and `audit.md` includes per-split processing timings. `manifest.json` also records worker count, download timings and total elapsed time. Compare these on your VM: extra workers cannot speed up network or disk bottlenecks, and may slow small runs.

Each command creates a new run directory and processes its inputs afresh. Cached raw downloads are reused; processing does not resume from earlier runs. Existing outputs are not overwritten or automatically deleted.

Only consume a run when its `manifest.json` says `"status": "complete"`. Interrupted or failed runs may contain partial artifacts.

## What Phase 0 does

- Checks schemas and applies conservative, task-specific cleaning while preserving Arabic spelling, dialects, diacritics and intentional errors in correction prompts.
- Preserves source/task labels and structured tool calls. AISA's Gemma-formatted `text` is not used as a ready-made training conversation.
- Removes exact duplicates and groups identical inputs into the same split. Different targets are retained: multiple creative or dialect responses can be valid. Reviewing contradictory answers is a later semantic check.
- Checks supplied benchmark references using NFC and whitespace-normalized exact matching.
- Creates a deterministic grouped validation split for the mix. AISA's official dev/test splits remain held out and take priority when checking training overlap.
- Reports counts, exclusions and small review samples. It does not balance sources, repair factual answers automatically or certify data quality.

Tool checks cover function names, argument JSON, nested types, required fields, enums and additional properties. They do not implement every JSON Schema constraint or execute tools to verify outcomes.

AISA's public test split is **unlabeled**. Its empty assistant placeholders are exported as holdout inputs, not training targets or no-call examples. Official test scoring requires the withheld labels or an external evaluator.

Overlap protection covers the selected datasets and inspected rows only. For a combined final preparation, run `prepare` with both datasets, without `--dataset`.

## Review prepared data (Phase 0.1)

After preparation finishes, run this on the machine holding its outputs:

```sh
uv run --locked barq review --input data/processed/prepare/20260904T122021Z-d8a1e6ee --per-group 100
```

Replace the run ID with your own. Review reads `candidates.parquet` and `decisions.parquet`, and requires the original prepare manifest to report `"status": "complete"`. It finds that manifest under the corresponding `reports/prepare/<run-id>/` directory. If you moved the files elsewhere, retain the original run folder's name and pass `--manifest PATH` to identify the original manifest; review verifies the run ID. `--output PATH` selects another output workspace root. By default review uses the root containing `reports/prepare/<run-id>/manifest.json`, or the current directory if the manifest was moved outside that layout.

Review scans the existing data without downloading datasets or calling models. It preserves every candidate and split, and selects **training rows only** for manual inspection. Validation and test rows remain held out.

- `--per-group 100` samples up to 100 training examples per source/task/review-hint/dialect/tool-behavior group. AISA groups distinguish calls from no-call answers so rare no-call examples are visible. Simple creative-writing hints help route review; these hints do not replace original task labels.
- `--flagged-per-group 5` keeps up to five diagnostic examples per source/task/flag in a separate sample. These intentionally selected examples do not estimate a source's overall defect rate.
- `--seed 42` makes sample selection reproducible for the same inputs and settings.

Checks look for changed underlying letters or absent added diacritics in diacritization tasks, unexpected sentiment labels, Python syntax/function issues when applicable, and visible web boilerplate. Python snippets are parsed, never executed. Flags are review hints: valid syntax does not establish correct code, preserved letters do not establish correct diacritics, and unflagged answers may still be wrong. The report includes check coverage and carries forward the preparation run's benchmark status; review does not add new benchmark checks.

Python checks distinguish a requested function from a program that merely mentions functions, and recognize lambda expressions as functions. A parse failure in an unfenced answer receives `python_answer_unparsed`, with a skipped-check count: prose may surround valid code. `python_syntax_invalid` is reserved for failed parsing of explicitly Python-fenced code. Reports created before these corrections retain their original flags; inspect those cases before deciding what to exclude.

Read `reports/review/<run-id>/review.md`, then inspect `review_samples.jsonl` and `flagged_samples.jsonl`. The report also summarizes source/task coverage and preparation decisions. Share those small files and `manifest.json` for review; the large dataset can stay on the VM. Keep the prepared outputs on retained storage before deleting a VM.

This stage does not judge factual correctness with an LLM, rewrite or delete answers, assign an automatic quality score, choose source weights, or produce a final SFT dataset.

## Curate prepared data (Phase 1)

On the VM holding the completed prepare run, update the code and run:

```sh
git pull --ff-only
uv sync --locked
uv run --locked barq curate --input data/processed/prepare/20260904T122021Z-d8a1e6ee
```

Replace the prepare run ID with your own. Curation reuses its files without network or model calls. It assesses **labeled training candidates only**, assigning `accept`, `review`, `repair` or `exclude` with reasons and check results. `accept` means the row passed the implemented offline checks; it is neither semantic approval nor a final training export. `repair` identifies work needed; this command does not rewrite answers.

The default `configs/curation.yaml` holds News Commentary translations and summaries for semantic alignment review, and generic AISA no-call answers for adaptation. Defective structures are excluded; uncertainty signals go to review, including any flag not explicitly routed by the config. Suspected unsupported numeric tool arguments are review hints, not proof of invention or grounds for automatic exclusion. The strict config controls `seed`, `batch_size`, `sample_per_group`, `repair_flags`, `review_sources` and `review_tasks`; use `--config PATH` to supply another config.

Curation adds checks for replacement characters, stored assistant reasoning, and numeric tokens introduced in supported translation, summary, dialect-conversion and correction tasks. Tool checks compare numeric/date/ID values against preceding non-assistant context. Only narrowly recognized standalone writing directives get literal opening/ending/comma checks; other writing constraints remain semantic review work. Unrecognized task wrappers go to review. Stored `think` fields remain in the original intermediate records and are flagged for a supervision decision; they are not approved training targets.

New outputs are written under `data/processed/curate/<run-id>/`: `candidates.parquet` contains accepted training rows, while `decisions.parquet` records every assessed training candidate, its decision, reasons, checks and source/task/dialect/hint metadata. Original preparation decisions, validation rows and test inputs remain in the original prepare outputs. No source balancing or split changes occur.

Read `reports/curate/<run-id>/curation.md` and `manifest.json` first. `review_samples.jsonl` supplies bounded samples per source/task/hint/dialect/tool-behavior/decision group (three per group by default). Share the report and manifest, then samples if needed. Use `--manifest PATH` for moved inputs, retaining the original prepare run folder's name; `--output PATH` selects an output workspace root.

Full semantic verification, calibrated model judging, benchmark checks, actual repairs and SFT export still come later. Reports and data are ignored by Git and persist only on the storage holding them: retain or back up the VM storage before deleting it.

## Persistent Modal run

For Modal, launch a background job from your own computer rather than keeping the pipeline in a temporary shell. With the authenticated Modal CLI installed (tested with `1.5.2`), run from the repository root:

```sh
git pull --ff-only
modal run --detach scripts/modal_pipeline.py
```

This optional wrapper requests four CPUs and 16 GiB memory, with a four-hour function timeout. It runs `prepare` and then `curate` on Modal. The full dataset is downloaded there; only source/configuration files are uploaded from your computer. The dependencies come from `uv.lock`; Modal is not added to the core project's dependencies. This launch uses your Modal compute and storage account.

All raw downloads, prepared data, curation outputs, reports and the Hugging Face cache live on the named **`barq-data` Volume**, mounted at `/barq`. Completed stages are explicitly committed. Rerunning the command reuses compatible completed stages; a failed/incomplete stage starts a new run using retained downloads. It does not resume partway through individual rows or recover files from an already-dead shell's unmounted disk.

The local entrypoint uses `.spawn()` and `--detach` so the submitted job can outlive the launching terminal. Wait for the printed submission receipt before closing it. The receipt is saved under `reports/modal-launch/` and includes the app ID and log command. Run one job at a time against this Volume; a function's container limit does not prevent a second separately launched app from writing to it.

```sh
modal app logs APP_ID -f
modal volume ls barq-data reports/curate
modal volume get barq-data latest.json -
```

After completion, `latest.json` identifies the saved stage runs and report paths. Download individual reports with `modal volume get barq-data reports/curate/RUN_ID/curation.md ./curation.md`. The Volume survives the compute job, but deleting the Volume deletes that retained copy. Commit behavior and detached jobs are described in [Modal Volumes](https://modal.com/docs/guide/volumes) and [invocation durability](https://modal.com/docs/guide/function-invocation-methods).

## Benchmark references

Every requested benchmark appears in `configs/data.yaml`. A `path: null` means **not checked**. No benchmark is silently downloaded or assumed clean.

To enable a check, supply a local UTF-8 JSONL file containing one record per reference question:

```json
{"prompt":"نص السؤال الأصلي","variants":["صياغة أخرى معروفة للسؤال"]}
```

`variants` is optional. Paths resolve relative to the config file's directory, for example `../data/benchmarks/arabicmmlu.jsonl` from `configs/data.yaml`. A configured path that does not exist fails the run.

Use the exact intended benchmark version and record its provenance with the reference file. Matching does not discover translations, paraphrases or fuzzy duplicates; a passing check is not a claim of contamination-free data. The run manifest records check coverage.

## Files

```text
configs/data.yaml                    # Pinned inputs and processing settings
configs/curation.yaml                # Offline curation routing and sample settings
src/barq/data.py                     # Loading, CLI, processing and reports
src/barq/rules.py                    # Validation, cleaning and fingerprints
src/barq/review.py                   # Offline review sampling and reports
src/barq/curate.py                   # Offline training-candidate decisions and exports
src/barq/persistent.py               # Completed-stage reuse and persistence callbacks
scripts/modal_pipeline.py            # Optional detached Modal job and named Volume
tests/test_rules.py                  # Arabic and tool-preservation checks
tests/test_data.py                   # Split, benchmark and export integration checks
tests/test_review.py                # Review sampling and input integrity checks
tests/test_quality.py               # Task-specific review hints
tests/test_curate.py                # Offline routing, evaluation isolation and safe failure
tests/test_curation_rules.py        # Grounding signals and conservative constraint checks
data/raw/<dataset>/<revision>/       # Cached pinned source files
data/processed/{audit,prepare}/<run-id>/
  candidates.parquet                # Labeled candidate examples
  decisions.parquet                 # Processing decisions and reasons
  holdout.parquet                   # Unlabeled evaluation inputs
  index.sqlite3                     # Disk-backed duplicate/group index
reports/{audit,prepare}/<run-id>/
  audit.md                          # Counts, coverage and limitations
  manifest.json                     # Inputs, settings and reproducibility data
  samples.jsonl                     # Small review samples
reports/review/<run-id>/
  review.md                         # Source/task findings and check coverage
  manifest.json                     # Input provenance, settings and run status
  review_samples.jsonl              # Training examples for manual quality review
  flagged_samples.jsonl             # Targeted diagnostic training examples
  flags.parquet                     # Machine-readable review flags
data/processed/curate/<run-id>/
  candidates.parquet                # Accepted training rows; offline checks only
  decisions.parquet                 # All assessed training rows and routing reasons
reports/curate/<run-id>/
  curation.md                       # Decisions, coverage and remaining review work
  manifest.json                     # Inputs, configuration and run status
  review_samples.jsonl              # Bounded examples from each decision group
```

Run IDs include UTC time and a random suffix. Candidate records retain IDs, source/task/split metadata, original split, revision, row index, labeling status and hashes. Conversations, tools and additional metadata are serialized in `messages_json`, `tools_json` and `metadata_json` columns. These are intermediate data exports; model-specific training rendering comes later.

Raw data and generated outputs are excluded from Git. The local commands do not upload them. The optional Modal runner uploads its source/configuration and writes generated data to your Modal Volume.

## Verify

```sh
uv run --locked python -m unittest discover -s tests -v
```

## Later phases

Keep Phase 0 outputs as the input contract. Use review samples and Phase 1 curation decisions to calibrate a teacher judge against manual reviews, configure benchmark references, and decide source weights and targeted repairs. Offline acceptance alone is insufficient for training. Teacher review and generation are future work; DataTrove can be added if near-duplicate detection warrants it. Then add `sft.py` and `evaluate.py` for Tinker SFT and baseline comparisons, and `rl.py` and `rewards.py` when the environments and reward checks are ready. No training renderer or RL prompts are generated yet.
