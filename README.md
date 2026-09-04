# Barq — Arabic data preparation

Phase 0 prepares and audits Arabic chat and tool-use data before training. It uses pinned copies of [arabic-sft-mix-2](https://huggingface.co/datasets/Mohaddz/arabic-sft-mix-2) and [AISA-ArabicFC](https://huggingface.co/datasets/TuwaiqAcademy/AISA-ArabicFC).

No training, RL, model judging, paid API calls or credentials are required. Internet access is needed to fetch the public data.

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
src/barq/data.py                     # Loading, CLI, processing and reports
src/barq/rules.py                    # Validation, cleaning and fingerprints
tests/test_rules.py                  # Arabic and tool-preservation checks
tests/test_data.py                   # Split, benchmark and export integration checks
data/raw/<dataset>/<revision>/       # Cached pinned source files
data/processed/<mode>/<run-id>/
  candidates.parquet                # Labeled candidate examples
  decisions.parquet                 # Processing decisions and reasons
  holdout.parquet                   # Unlabeled evaluation inputs
  index.sqlite3                     # Disk-backed duplicate/group index
reports/<mode>/<run-id>/
  audit.md                          # Counts, coverage and limitations
  manifest.json                     # Inputs, settings and reproducibility data
  samples.jsonl                     # Small review samples
```

Run IDs include UTC time and a random suffix. Candidate records retain IDs, source/task/split metadata, original split, revision, row index, labeling status and hashes. Conversations, tools and additional metadata are serialized in `messages_json`, `tools_json` and `metadata_json` columns. These are intermediate data exports; model-specific training rendering comes later.

Raw data and generated outputs are excluded from Git. Nothing is uploaded or published.

## Verify

```sh
uv run --locked python -m unittest discover -s tests -v
```

## Later phases

Keep Phase 0 outputs as the input contract. Add `review.py` for selective teacher review, repair and gap filling (and DataTrove if near-duplicate detection warrants it), then `sft.py` and `evaluate.py` for Tinker SFT and baseline comparisons. Add `rl.py` and `rewards.py` only when the environments and reward checks are ready. No training renderer or RL prompts are generated in Phase 0.
