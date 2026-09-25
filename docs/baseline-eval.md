# Nemotron Arabic MCQ baseline

The first baseline covers ArabicMMLU (`All` config only), AlGhafa's native Arabic EXAMS task (`mcq_exams_test_ar`), and the OALL Arabic EXAMS test set. It uses official `test` splits only; ArabicMMLU `dev`, AlGhafa `validation`, and Arabic EXAMS `validation` are not sampled. ArabicMMLU's `All` config aggregates its subjects, so subject configs are deliberately excluded from the evaluation input. Exact repeated prompts within a benchmark are deduplicated; all rows for a repeated prompt with conflicting labels are excluded. The two EXAMS releases are reported separately because their split definitions differ and may overlap.

Every dataset repository SHA is resolved before streaming and stored in `manifest.json`. Rows are loaded through Hugging Face streaming on Modal; benchmark examples and predictions stay in the `barq-data` Modal volume. A deterministic hash-ranked sample of 100 examples per benchmark is the default. The full official test splits can be selected later by setting a larger limit after cost and run time are reviewed.

The fixed Arabic instruction asks for a single A–E option letter. The report scores strict answer-letter accuracy, counts unparseable outputs as incorrect, and includes ArabicMMLU subject-level results. This is a Barq zero-shot Arabic-prompt result, not a reproduction of every leaderboard's prompt protocol. No validation or few-shot rows are used.

## Dry run

Run the synthetic fixtures locally. It checks all three source schemas, Arabic prompt construction, correct-label mapping, reasoning-aware answer extraction, fail-closed ambiguous extraction, and the report schema. It makes no network or model calls and reads no datasets:

```powershell
uv run --locked python -m barq.baseline_eval --dry-run
```

The report is written under `reports/baseline-eval/dry-run.json` (ignored by Git).

## Remote source preflight

This fetches and streams only the official `test` splits on Modal, checks real source columns/labels and generated prompts, and saves aggregate counts and resolved revisions to the `barq-data` volume. It makes **zero Tinker calls** and returns no dataset rows locally:

```powershell
uv run --locked --group modal modal run --detach scripts/modal_baseline_preflight.py --limit-per-benchmark 100 --run-id baseline-preflight-20260925
```

The launcher prints a Modal app ID, call ID, and preflight run ID. The aggregate manifest is `/barq/reports/baseline-eval/source-preflight/<run-id>/manifest.json` in the volume. Use `modal app logs <app-id> -f` to follow the bounded job.

The completed remote preflight at `/barq/reports/baseline-eval/source-preflight/baseline-preflight-20260925/manifest.json` found 14,455 ArabicMMLU test rows: 29 exact repeats removed and one conflicting prompt group (2 rows) excluded, leaving 14,424 eligible questions. AlGhafa had 557 eligible test rows; OALL Arabic EXAMS had 537. The deterministic 100-row sample from each source had zero malformed rows and 300 valid prompt/answer checks. Dataset revisions and row counts are pinned in that remote manifest. No dataset rows were downloaded locally.

## Live run prerequisites

Tinker credentials are not currently configured. In the Modal dashboard, create a secret named `tinker-secret` with a `TINKER_API_KEY` key. Keep the key in the secret; do not paste it into a command or file. Then the limited live command is:

```powershell
uv run --locked --group modal modal run --detach scripts/modal_baseline_eval.py --live --limit-per-benchmark 100
```

Create the secret in the Modal dashboard before running this command; do not commit the key or place it in a file. To inspect status and logs, use the app ID and call ID printed by the launcher and `modal app logs <app-id> -f`. Output is written to `/barq/reports/baseline-eval/<run-id>/manifest.json` and `predictions.jsonl` in the `barq-data` volume.

The live path pins the BF16 Tinker base model `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` and cookbook renderer `nemotron3_ultra`. It uses temperature 0, one completion, and a 128-token maximum. It uses Tinker's SamplingClient, not a local copy of the model weights. The launcher prints the run ID. If Modal stops the job, rerun with that same `--run-id <id>` to resume from the last committed checkpoint. Results and progress commit every 25 completions; an interruption before a checkpoint can repeat up to 25 calls.

At the current listed sampling prices of $0.195/M prefill tokens and $0.495/M generated tokens, a full 15,518-row run at an assumed 250 prompt tokens and the 128-token completion cap would be about **$1.74**. Actual generated tokens are usually below the cap; long prompts raise prefill usage. The report records actual prompt/completion token counts and recomputes estimated cost using the prices above. Modal compute is billed separately. Recheck current Tinker pricing before a full run.

## References

- [Tinker model list and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- [Tinker Cookbook renderer factory](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/renderers/get_renderer/)
- [Tinker SamplingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/)
- [ArabicMMLU dataset and task configs](https://huggingface.co/datasets/MBZUAI/ArabicMMLU)
- [AlGhafa native Arabic benchmark](https://huggingface.co/datasets/OALL/AlGhafa-Arabic-LLM-Benchmark-Native)
- [OALL Arabic EXAMS](https://huggingface.co/datasets/OALL/Arabic_EXAMS)
