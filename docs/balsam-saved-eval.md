# Saved BALSAM v2 evaluation

The saved `dev(6).zip` through `dev(9).zip` archives are evaluated as the separately labeled `balsam_v2_saved_dev_subset` benchmark. The run uses Barq's custom Tinker inference path and the Nemotron model configured in `barq.baseline_eval`; it does not use `lm-evaluation-harness` to execute these rows.

The Modal preflight parsed 654 rows and compared their prompts and references with the current SFT candidate. One exact overlap was excluded, leaving 653 rows: 645 Machine Translation/BLEU, 2 Sentence Composition/ROUGE, 2 Text Completion/ROUGE, 2 Text Completion/accuracy, and 2 Question Decomposition/ROUGE. No Tinker calls were made during preflight. The aggregate report is stored at `/barq/reports/balsam-saved-eval/preflight.json` on Modal volume `barq-data`.

From the Barq repository root in PowerShell, stage the four ZIPs, then rerun the credential-free preflight after changing the data or SFT candidate:

```powershell
.\scripts\stage_balsam_saved_zips.ps1
modal run scripts/modal_balsam_saved_preflight.py
```

The live run requires Modal secret `tinker-secret` with `TINKER_API_KEY` set. It writes row-level predictions and the manifest only to the Modal volume. Use a unique run ID; the same ID resumes from durable progress after interruption:

```powershell
modal run scripts/modal_balsam_saved_eval.py --live --run-id balsam-dev-20260925
```

Scores use local BLEU, ROUGE, and fuzzy-accuracy approximations based on the vendored KSAA metric setup. They are not exact KSAA evaluator scores: this path does not call KSAA's `evaluate` or RapidFuzz implementations. Translation prompts preserve the source task's target-language direction.
