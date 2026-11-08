# Results directory guide

All results are from simulated observations. The selected checkpoint and frozen test were not rerun during repository cleanup.

| Folder | Contents and purpose |
| --- | --- |
| `frozen_test/` | One-pass final evidence: locks, per-condition and pooled predictions, full observation audits, metrics, plots, dashboard, and report. Open `frozen_test/index.html`. |
| `frozen_test/analysis/generated_once/` | Byte-preserved report outputs created before a report-only correction. This archive is for provenance; its copied HTML is incomplete and is not the dashboard to open. Use the corrected dashboard in `frozen_test/`. |
| `frozen_test/provenance/` | Original report-correction source. Its paths refer to the implementation used during scoring. Do not run it against frozen results. |
| `historical/early_takeover_evaluation/` | Earlier takeover evaluation with its saved predictions, metrics, report, and locks. |
| `historical/pre_fallback_evaluation/` | Earlier detector evaluation before short-history fallback and explicit gap accounting. |
| `selection/` | Training and validation records for model, threshold, fallback, and alert-policy selection. |
| `protocol_configs/` | Saved condition configurations for validation and frozen evaluation. |
| `logs/` | Execution records that support the frozen evaluation provenance. |
| `replays/` | Fresh validation replay outputs. These are execution checks, not new test results. |
| `verification/` | Cleanup and runtime verification records, prediction-equivalence ledgers, and audit reports. |

The frozen report and checkpoint manifest retain the source paths recorded at scoring time. The [relocation manifest](../docs/relocation_manifest.json) maps those paths to the current project structure without rewriting the historical records.
