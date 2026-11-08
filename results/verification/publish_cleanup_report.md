# ACPDAD repository cleanup verification

All data and results discussed here are **simulated**. The frozen final test was neither regenerated nor rescored; no model was retrained and no threshold was changed.

## Structure and changes

The repository now has one launcher (`run_project.py`) and one active package (`src/acpdad/`). `core.py` moved byte-for-byte to `simulation.py`; `experiment.py` became `workflow.py`; `protocol.py` became `evaluation_protocol.py`. The latter two and the launcher were edited only for package imports, relocated defaults, source-path references, and safe reproduction calibration bookkeeping. `config/`, `data/train_validation/`, `data/validation/`, `data/frozen_test/`, `models/selected/`, and `results/frozen_test/` hold the active configuration and evidence.

Removed from the publishable tree: inactive V4–V5.4R1 launchers and implementations, obsolete datasets/checkpoints not needed by the selected workflow, notebooks and benchmark builders, duplicate reports and status notes, research-figure build dependencies, temporary outputs, caches, old migration notes, and in-tree backup environments. The full pre-cleanup state is recoverable from the archive named in `docs/relocation_manifest.json`. Earlier V5.4R1 and V5.3 **result** evidence remains under `results/historical/`; their old code and data remain in the recovery archive. The active results and frozen report remain unchanged.

The historical result directories now use purpose names: `early_takeover_evaluation/` contains the earlier V5.3 result; `pre_fallback_evaluation/` contains the V5.4R1 result before the current fallback path. The original report, locks, and prediction files inside them retain their names and bytes. The rename-specific archive and path map are recorded in `docs/relocation_manifest.json`.

## Integrity

- Selected checkpoint SHA-256: `cdbe7a1b7fba8823c3c17ca722b34002c0852de446c5e7e0a41ba878a741bc4d` (matches its manifest).
- Frozen test lock SHA-256: `66f905181503db16f8e6610d86d6e0b2c4820b71da6e1be276b8693edd000965`.
- Frozen test specification SHA-256: `c1cca3555e2322685140b96362a6329d760b0e1c7737ab8a760f21bcd7cc8e70`.
- The relocation manifest checks 231 retained dataset, checkpoint, source-provenance, and frozen-result files as byte-identical. It records before/after source hashes separately and leaves historical locks unchanged.
- A follow-up audit rehashed all 237 files marked byte-identical in that manifest (including unchanged code and dependency files); zero mismatches were found (`results/verification/followup_hash_audit.json`).
- All 99 supplied checksums across the 11 retained simulated datasets match; see `results/verification/integrity.json`.
- Existing `.venv` contains the pinned versions in `requirements.txt`: NumPy 2.5.4, pandas 2.3.3, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2.

## Execution checks

| Check | Result | Evidence |
| --- | --- | --- |
| Main `verify` from cleaned root and final edited source | PASS: 22 base, 88 temporal, checkpoint digest, fallback switching, gap reset, out-of-order rejection, run isolation | `results/verification/verify_final_source.log` |
| Complete validation replay, one Evil_Twin mission | PASS: 300 scheduled, 257 timely predictions, dashboard generated | `results/replays/validation_001/`, `results/verification/validation_replay.log` |
| Selected-run inference | PASS: 257 rows for `v5_4r2_val_3_000` | `results/verification/selected_validation.csv` |
| All-run inference | PASS: 60 manifest runs, 16,814 rows, exact delivered-packet mapping and per-run arrival order; no label/metadata columns | `results/verification/inference_mapping.json`, `all_validation_condition.csv` |
| Unknown run ID | PASS: nonzero exit with available IDs; no output CSV created | `results/verification/unknown_run.log` |
| Original versus reorganized predictions | PASS: all 257 classifications, model paths, alert flags, mapping, confidence and probabilities match exactly | `results/verification/prediction_equivalence.json`; baseline ledger retained as `original_validation_baseline.csv` |
| Validation/final config references | PASS: both protocol render commands locate unchanged condition files | `results/verification/protocol_render_check.log`, `frozen_config_path_check.log` |
| README links and active paths | PASS: all five document links and 12 command resources exist; no active hardcoded absolute paths | `results/verification/reference_check.json` |
| Dashboard assets | PASS: 37 local links resolve and 20 images decode across seven pages | `results/verification/dashboard_assets.json` |
| Full `results/` audit | PASS for active and historical dashboards: 20 HTML pages checked, with no missing links outside the preserved `generated_once` archive | `results/verification/results_folder_audit.json`, `results/README.md` |
| Historical directory rename | PASS: all 65 historical files retain their original SHA-256 hashes; both locks and all non-archival dashboard links resolve | `results/verification/historical_rename_check.json` |
| Browser rendering | NOT VERIFIED: Chromium exited 133 at crashpad sandbox setup; Firefox exited 139 in headless mode | `results/verification/browser_render.log`, `firefox_render.log` |

Runtime timing columns in two separate prediction runs differ because they are measured during execution; classification outputs and confidences match. The historical frozen lock still describes the original source paths and hashes. The relocation manifest maps those paths to the current tree; no historical record was rewritten.

`results/` contains 105 frozen-test files, 65 historical-result files, ten protocol configurations, seven selection records, 34 small original execution logs, the validation replay, and cleanup verification evidence. The `analysis/generated_once/` archive has 22 unresolved local references in its copied HTML pages because it preserves only the report-era files needed to document the correction. This is pre-existing archival incompleteness, not a failure of the current or historical dashboards; the other HTML pages have no missing local references. Do not use the archive's HTML as the published dashboard.

## Known research limits

The frozen simulated test met pooled end-to-end Evil_Twin recall (84.76%) and pooled precision (98.09%) targets, but unseen-AP/benign-roaming precision was 93.75%, benign missions with alerts were 36%, and false alert episodes were 0.79/min. DoS scored recall was 40.65%. The one-mission validation replay is an execution check, not an independent accuracy estimate. The project has no real-drone, SITL, captured-Wi-Fi, or field validation. A publication license has not been selected.

## Recovery

See `docs/relocation_manifest.json` for the full old-to-new mapping, source hashes, archive SHA-256, and rollback instructions. The recovery archive is under `/tmp`, which can be temporary storage; copy it to durable storage before clearing `/tmp`.
