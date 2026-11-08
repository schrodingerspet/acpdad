"""Correct the scored-row Evil_Twin recall denominator from frozen ledgers.

This utility changes derived report fields only. It never loads the model or
reruns inference; the one-pass frozen evaluation remains the sole prediction
run. Original generated-once report artifacts are copied into analysis/
generated_once before correction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "results/acpdad_research_v5_4r2/final_test"
ANALYSIS = FINAL / "analysis"
GENERATED = ANALYSIS / "generated_once"
CONDITIONS = (
    "mission_initial_state",
    "wind_background_traffic",
    "attack_timing_duration_intensity",
    "replay_delay_source_segment",
    "unseen_ap_benign_roaming",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def timely_recall(path: Path) -> tuple[float, int, int, int]:
    # Evaluation-only audit: labels are joined after inference in this ledger.
    import pandas as pd

    audit = pd.read_csv(path)
    truth = audit["Target_Label"].astype(str).eq("Evil_Twin")
    timely = audit["timely_prediction"].astype(bool)
    tp = int((truth & timely & audit["prediction"].astype(str).eq("Evil_Twin")).sum())
    scored = int((truth & timely).sum())
    scheduled = int(truth.sum())
    return tp / max(1, scored), tp, scored, scheduled


def clustered_mean_interval(values, *, seed: int, replicates: int = 10000) -> list[float]:
    import numpy as np

    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(replicates, len(a)))
    means = a[indices].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def alert_uncertainty(rows, *, seed_offset: int) -> dict:
    import pandas as pd

    benign = rows[rows.scenario.eq("Benign")]
    evil = rows[rows.scenario.eq("Evil_Twin")]
    benign_minutes = (benign.scheduled_rows.astype(float) / 5.0 / 60.0).to_numpy()
    episodes = benign.false_alert_episodes.astype(float).to_numpy()
    episode_rate = episodes / benign_minutes
    return {
        "benign_mission_alert_rate_run_clustered_95_ci": clustered_mean_interval(
            benign.false_alert_mission.astype(float), seed=5100 + seed_offset),
        "false_alert_episodes_per_benign_minute_run_clustered_95_ci": clustered_mean_interval(
            episode_rate, seed=5200 + seed_offset),
        "evil_twin_pre_attack_alert_rate_run_clustered_95_ci": clustered_mean_interval(
            evil.pre_attack_attack_alert.astype(float), seed=5300 + seed_offset),
    }


def main() -> None:
    sys.path.insert(0, str(ROOT / "simulation/research_v5_4r2"))
    from experiment import score_ledger

    execution_path = FINAL / "evaluation_execution.json"
    execution = read_json(execution_path)
    if execution.get("status") != "completed" or execution.get("scored_once") is not True:
        raise RuntimeError("Refusing to correct reports before the frozen one-pass evaluation completes")

    report = FINAL / "IMPLEMENTATION_REPORT.md"
    dashboard = FINAL / "index.html"
    metrics_path = ANALYSIS / "metrics.json"
    condition_csv = ANALYSIS / "per_condition_metrics.csv"
    cross_csv = ANALYSIS / "cross_verification.csv"
    mutable = [report, dashboard, metrics_path, condition_csv, cross_csv, ANALYSIS / "per_mission_metrics.csv"]
    for name in CONDITIONS:
        mutable.extend([
            FINAL / "evaluations" / name / "metrics.json",
            FINAL / "evaluations" / name / "index.html",
        ])

    # Keep byte-identical originals for the hashes recorded by the one-pass evaluator.
    for path in mutable:
        if not path.exists():
            raise FileNotFoundError(path)
        relative = path.relative_to(FINAL)
        archived = GENERATED / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        if not archived.exists():
            shutil.copy2(path, archived)

    raw_hash_checks = {
        "aggregate_metrics": (metrics_path, execution["aggregate_metrics_sha256"]),
        "implementation_report": (report, execution["report_sha256"]),
        "dashboard": (dashboard, execution["dashboard_sha256"]),
    }
    for condition, result in execution["condition_results"].items():
        if condition in CONDITIONS:
            raw_hash_checks[f"{condition}_metrics"] = (
                FINAL / "evaluations" / condition / "metrics.json",
                result["metrics_sha256"],
            )
    for label, (path, expected) in raw_hash_checks.items():
        archived = GENERATED / path.relative_to(FINAL)
        actual = sha256(archived)
        if actual != expected:
            raise RuntimeError(f"Generated-once artifact hash mismatch for {label}: {actual} != {expected}")

    import pandas as pd

    recalls: dict[str, dict] = {}
    corrected_missions = []
    for condition in CONDITIONS:
        audit_path = FINAL / "evaluations" / condition / "observation_audit.csv"
        audit = pd.read_csv(audit_path)
        audit.attrs["sampling_hz"] = 5.0
        audit.attrs["duration_s"] = 60.0
        episode_path = FINAL / "evaluations" / condition / "attack_alert_episodes.csv"
        episodes = pd.read_csv(episode_path).to_dict("records") if episode_path.exists() else []
        recomputed = score_ledger(audit, episodes)
        for mission in recomputed["per_mission"]:
            mission["condition"] = condition
            corrected_missions.append(mission)
        recall, tp, scored, scheduled = timely_recall(audit_path)
        recalls[condition] = {"conditional_evil_twin_recall": recall, "tp": tp, "timely_evil_twin_rows": scored, "scheduled_evil_twin_rows": scheduled}
        path = FINAL / "evaluations" / condition / "metrics.json"
        current = read_json(path)
        # Assert this is correcting only the reporting field, not its underlying class metrics.
        class_recall = current["row_level"]["per_class"]["Evil_Twin"]["recall"]
        if abs(class_recall - recall) > 1e-12:
            raise RuntimeError(f"Saved class metric and audit ledger disagree for {condition}: {class_recall} != {recall}")
        if abs(recomputed["conditional_evil_twin_recall"] - recall) > 1e-12:
            raise RuntimeError(f"Corrected scorer and direct audit recall disagree for {condition}")
        current["conditional_evil_twin_recall"] = recall
        write_json(path, current)

    mission_metrics = pd.DataFrame(corrected_missions)
    archived_missions_path = GENERATED / "analysis/per_mission_metrics.csv"
    per_mission_support_unchanged = False
    if archived_missions_path.exists():
        original_missions = pd.read_csv(archived_missions_path)
        old_support = original_missions.set_index("run_id")["evil_twin_scored_support"].sort_index()
        new_support = mission_metrics.set_index("run_id")["evil_twin_scored_support"].sort_index()
        if not old_support.equals(new_support):
            raise RuntimeError("Run-level scored Evil_Twin support changed while rebuilding the report")
        per_mission_support_unchanged = True
    mission_metrics.to_csv(ANALYSIS / "per_mission_metrics.csv", index=False)
    alert_intervals = {"pooled": alert_uncertainty(mission_metrics, seed_offset=0)}
    for condition in CONDITIONS:
        alert_intervals[condition] = alert_uncertainty(
            mission_metrics[mission_metrics.condition.eq(condition)],
            seed_offset=CONDITIONS.index(condition) + 1,
        )
        path = FINAL / "evaluations" / condition / "metrics.json"
        current = read_json(path)
        current.update(alert_intervals[condition])
        write_json(path, current)

    pooled_recall, pooled_tp, pooled_scored, pooled_scheduled = timely_recall(ANALYSIS / "all_observation_audit.csv")
    aggregate = read_json(metrics_path)
    aggregate_class_recall = aggregate["row_level"]["per_class"]["Evil_Twin"]["recall"]
    if abs(aggregate_class_recall - pooled_recall) > 1e-12:
        raise RuntimeError(f"Saved pooled class metric and audit ledger disagree: {aggregate_class_recall} != {pooled_recall}")
    aggregate["conditional_evil_twin_recall"] = pooled_recall
    for condition, values in recalls.items():
        aggregate["per_condition"][condition]["conditional_evil_twin_recall"] = values["conditional_evil_twin_recall"]
        aggregate["per_condition"][condition].update(alert_intervals[condition])
    aggregate.update(alert_intervals["pooled"])
    ev_missions = mission_metrics[mission_metrics.scenario.eq("Evil_Twin")]
    aggregate["evil_twin_missions_at_least_60pct_end_to_end_recall"] = int(
        (ev_missions.evil_twin_end_to_end_recall >= .60).sum())
    aggregate["evil_twin_missions_count"] = int(len(ev_missions))
    aggregate["post_evaluation_reporting_correction"] = {
        "correction": "Fixed pooled conditional-recall denominator shadowing; verified run-local scored-support values against the saved audit and original report ledger.",
        "formula": "Evil_Twin true positives among timely predictions / scheduled Evil_Twin rows with timely predictions.",
        "predictions_rerun": False,
        "labels_or_predictions_modified": False,
        "per_mission_metrics_reconstructed_from_run_mapped_audit": True,
        "per_mission_support_values_verified_unchanged": per_mission_support_unchanged,
        "generated_once_artifacts_preserved_at": "analysis/generated_once/",
    }
    write_json(metrics_path, aggregate)

    # Keep the CSV in sync with the corrected ledger-derived conditional metric.
    rows = list(csv.DictReader(condition_csv.open(encoding="utf-8", newline="")))
    for row in rows:
        if row.get("condition") in recalls:
            row["conditional_evil_twin_recall"] = f"{recalls[row['condition']]['conditional_evil_twin_recall']:.12g}"
    with condition_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    correction_note = (
        "**Reporting erratum (post-evaluation):** A report-arithmetic defect in `simulation/research_v5_4r2/experiment.py` reused a per-mission denominator when formatting pooled conditional Evil_Twin recall, printing TP counts as percentages. Conditional recall was recomputed from the saved run-mapped observation audits and cross-checked against the five-class recall. Per-mission scored-support values were also checked against the audit and matched the original run-level values. No model inference was rerun and no prediction, label, threshold, or test condition changed. Byte-identical generated-once report artifacts and their original hashes are preserved in `analysis/generated_once/` and `evaluation_execution.json`. Recreate the derived report values with `.venv/bin/python simulation/research_v5_4r2/reporting_correction.py`; it does not load the model or rerun inference."
    )
    report_text = report.read_text(encoding="utf-8")
    report_text = re.sub(
        r"Conditional Evil_Twin recall on timely scored rows: [0-9.]+%",
        f"Conditional Evil_Twin recall on timely scored rows: {pooled_recall:.2%}",
        report_text,
    )
    for condition, values in recalls.items():
        prefix = f"| {condition} |"
        new_lines = []
        for line in report_text.splitlines():
            if line.startswith(prefix):
                cells = line.split("|")
                cells[3] = f" {values['conditional_evil_twin_recall']:.2%} "
                line = "|".join(cells)
            new_lines.append(line)
        report_text = "\n".join(new_lines)
    note_pattern = r"\*\*Reporting erratum \(post-evaluation\):\*\*.*?(?=\n\n|\Z)"
    if re.search(note_pattern, report_text):
        report_text = re.sub(note_pattern, correction_note, report_text, count=1)
    else:
        anchor = "## Integrity, verification and limitations\n"
        if anchor not in report_text:
            raise RuntimeError("Could not find report integrity section for correction note")
        report_text = report_text.replace(anchor, anchor + "\n" + correction_note + "\n", 1)
    alert_anchor = "### Five-class scored metrics\n"
    alert_rows = []
    for scope, mission_rows in [("pooled", mission_metrics)]:
        values = alert_intervals[scope]
        benign_rows = mission_rows[mission_rows.scenario.eq("Benign")]
        evil_rows = mission_rows[mission_rows.scenario.eq("Evil_Twin")]
        alert_rows.append((
            scope,
            int(benign_rows.false_alert_mission.sum()), len(benign_rows),
            float(benign_rows.false_alert_episodes.sum()) / max(1e-12, float((benign_rows.scheduled_rows / 5.0 / 60.0).sum())),
            int(evil_rows.pre_attack_attack_alert.sum()), len(evil_rows),
            values["benign_mission_alert_rate_run_clustered_95_ci"],
            values["false_alert_episodes_per_benign_minute_run_clustered_95_ci"],
            values["evil_twin_pre_attack_alert_rate_run_clustered_95_ci"],
        ))
    for condition in CONDITIONS:
        subset = mission_metrics[mission_metrics.condition.eq(condition)]
        values = alert_intervals[condition]
        benign_rows = subset[subset.scenario.eq("Benign")]
        evil_rows = subset[subset.scenario.eq("Evil_Twin")]
        alert_rows.append((
            condition,
            int(benign_rows.false_alert_mission.sum()), len(benign_rows),
            float(benign_rows.false_alert_episodes.sum()) / max(1e-12, float((benign_rows.scheduled_rows / 5.0 / 60.0).sum())),
            int(evil_rows.pre_attack_attack_alert.sum()), len(evil_rows),
            values["benign_mission_alert_rate_run_clustered_95_ci"],
            values["false_alert_episodes_per_benign_minute_run_clustered_95_ci"],
            values["evil_twin_pre_attack_alert_rate_run_clustered_95_ci"],
        ))
    alert_section = [
        "### Run-clustered alert-rate intervals", "",
        "Intervals resample independent mission IDs (10,000 replicates), not correlated observations.", "",
        "| Scope | Benign alert missions | False episodes / benign minute | Evil_Twin pre-attack alert missions |",
        "|---|---:|---:|---:|",
    ]
    for row in alert_rows:
        alert_section.append(
            f"| {row[0]} | {row[1]}/{row[2]} ({row[1]/max(1,row[2]):.1%}; 95% CI {row[6][0]:.1%}–{row[6][1]:.1%}) | {row[3]:.3f} (95% CI {row[7][0]:.3f}–{row[7][1]:.3f}) | {row[4]}/{row[5]} ({row[4]/max(1,row[5]):.1%}; 95% CI {row[8][0]:.1%}–{row[8][1]:.1%}) |"
        )
    alert_section += [""]
    if "### Run-clustered alert-rate intervals" not in report_text:
        if alert_anchor not in report_text:
            raise RuntimeError("Could not find report class-metrics section for alert intervals")
        report_text = report_text.replace(alert_anchor, "\n".join(alert_section) + alert_anchor, 1)
    report.write_text(report_text.rstrip() + "\n", encoding="utf-8")

    dashboard_text = dashboard.read_text(encoding="utf-8")
    dashboard_text = re.sub(
        r"Scored-row conditional recall is [0-9.]+%",
        f"Scored-row conditional recall is {pooled_recall:.1%}",
        dashboard_text,
    )
    dashboard.write_text(dashboard_text, encoding="utf-8")
    for condition, values in recalls.items():
        path = FINAL / "evaluations" / condition / "index.html"
        text = path.read_text(encoding="utf-8")
        text = re.sub(
            r"conditional recall on scored rows: [0-9.]+%",
            f"conditional recall on scored rows: {values['conditional_evil_twin_recall']:.1%}",
            text,
        )
        path.write_text(text, encoding="utf-8")

    # Add an explicit cross-check record without rewriting earlier rows.
    cross = pd.read_csv(cross_csv)
    item = "Conditional Evil_Twin recall recomputed from frozen audit ledgers"
    if not cross["item"].astype(str).eq(item).any():
        cross.loc[len(cross)] = {
            "item": item,
            "status": "PASS",
            "evidence": f"Pooled {pooled_tp}/{pooled_scored}={pooled_recall:.8f}; all five condition rates match saved class recall; {pooled_scheduled} scheduled EV rows checked.",
            "issue": "Original report denominator shadowing corrected after the one-pass evaluation; predictions and labels unchanged.",
        }
    ev_missions = mission_metrics[mission_metrics.scenario.eq("Evil_Twin")]
    support_item = "Per-mission Evil_Twin support counts validated against run-mapped audit"
    old_support_item = "Per-mission Evil_Twin support counts reconstructed from run-mapped audit"
    if cross["item"].astype(str).eq(old_support_item).any():
        cross.loc[cross["item"].astype(str).eq(old_support_item), "item"] = support_item
        cross.loc[cross["item"].astype(str).eq(support_item), "status"] = "PASS"
        cross.loc[cross["item"].astype(str).eq(support_item), "evidence"] = f"{len(ev_missions)} Evil_Twin missions; run-local scored support matches the rederived audit values and archived original rows."
        cross.loc[cross["item"].astype(str).eq(support_item), "issue"] = "No per-mission support values required correction."
    elif not cross["item"].astype(str).eq(support_item).any():
        cross.loc[len(cross)] = {
            "item": support_item,
            "status": "PASS",
            "evidence": f"{len(ev_missions)} Evil_Twin missions; run-local scored support matches the rederived audit values and archived original rows; pooled support={int(ev_missions.evil_twin_scored_support.sum())}.",
            "issue": "No per-mission support values required correction.",
        }
    cross.to_csv(cross_csv, index=False)

    correction = {
        "protocol": "post-evaluation reporting correction; no inference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_evaluation_status": execution["status"],
        "frozen_evaluation_scored_once": execution["scored_once"],
        "original_experiment_py_sha256": "20661ecaacb49bdf3d0d67aab9f14f139abee509e6163c1752b490a0a56c127e",
        "reporting_fixed_experiment_py_sha256": sha256(ROOT / "simulation/research_v5_4r2/experiment.py"),
        "frozen_checkpoint_sha256": execution["checkpoint_sha256"],
        "frozen_test_spec_sha256": execution["test_spec_sha256"],
        "corrected_conditional_recall": {
            "pooled": {"recall": pooled_recall, "tp": pooled_tp, "timely_evil_twin_rows": pooled_scored, "scheduled_evil_twin_rows": pooled_scheduled},
            "per_condition": recalls,
        },
        "run_clustered_alert_rate_intervals": alert_intervals,
        "original_generated_once_hashes": {**{name: expected for name, (_, expected) in raw_hash_checks.items()},
                                            "per_mission_metrics": sha256(GENERATED / "analysis/per_mission_metrics.csv")},
        "corrected_artifact_sha256": {
            "metrics": sha256(metrics_path),
            "report": sha256(report),
            "dashboard": sha256(dashboard),
            "per_condition_metrics": sha256(condition_csv),
            "per_mission_metrics": sha256(ANALYSIS / "per_mission_metrics.csv"),
            "cross_verification": sha256(cross_csv),
        },
        "generated_once_archive": str(GENERATED.relative_to(ROOT)),
        "inference_rerun": False,
        "predictions_or_labels_changed": False,
        "reporting_correction_script_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(ANALYSIS / "reporting_correction.json", correction)
    print(json.dumps({
        "pooled_conditional_evil_twin_recall": pooled_recall,
        "pooled_tp": pooled_tp,
        "pooled_timely_evil_twin_rows": pooled_scored,
        "pooled_scheduled_evil_twin_rows": pooled_scheduled,
        "per_condition": recalls,
        "corrected_artifact_sha256": correction["corrected_artifact_sha256"],
        "generated_once_archive": str(GENERATED),
        "inference_rerun": False,
    }, indent=2))


if __name__ == "__main__":
    main()
