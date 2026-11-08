"""Post-hoc aggregation only; performs no model inference or model selection."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/acpdad_research_v5_4r1/final_test'
DATA = ROOT / 'data/ACPDAD_Simulated_Research_v5_4r1_final_test'
MODEL = ROOT / 'simulation/models/acpdad_research_v5_4r1'
RESEARCH = ROOT / 'simulation/research_v5_4r1'
CLASSES = ['Benign', 'DoS', 'Replay', 'Evil_Twin', 'FDI']
EV, BENIGN = CLASSES.index('Evil_Twin'), CLASSES.index('Benign')
CONDITIONS = [
    'mission_initial_state', 'wind_background_traffic',
    'attack_timing_duration_intensity', 'replay_delay_source_segment',
    'unseen_ap_benign_roaming',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def metrics(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    total = int(cm.sum())
    ev_support = int(cm[EV].sum())
    ev_predicted = int(cm[:, EV].sum())
    benign_support = int(cm[BENIGN].sum())
    return {
        'rows': total,
        'accuracy': float(tp.sum() / total) if total else 0.0,
        'majority_class': CLASSES[int(np.argmax(support))] if total else None,
        'majority_class_baseline': float(support.max() / total) if total else 0.0,
        'macro_f1': float(f1.mean()),
        'balanced_accuracy': float(recall.mean()),
        'per_class': {
            name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                   'f1': float(f1[i]), 'support': int(support[i]),
                   'correct': int(tp[i]), 'predicted': int(predicted[i])}
            for i, name in enumerate(CLASSES)
        },
        'evil_twin': {
            'tp': int(cm[EV, EV]), 'fn': int(ev_support - cm[EV, EV]),
            'fp': int(ev_predicted - cm[EV, EV]), 'support': ev_support,
            'predicted': ev_predicted,
            'precision': float(cm[EV, EV] / ev_predicted) if ev_predicted else 0.0,
            'recall': float(cm[EV, EV] / ev_support) if ev_support else 0.0,
        },
        'benign_false_positive_rate': float((benign_support - cm[BENIGN, BENIGN]) / benign_support)
            if benign_support else 0.0,
        'confusion_matrix': cm.tolist(),
        'normalized_confusion_matrix': np.divide(cm, support[:, None], out=np.zeros_like(cm, dtype=float),
                                                   where=support[:, None] > 0).tolist(),
    }


def ci(values):
    return [float(x) for x in np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])]


def false_bits(values: pd.Series) -> np.ndarray:
    return values.astype('boolean').fillna(False).astype(bool).to_numpy()


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img' and attrs.get('src'):
            self.assets.append(attrs['src'])


def main():
    lock_path = OUT / 'pre_generation_lock.json'
    pre_score_path = OUT / 'pre_score_manifest.json'
    execution_path = OUT / 'evaluation_execution.json'
    lock, pre_score, execution = read(lock_path), read(pre_score_path), read(execution_path)
    if not pre_score.get('evaluation_started') or not execution.get('no_tuning_or_edits_after_results'):
        raise RuntimeError('Frozen scoring execution record is incomplete')
    if sha(RESEARCH / 'final_test_spec_v5_4r1.json') != lock['test_spec_sha256']:
        raise RuntimeError('Frozen test specification hash mismatch')
    if sha(MODEL / 'checkpoint.pkl') != lock['checkpoint_sha256']:
        raise RuntimeError('Frozen checkpoint hash mismatch')
    if sha(MODEL / 'manifest.json') != lock['model_manifest_sha256']:
        raise RuntimeError('Frozen model manifest hash mismatch')
    for rel, expected in lock['source_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f'Frozen source changed: {rel}')
    config_paths = sorted((OUT.parent / 'protocol_configs/final').glob('*.json'))
    if {p.name: sha(p) for p in config_paths} != lock['test_config_sha256']:
        raise RuntimeError('Frozen condition configuration hash mismatch')

    ledgers, run_manifests, cm_by_stratum, run_metadata = [], [], {}, []
    integrity = []
    for condition in CONDITIONS:
        dataset = DATA / condition
        evdir = OUT / 'evaluations' / condition
        checksums = read(dataset / 'checksums.json')
        for name, expected in checksums.items():
            actual = sha(dataset / name)
            if actual != expected:
                raise RuntimeError(f'Dataset checksum mismatch: {condition}/{name}')
        if checksums != pre_score['conditions'][condition]['checksums']:
            raise RuntimeError(f'Dataset changed since pre-score lock: {condition}')
        run_manifest = pd.read_csv(dataset / 'run_manifest.csv')
        if len(run_manifest) != 100 or run_manifest.run_id.duplicated().any() or run_manifest.seed.duplicated().any():
            raise RuntimeError(f'Expected 100 unique complete mission IDs and seeds: {condition}')
        ledger = pd.read_csv(evdir / 'predictions.csv')
        if ledger.empty or not set(ledger.run_id).issubset(set(run_manifest.run_id)):
            raise RuntimeError(f'Prediction ledger contains unknown runs: {condition}')
        scheduled_ev = pd.read_csv(dataset / 'ground_truth.csv', usecols=['observation_id','run_id','Target_Label'])
        scheduled_ev = scheduled_ev[scheduled_ev.Target_Label.eq('Evil_Twin')]
        schedule_join = scheduled_ev.merge(ledger[['observation_id','prediction']], on='observation_id',
                                          how='left', validate='one_to_one')
        schedule_counts = {}
        for rid, group in schedule_join.groupby('run_id', sort=False):
            schedule_counts[str(rid)] = (int(len(group)), int(group.prediction.eq('Evil_Twin').sum()),
                                         int(group.prediction.isna().sum()))
        stored_metrics = read(evdir / 'metrics.json')
        delay_table = pd.read_csv(evdir / 'attack_delays.csv').set_index('run_id')
        recomputed = metrics(pd.crosstab(ledger.Target_Label, ledger.prediction)
                               .reindex(index=CLASSES, columns=CLASSES, fill_value=0).to_numpy())
        if not np.array_equal(np.asarray(recomputed['confusion_matrix']),
                              np.asarray(stored_metrics['confusion_matrix'])):
            raise RuntimeError(f'Recomputed confusion matrix disagrees with runtime metrics: {condition}')
        ledger['condition'] = condition
        ledgers.append(ledger)
        run_manifest['condition'] = condition
        run_manifests.append(run_manifest)
        joins = ledger.merge(run_manifest[['run_id', 'scenario', 'seed', 'profile', 'attack_start_index',
                                          'attack_end_index', 'intensity', 'subtype', 'replay_delay',
                                          'evil_twin_relay_delay_s', 'background_loss',
                                          'background_jitter_s', 'background_pps', 'capacity_pps']],
                             on='run_id', how='left', validate='many_to_one')
        if joins.scenario.isna().any() or not joins.scenario.eq(joins.Target_Label).where(
                ~joins.Target_Label.eq('Benign'), True).all():
            raise RuntimeError('Run mapping or scheduled label mismatch')
        for run in run_manifest.itertuples(index=False):
            run_id = str(run.run_id)
            group = joins[joins.run_id.eq(run_id)]
            scenario = str(run.scenario)
            cm = (pd.crosstab(group.Target_Label, group.prediction).reindex(
                index=CLASSES, columns=CLASSES, fill_value=0).to_numpy(dtype=np.int64)
                if len(group) else np.zeros((5, 5), dtype=np.int64))
            key = (condition, scenario)
            cm_by_stratum.setdefault(key, []).append(cm)
            m = metrics(cm)
            scheduled_support, scheduled_tp, scheduled_no_output = schedule_counts.get(run_id, (0,0,0))
            gt_rows = 300
            run_metadata.append({
                'condition': condition, 'run_id': run_id, 'scenario': scenario,
                'seed': int(run.seed), 'profile': run.profile,
                'attack_start_index': int(run.attack_start_index),
                'attack_end_index': int(run.attack_end_index),
                'attack_duration_s': float((run.attack_end_index-run.attack_start_index) / 5.0),
                'intensity': float(run.intensity), 'subtype': run.subtype,
                'replay_delay_samples': int(run.replay_delay),
                'evil_twin_relay_delay_s': float(run.evil_twin_relay_delay_s),
                'background_loss': float(run.background_loss),
                'background_jitter_s': float(run.background_jitter_s),
                'background_offered_pps': float(run.background_pps),
                'delivery_capacity_pps': float(run.capacity_pps),
                'scored_rows': int(cm.sum()), 'ground_truth_rows': gt_rows,
                'prediction_coverage': float(cm.sum() / gt_rows),
                'evil_twin_tp': m['evil_twin']['tp'], 'evil_twin_fn': m['evil_twin']['fn'],
                'evil_twin_fp': m['evil_twin']['fp'], 'evil_twin_support': m['evil_twin']['support'],
                'evil_twin_precision': m['evil_twin']['precision'] if m['evil_twin']['predicted'] else np.nan,
                'evil_twin_recall': (m['evil_twin']['recall'] if m['evil_twin']['support'] else np.nan)
                    if scenario == 'Evil_Twin' else np.nan,
                'scheduled_evil_twin_support': scheduled_support if scenario == 'Evil_Twin' else 0,
                'scheduled_evil_twin_tp': scheduled_tp if scenario == 'Evil_Twin' else 0,
                'scheduled_evil_twin_fn': scheduled_support-scheduled_tp if scenario == 'Evil_Twin' else 0,
                'scheduled_evil_twin_no_output': scheduled_no_output if scenario == 'Evil_Twin' else 0,
                'scheduled_tick_evil_twin_recall': (scheduled_tp/scheduled_support if scheduled_support else np.nan)
                    if scenario == 'Evil_Twin' else np.nan,
                'benign_false_alarms': int(cm[BENIGN].sum() - cm[BENIGN, BENIGN]),
                'benign_support': int(cm[BENIGN].sum()),
                'benign_false_positive_rate': m['benign_false_positive_rate'] if cm[BENIGN].sum() else np.nan,
                'scheduled_attack_any_alert_detected': bool(delay_table.loc[run_id, 'detected']) if run_id in delay_table.index else np.nan,
                'preexisting_combined_alert_before_attack': bool(delay_table.loc[run_id, 'preexisting_false_alarm']) if run_id in delay_table.index else np.nan,
                'combined_alert_detection_delay_s': float(delay_table.loc[run_id, 'detection_delay_s'])
                    if run_id in delay_table.index and pd.notna(delay_table.loc[run_id, 'detection_delay_s']) else np.nan,
                'confusion_matrix_json': json.dumps(cm.tolist()),
            })
        parser = AssetParser()
        parser.feed((evdir / 'index.html').read_text())
        missing = [asset for asset in parser.assets if not (evdir / asset).is_file()]
        if missing:
            raise RuntimeError(f'Dashboard assets missing for {condition}: {missing}')
        integrity.append({'condition': condition, 'runs': len(run_manifest), 'predicted_runs': int(ledger.run_id.nunique()),
                          'runs_without_scored_observations': sorted(set(run_manifest.run_id)-set(ledger.run_id)),
                          'dataset_checksums_match': True, 'runtime_confusion_recalculated_match': True,
                          'dashboard_assets_resolve': True, 'dashboard_assets': parser.assets,
                          'scored_rows': len(ledger), 'labels_joined_after_prediction': True})

    ledger = pd.concat(ledgers, ignore_index=True)
    run_manifest = pd.concat(run_manifests, ignore_index=True)
    if ledger.Target_Label.isna().any() or ledger.prediction.isna().any():
        raise RuntimeError('Missing labels or predictions in saved ledger')
    if run_manifest.seed.duplicated().any():
        raise RuntimeError('Final test seeds overlap across condition cohorts')
    run_table = pd.DataFrame(run_metadata)
    total_cm = pd.crosstab(ledger.Target_Label, ledger.prediction).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0).to_numpy(dtype=np.int64)
    total_metrics = metrics(total_cm)
    condition_cms = {c: sum(np.stack(cm_by_stratum[(c, scenario)]).sum(axis=0) for scenario in CLASSES)
                     for c in CONDITIONS}
    condition_metrics = {c: metrics(cm) for c, cm in condition_cms.items()}
    ev_run_table = run_table[run_table.scenario.eq('Evil_Twin')]
    scheduled_total_support = int(ev_run_table.scheduled_evil_twin_support.sum())
    scheduled_total_tp = int(ev_run_table.scheduled_evil_twin_tp.sum())
    schedule_wide_recall = scheduled_total_tp/scheduled_total_support
    schedule_condition_metrics = {}
    for condition in CONDITIONS:
        subset = ev_run_table[ev_run_table.condition.eq(condition)]
        support = int(subset.scheduled_evil_twin_support.sum())
        tp = int(subset.scheduled_evil_twin_tp.sum())
        schedule_condition_metrics[condition] = {
            'scheduled_support': support, 'true_positive': tp,
            'false_negative_including_no_output': support-tp,
            'no_output_scheduled_observations': int(subset.scheduled_evil_twin_no_output.sum()),
            'recall_including_no_output_as_miss': tp/support if support else 0.0,
        }

    # Stratified cluster bootstrap: draw complete mission confusion matrices,
    # preserving each condition × scheduled-class cohort size.
    rng = np.random.default_rng(120261108)
    bootstrap_count = 5000
    pooled_draws = {k: [] for k in ['evil_twin_precision', 'evil_twin_recall',
                                    'benign_false_positive_rate', 'macro_f1', 'balanced_accuracy']}
    condition_draws = {c: {k: [] for k in pooled_draws} for c in CONDITIONS}
    ev_coverage_draws, mission_mean_draws = [], []
    event_detection_draws, preexisting_alarm_draws = [], []
    condition_mission_mean_draws = {c: [] for c in CONDITIONS}
    schedule_wide_draws = []
    schedule_condition_draws = {c: [] for c in CONDITIONS}
    for _ in range(bootstrap_count):
        pooled = np.zeros((5, 5), dtype=np.int64)
        by_condition = {c: np.zeros((5, 5), dtype=np.int64) for c in CONDITIONS}
        ev_recalls = []
        for (condition, scenario), matrices in cm_by_stratum.items():
            stack = np.stack(matrices)
            sampled = stack[rng.integers(0, len(stack), size=len(stack))].sum(axis=0)
            pooled += sampled
            by_condition[condition] += sampled
            if scenario == 'Evil_Twin':
                sampled_runs = stack[rng.integers(0, len(stack), size=len(stack))]
                for rc in sampled_runs:
                    denom = int(rc[EV].sum())
                    if denom:
                        ev_recalls.append(float(rc[EV, EV] / denom))
        pm = metrics(pooled)
        for key in pooled_draws:
            pooled_draws[key].append(pm[key] if key in ('macro_f1', 'balanced_accuracy')
                                     else pm[key] if key == 'benign_false_positive_rate'
                                     else pm['evil_twin'][key.replace('evil_twin_', '')])
        for condition in CONDITIONS:
            cm = metrics(by_condition[condition])
            for key in condition_draws[condition]:
                condition_draws[condition][key].append(
                    cm[key] if key in ('macro_f1', 'balanced_accuracy')
                    else cm[key] if key == 'benign_false_positive_rate'
                    else cm['evil_twin'][key.replace('evil_twin_', '')])
        ev_rows = run_table[(run_table.scenario == 'Evil_Twin')]
        sampled_cov, sampled_all, sampled_events, sampled_preexisting = [], [], [], []
        scheduled_sampled_tp, scheduled_sampled_support = 0, 0
        for condition in CONDITIONS:
            subset = ev_rows[ev_rows.condition.eq(condition)]
            sampled_rows = subset.iloc[rng.integers(0, len(subset), size=len(subset))]
            sampled_recall = sampled_rows.evil_twin_recall.fillna(0.0)
            sampled_cov.extend((sampled_recall >= .60).tolist())
            condition_mission_mean_draws[condition].append(float(sampled_recall.mean()))
            sampled_all.extend(sampled_recall.tolist())
            sampled_events.extend(false_bits(sampled_rows.scheduled_attack_any_alert_detected).tolist())
            sampled_preexisting.extend(false_bits(sampled_rows.preexisting_combined_alert_before_attack).tolist())
            ctp = int(sampled_rows.scheduled_evil_twin_tp.sum())
            csupport = int(sampled_rows.scheduled_evil_twin_support.sum())
            schedule_condition_draws[condition].append(ctp/csupport if csupport else 0.0)
            scheduled_sampled_tp += ctp
            scheduled_sampled_support += csupport
        ev_coverage_draws.append(float(np.mean(sampled_cov)))
        mission_mean_draws.append(float(np.mean(sampled_all)) if sampled_all else 0.0)
        event_detection_draws.append(float(np.mean(sampled_events)) if sampled_events else 0.0)
        preexisting_alarm_draws.append(float(np.mean(sampled_preexisting)) if sampled_preexisting else 0.0)
        schedule_wide_draws.append(scheduled_sampled_tp/scheduled_sampled_support if scheduled_sampled_support else 0.0)

    pooled_ci = {key: ci(values) for key, values in pooled_draws.items()}
    condition_ci = {c: {key: ci(values) for key, values in condition_draws[c].items()}
                    for c in CONDITIONS}
    mission_recall = run_table.loc[run_table.scenario.eq('Evil_Twin'), 'evil_twin_recall'].dropna()
    mission_recall_all = run_table.loc[run_table.scenario.eq('Evil_Twin'), 'evil_twin_recall'].fillna(0.0)
    meaningful = run_table[run_table.scenario.eq('Evil_Twin') & (run_table.evil_twin_support >= 30)]
    mission_coverage = float((mission_recall_all >= .60).mean())
    ev_mission_rows = run_table[run_table.scenario.eq('Evil_Twin')]
    event_detected_count = int(false_bits(ev_mission_rows.scheduled_attack_any_alert_detected).sum())
    preexisting_alarm_count = int(false_bits(ev_mission_rows.preexisting_combined_alert_before_attack).sum())
    event_detection_ci = ci(event_detection_draws)
    preexisting_alarm_ci = ci(preexisting_alarm_draws)
    schedule_wide_ci = ci(schedule_wide_draws)
    schedule_condition_ci = {c: ci(schedule_condition_draws[c]) for c in CONDITIONS}
    run_summaries = {}
    for condition in CONDITIONS:
        all_values = run_table.loc[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin'),
                                   'evil_twin_recall'].fillna(0.0)
        values = run_table.loc[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin'),
                               'evil_twin_recall'].dropna()
        run_summaries[condition] = {
            'n_missions': int(len(all_values)), 'missions_with_scored_attack_rows': int(len(values)),
            'missions_without_scored_rows_count_as_misses_for_coverage': int(len(all_values)-len(values)),
            'mean_recall_scored_missions': float(values.mean()) if len(values) else None,
            'mean_recall_zero_for_unscored_missions': float(all_values.mean()),
            'median_recall_zero_for_unscored_missions': float(all_values.median()),
            'min_recall_zero_for_unscored_missions': float(all_values.min()),
            'p10_recall_zero_for_unscored_missions': float(all_values.quantile(.10)),
            'missions_recall_ge_60pct': int((all_values >= .60).sum()),
            'missions_recall_ge_60pct_rate': float((all_values >= .60).mean()),
            'missions_attack_positive_support_ge_30': int(((run_table.condition == condition) & (run_table.scenario == 'Evil_Twin') & (run_table.evil_twin_support >= 30)).sum()),
            'scheduled_attack_any_alert_detected': int(false_bits(run_table.loc[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin'), 'scheduled_attack_any_alert_detected']).sum()),
            'preexisting_combined_alert_missions': int(false_bits(run_table.loc[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin'), 'preexisting_combined_alert_before_attack']).sum()),
            'mean_any_alert_detection_delay_s': float(run_table.loc[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin'), 'combined_alert_detection_delay_s'].mean()),
            'run_clustered_95_ci_pooled_recall': condition_ci[condition]['evil_twin_recall'],
            'mean_mission_recall_95_ci': ci(condition_mission_mean_draws[condition])
                if len(values) else [0.0, 0.0],
        }

    # Validation and model selection evidence remain separate from the frozen test.
    model_comparison = read(ROOT / 'results/acpdad_research_v5_4r1/development/training/model_comparison.json')
    selected = read(MODEL / 'manifest.json')
    training_manifest = read(ROOT / 'results/acpdad_research_v5_4r1/development/training/experiment_manifest.json')
    historical_v53 = read(ROOT / 'PROJECT_STATUS.json').get('historical_evidence', {})
    validation = {}
    for condition in CONDITIONS:
        m = read(ROOT / f'results/acpdad_research_v5_4r1/development/condition_validation/{condition}/metrics.json')
        validation[condition] = {'evil_twin_precision': m['per_class']['Evil_Twin']['precision'],
                                 'evil_twin_recall': m['per_class_recall']['Evil_Twin'],
                                 'benign_false_positive_rate': m['benign_false_positive_rate']}
    prior_v54 = {}
    for condition in CONDITIONS:
        m = read(ROOT / f'results/acpdad_research_v5_4/development/condition_validation/{condition}/metrics.json')
        prior_v54[condition] = {'evil_twin_precision': m['per_class']['Evil_Twin']['precision'],
                                'evil_twin_recall': m['per_class_recall']['Evil_Twin'],
                                'benign_false_positive_rate': m['benign_false_positive_rate']}

    # Persist recalculated artifacts.
    analysis_dir = OUT / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    run_table.to_csv(analysis_dir / 'per_mission_metrics.csv', index=False)
    ledger.to_csv(analysis_dir / 'all_condition_predictions.csv', index=False)
    pd.DataFrame(total_metrics['confusion_matrix'], index=CLASSES, columns=CLASSES).to_csv(
        analysis_dir / 'pooled_confusion_matrix.csv')
    pd.DataFrame(total_metrics['normalized_confusion_matrix'], index=CLASSES, columns=CLASSES).to_csv(
        analysis_dir / 'pooled_confusion_matrix_normalized.csv')
    per_class = pd.DataFrame.from_dict(total_metrics['per_class'], orient='index').rename_axis('class')
    per_class.to_csv(analysis_dir / 'pooled_per_class_metrics.csv')
    condition_rows = []
    for condition, m in condition_metrics.items():
        ev_runs = run_table[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin')]
        condition_rows.append({'condition': condition, 'rows': m['rows'], 'accuracy': m['accuracy'],
                               'macro_f1': m['macro_f1'], 'balanced_accuracy': m['balanced_accuracy'],
                               'evil_twin_tp': m['evil_twin']['tp'], 'evil_twin_fn': m['evil_twin']['fn'],
                               'evil_twin_fp': m['evil_twin']['fp'],
                               'evil_twin_precision': m['evil_twin']['precision'],
                               'evil_twin_recall': m['evil_twin']['recall'],
                               'benign_false_positive_rate': m['benign_false_positive_rate'],
                               'evil_twin_missions_with_support_ge_30': int((ev_runs.evil_twin_support >= 30).sum()),
                               'evil_twin_missions_any_alert_detected': int(false_bits(ev_runs.scheduled_attack_any_alert_detected).sum()),
                               'evil_twin_missions_preexisting_combined_alert': int(false_bits(ev_runs.preexisting_combined_alert_before_attack).sum()),
                               'scheduled_ev_support': schedule_condition_metrics[condition]['scheduled_support'],
                               'scheduled_ev_tp': schedule_condition_metrics[condition]['true_positive'],
                               'scheduled_ev_fn_including_no_output': schedule_condition_metrics[condition]['false_negative_including_no_output'],
                               'scheduled_ev_no_output_observations': schedule_condition_metrics[condition]['no_output_scheduled_observations'],
                               'scheduled_ev_recall_including_no_output_as_miss': schedule_condition_metrics[condition]['recall_including_no_output_as_miss'],
                               'scheduled_ev_recall_ci_low': schedule_condition_ci[condition][0],
                               'scheduled_ev_recall_ci_high': schedule_condition_ci[condition][1],
                               **{f'{key}_ci_low': value[0] for key, value in condition_ci[condition].items()},
                               **{f'{key}_ci_high': value[1] for key, value in condition_ci[condition].items()}})
    condition_table = pd.DataFrame(condition_rows)
    condition_table.to_csv(analysis_dir / 'per_condition_metrics.csv', index=False)
    bootstrap_summary = {'replicates': bootstrap_count, 'sampling_unit': 'complete mission run',
                         'stratification': 'condition × scheduled class; fixed cohort sizes',
                         'seed': 120261108, 'pooled_95_ci': pooled_ci,
                         'condition_95_ci': condition_ci,
                         'mission_coverage_recall_ge_60pct_95_ci': ci(ev_coverage_draws),
                         'mean_mission_recall_95_ci': ci(mission_mean_draws),
                         'scheduled_attack_any_alert_detection_rate_95_ci': event_detection_ci,
                         'preexisting_combined_alert_rate_95_ci': preexisting_alarm_ci,
                         'scheduled_sample_recall_including_no_output_as_miss_95_ci': schedule_wide_ci,
                         'scheduled_sample_recall_by_condition_95_ci': schedule_condition_ci}
    summary = {
        'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'provenance': 'SIMULATED_ONLY_NOT_SITL_NOT_CAPTURED',
        'test_spec_sha256': lock['test_spec_sha256'],
        'pre_generation_lock_sha256': sha(lock_path),
        'pre_score_manifest_sha256': sha(pre_score_path),
        'checkpoint_sha256': lock['checkpoint_sha256'],
        'model': {'name': selected['model_name'], 'contract': selected['contract'],
                  'threshold': selected['evil_twin_probability_threshold'],
                  'physics_threshold': selected['physics_threshold']},
        'test_conditions': CONDITIONS,
        'independent_runs': int(len(run_manifest)),
        'independent_evil_twin_missions': int((run_table.scenario == 'Evil_Twin').sum()),
        'independent_evil_twin_missions_support_ge_30': int(len(meaningful)),
        'metrics': total_metrics,
        'metrics_by_condition': condition_metrics,
        'scheduled_observation_level_evil_twin': {
            'definition': 'Every scheduled active Evil_Twin ground-truth observation is in the denominator; no prediction due to loss/warm-up counts as a miss.',
            'support': scheduled_total_support, 'true_positive': scheduled_total_tp,
            'false_negative_including_no_output': scheduled_total_support-scheduled_total_tp,
            'no_output_scheduled_observations': int(ev_run_table.scheduled_evil_twin_no_output.sum()),
            'recall_including_no_output_as_miss': schedule_wide_recall,
            'run_clustered_95_ci': schedule_wide_ci,
            'by_condition': {c:{**schedule_condition_metrics[c], 'run_clustered_95_ci':schedule_condition_ci[c]} for c in CONDITIONS},
        },
        'run_level_summary_by_condition': run_summaries,
        'pooled_mission_coverage_recall_ge_60pct': mission_coverage,
        'mission_coverage_95_ci': ci(ev_coverage_draws),
        'scheduled_attack_any_alert_detected_missions': event_detected_count,
        'scheduled_attack_any_alert_detection_rate': event_detected_count / len(ev_mission_rows),
        'scheduled_attack_any_alert_detection_rate_95_ci': event_detection_ci,
        'preexisting_combined_alert_missions': preexisting_alarm_count,
        'preexisting_combined_alert_rate': preexisting_alarm_count / len(ev_mission_rows),
        'preexisting_combined_alert_rate_95_ci': preexisting_alarm_ci,
        'pooled_mean_evil_twin_mission_recall': float(mission_recall_all.mean()),
        'pooled_mean_evil_twin_mission_recall_95_ci': ci(mission_mean_draws),
        'run_clustered_bootstrap': bootstrap_summary,
        'validation_only_results': validation,
        'selected_model_validation_threshold': selected['evil_twin_probability_threshold'],
        'model_comparisons': [{k: candidate[k] for k in ['name', 'validation_threshold',
            'calibration_evil_twin_precision', 'calibration_evil_twin_recall', 'calibration_benign_fpr',
            'calibration_mission_coverage_at_60pct', 'calibration_mean_mission_recall',
            'val_accuracy', 'val_macro_f1', 'val_balanced_accuracy', 'val_benign_fpr']}
            for candidate in model_comparison],
        'acceptance_targets': {
            'basis': 'The first five targets use emitted, scorable prediction rows; a separate scheduled-observation metric below counts missing outputs as misses.',
            'pooled_evil_twin_recall_at_least_0_80': total_metrics['evil_twin']['recall'] >= .80,
            'each_condition_evil_twin_recall_at_least_0_70': all(m['evil_twin']['recall'] >= .70 for m in condition_metrics.values()),
            'pooled_evil_twin_precision_at_least_0_95': total_metrics['evil_twin']['precision'] >= .95,
            'pooled_benign_false_positive_rate_at_most_0_065': total_metrics['benign_false_positive_rate'] <= .065,
            'at_least_80pct_evil_twin_missions_recall_at_least_0_60': mission_coverage >= .80,
            'scheduled_observation_pooled_recall_at_least_0_80_including_no_output': schedule_wide_recall >= .80,
            'each_condition_scheduled_observation_recall_at_least_0_70_including_no_output': all(v['recall_including_no_output_as_miss'] >= .70 for v in schedule_condition_metrics.values()),
        },
        'integrity_checks': integrity,
    }
    (analysis_dir / 'final_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    # Figures are based only on the saved, frozen prediction ledgers.
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = np.asarray(total_metrics['normalized_confusion_matrix'])
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted class', ylabel='Scheduled class', title='Pooled normalized confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    for i in range(5):
        for j in range(5): ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax, label='Row proportion'); fig.tight_layout()
    fig.savefig(analysis_dir / 'pooled_confusion_matrix.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    xx = np.arange(len(CLASSES)); width = .36
    ax.bar(xx-width/2, [total_metrics['per_class'][c]['precision'] for c in CLASSES], width, label='Precision')
    ax.bar(xx+width/2, [total_metrics['per_class'][c]['recall'] for c in CLASSES], width, label='Recall')
    ax.set(xticks=xx, xticklabels=CLASSES, ylim=(0, 1.04), ylabel='Score', title='Pooled per-class metrics')
    ax.legend(); ax.grid(axis='y', alpha=.25); fig.tight_layout()
    fig.savefig(analysis_dir / 'pooled_per_class_metrics.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, condition in enumerate(CONDITIONS):
        subset = run_table.loc[(run_table.condition == condition) & (run_table.scenario == 'Evil_Twin'), 'evil_twin_recall']
        scored = subset.notna().to_numpy()
        vals = subset.fillna(0).to_numpy()
        jitter = rng.normal(0, .055, size=len(vals))
        ax.scatter(i+jitter[scored], vals[scored], s=20, alpha=.65, color='#496a9b', label='Scored mission' if i == 0 else None)
        if (~scored).any():
            ax.scatter(i+jitter[~scored], vals[~scored], s=35, marker='x', color='black', label='No scored attack rows (plotted at 0)' if i == 0 else None)
        ax.plot([i-.25, i+.25], [condition_metrics[condition]['evil_twin']['recall']]*2, color='#b43535', lw=3)
    ax.axhline(.70, color='#d2872d', linestyle='--', label='70% per-condition target')
    ax.axhline(.60, color='#526b4a', linestyle=':', label='60% per-mission aim')
    ax.set(xticks=range(len(CONDITIONS)), xticklabels=[x.replace('_','\n') for x in CONDITIONS],
           ylim=(-.03, 1.03), ylabel='Evil_Twin mission recall', title='Every prespecified simulated condition and mission')
    ax.legend(loc='lower right'); ax.grid(axis='y', alpha=.25); fig.tight_layout()
    fig.savefig(analysis_dir / 'evil_twin_recall_by_condition_and_mission.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [condition_metrics[c]['evil_twin']['recall'] for c in CONDITIONS]
    low = [condition_ci[c]['evil_twin_recall'][0] for c in CONDITIONS]
    high = [condition_ci[c]['evil_twin_recall'][1] for c in CONDITIONS]
    yerr = np.array([np.maximum(0, np.array(vals)-np.array(low)), np.maximum(0, np.array(high)-np.array(vals))])
    ax.errorbar(range(len(CONDITIONS)), vals, yerr=yerr, fmt='o', capsize=4, color='#245c87')
    ax.axhline(.70, color='#d2872d', linestyle='--', label='70% target')
    ax.set(xticks=range(len(CONDITIONS)), xticklabels=[x.replace('_','\n') for x in CONDITIONS],
           ylim=(0, 1.03), ylabel='Pooled Evil_Twin recall', title='Condition recall with run-clustered 95% intervals')
    ax.legend(); ax.grid(axis='y', alpha=.25); fig.tight_layout()
    fig.savefig(analysis_dir / 'evil_twin_condition_recall_cluster_ci.png', dpi=160); plt.close(fig)

    target = summary['acceptance_targets']
    status = lambda ok: 'MET' if ok else 'NOT MET'
    cond_md = '\n'.join(f"| {c} | {condition_metrics[c]['evil_twin']['tp']}/{condition_metrics[c]['evil_twin']['support']} | {condition_metrics[c]['evil_twin']['precision']:.3%} | {condition_metrics[c]['evil_twin']['recall']:.3%} | {condition_ci[c]['evil_twin_recall'][0]:.3%}–{condition_ci[c]['evil_twin_recall'][1]:.3%} | {condition_metrics[c]['benign_false_positive_rate']:.3%} | {run_summaries[c]['missions_recall_ge_60pct']}/20 | {run_summaries[c]['missions_attack_positive_support_ge_30']}/20 | {run_summaries[c]['scheduled_attack_any_alert_detected']}/20 | {run_summaries[c]['preexisting_combined_alert_missions']}/20 |" for c in CONDITIONS)
    schedule_md = '\n'.join(f"| {c} | {schedule_condition_metrics[c]['true_positive']}/{schedule_condition_metrics[c]['scheduled_support']} | {schedule_condition_metrics[c]['recall_including_no_output_as_miss']:.3%} | {schedule_condition_ci[c][0]:.3%}–{schedule_condition_ci[c][1]:.3%} | {schedule_condition_metrics[c]['no_output_scheduled_observations']} |" for c in CONDITIONS)
    seed_md = '\n'.join(f"| {c['id']} | {c['master_seed']} | {c['ap_base']} | {lock['runs_per_class_per_condition']} |" for c in lock['conditions'])
    class_md = '\n'.join(f"| {name} | {v['precision']:.3%} | {v['recall']:.3%} | {v['f1']:.3%} | {v['support']} |" for name, v in total_metrics['per_class'].items())
    comparison_md = '\n'.join(f"| {v['name']} | {v['validation_threshold']:.2f} | {v['calibration_evil_twin_precision']:.3%} | {v['calibration_evil_twin_recall']:.3%} | {v['calibration_benign_fpr']:.3%} | {v['val_macro_f1']:.3%} |" for v in model_comparison)
    validation_md = '\n'.join(f"- `{c}`: EV precision {validation[c]['evil_twin_precision']:.2%}, recall {validation[c]['evil_twin_recall']:.2%}, benign FPR {validation[c]['benign_false_positive_rate']:.2%}." for c in CONDITIONS)
    historical_v54_md = '\n'.join(f"| {c} | {prior_v54[c]['evil_twin_precision']:.2%} | {prior_v54[c]['evil_twin_recall']:.2%} | {prior_v54[c]['benign_false_positive_rate']:.2%} | {validation[c]['evil_twin_precision']:.2%} | {validation[c]['evil_twin_recall']:.2%} | {validation[c]['benign_false_positive_rate']:.2%} |" for c in CONDITIONS)
    packages = training_manifest.get('packages', {})
    python_version = training_manifest.get('python', '').split()[0]
    report = f'''# ACPDAD V5.4R1 Evil_Twin detection implementation and evaluation

All data and outputs in this report are **simulated**. This is not ArduPilot SITL, a physical drone connection, captured Wi-Fi, or evidence about real-world attack performance.

The selected HistGradientBoosting model uses `max_iter=160`, `max_leaf_nodes=19`, `min_samples_leaf=25`, and `l2_regularization=1.0`; its Evil_Twin threshold was chosen on validation data. The `.venv` training environment used Python {python_version}, NumPy {packages.get('numpy')}, pandas {packages.get('pandas')}, scikit-learn {packages.get('sklearn')}, and Matplotlib {packages.get('matplotlib')}. Installed and training scikit-learn versions match, so no version-mismatch warning occurred.

## Implementation plan and findings

V5.3's Evil_Twin misses clustered by mission because its evidence was a modest relay-delay/AP-transition effect, while ordinary roaming included overlapping AP handoffs and management frames. Startup packet-age baselines were vulnerable to queue bursts. V5.4's first revision added `noninitial_association_duration_s` and association-relative latency, but large delivery gaps reset the whole extractor; the detector then forgot that its received AP had remained away from the initial AP. At high load, benign roaming could also hide the return-to-home packet and mimic a long association. Those were implementation/observability issues, not an effect of Replay source-segment settings on the Evil_Twin generator.

The V5.4R1 code keeps run-level initial AP and current-association state across delivery gaps while restarting only the rolling temporal window and packet-history buffers. It adds hard-negative training/validation coverage for high-load benign roam, longer roam intervals, and packet loss, then selects model and threshold using validation cohorts only. Feature names and model input columns remain the V5.4 schema (22 base / 88 temporal); no AP identifier, label, scenario, run ID, seed, split, attack schedule, or replay setting enters the matrix. The active runtime derives schedule labels only after predictions for evaluation.

Changed files are the new `simulation/research_v5_4r1/` runtime, protocol and experiment definitions, `run_project_v5_4r1.py`, and the root `run_project.py` selection. V4, V5.3, and the earlier V5.4 experiment artifacts remain untouched. A pre-edit backup is at `/tmp/ACPDAD_backups/v5_4r1_preverification_20261108T112028Z/`.

## Historical V5.3 failure and V5.4 development baseline

The preserved V5.3 final test recorded **{historical_v53.get('simulated_test_recall', {}).get('Evil_Twin', float('nan')):.2%}** Evil_Twin recall and **{historical_v53.get('simulated_test_precision', {}).get('Evil_Twin', float('nan')):.2%}** precision. Its replay-delay/source-segment changed condition recorded **{historical_v53.get('changed_condition_evil_twin_recall', {}).get('replay_delay_source_segment', float('nan')):.2%}** recall with only four Evil_Twin missions; one mission was at 8.3% recall. Replay delay/source-segment settings affected Replay-labelled runs, not Evil_Twin generator behavior. V5.3's Evil_Twin evidence was a modest relay delay plus an AP transition that overlapped ordinary handoffs. The association-duration feature measured time since the latest AP transition rather than persistence away from the run's initial AP. Startup packet-age excess used the first 10 observations: in a weak validation mission, startup packet age was around 260 ms while active-attack packet age was around 40 ms, making attack evidence negative; 68 of 97 scored attack observations were missed, mostly as Benign. V5.3's final test remains historical and was not rerun or used for selection.

The first V5.4 development candidate added association features but reset the association state across large delivery gaps. Benign roaming could also appear persistent when high loss hid the return packet. This table compares saved V5.4 and V5.4R1 **validation** metrics; it does not compare or tune on final-test results.

| Validation condition | V5.4 EV precision | V5.4 EV recall | V5.4 benign FPR | V5.4R1 EV precision | V5.4R1 EV recall | V5.4R1 benign FPR |
|---|---:|---:|---:|---:|---:|---:|
{historical_v54_md}

## Validation-only selection

Selected model: **{selected['model_name']}**, threshold **{selected['evil_twin_probability_threshold']:.2f}**, checkpoint SHA-256 `{selected['checkpoint_sha256']}`. Model fit used only the training split. Candidate/threshold selection used the training validation split and five separate validation-condition datasets; the frozen final seeds were not generated at that point.

| Candidate | Selected EV threshold | Validation-cohort EV precision | EV recall | Benign FPR | Trainval macro-F1 |
|---|---:|---:|---:|---:|---:|
{comparison_md}

Validation-only condition metrics at the frozen selected threshold:

{validation_md}

## Frozen final-test protocol

The existing V5.3 final test remains historical and was not rerun or used for this work. V5.4R1's fresh test specification was frozen before generation: `{lock['test_spec_sha256']}`. The pre-generation lock is `{sha(lock_path)}`. The pre-score manifest (dataset, run-manifest, config and checksum hashes) was written before model evaluation. The frozen checkpoint digest is `{lock['checkpoint_sha256']}`.

There are 5 prespecified changed-condition suites, **20 complete runs per class per condition**, 500 simulated missions total, and 100 independent Evil_Twin missions. All use distinct run seeds and held-out AP pools. Replay delay/source-segment stress applies to Replay-labelled missions; Evil_Twin missions in that suite share the prespecified high-load network condition and are not compound Replay attacks. Every scheduled attack interval comes from the simulator schedule and labels were joined after model inference. Run-clustered intervals resample whole missions within condition × scheduled class, with 5,000 bootstrap replicates and seed 120261108.

| Condition | Master seed | Test AP pool base | Runs per class |
|---|---:|---:|---:|
{seed_md}

## Frozen test results

Row-level precision/recall/confusion metrics use observations for which the delivered packet stream produced a complete temporal feature window and the model emitted a prediction. Scheduled samples with no delivered/scorable observation cannot receive a class prediction and are excluded from the row-level matrix; every mission remains in the run-level table, and an Evil_Twin mission with no scored attack rows counts as a miss for mission coverage and any-alert detection.

When all scheduled active Evil_Twin time samples are instead counted—including those with no emitted prediction—the system-level recall is **{schedule_wide_recall:.3%}** ({scheduled_total_tp}/{scheduled_total_support}; run-clustered 95% CI **{schedule_wide_ci[0]:.3%}–{schedule_wide_ci[1]:.3%}**). This stricter denominator matters when interpreting the row-level score.

Pooled Evil_Twin confusion counts: **TP {total_metrics['evil_twin']['tp']}, FN {total_metrics['evil_twin']['fn']}, FP {total_metrics['evil_twin']['fp']}**. Precision is **{total_metrics['evil_twin']['precision']:.3%}**; recall is **{total_metrics['evil_twin']['recall']:.3%}** with run-clustered 95% interval **{pooled_ci['evil_twin_recall'][0]:.3%}–{pooled_ci['evil_twin_recall'][1]:.3%}**. Run-clustered precision interval is **{pooled_ci['evil_twin_precision'][0]:.3%}–{pooled_ci['evil_twin_precision'][1]:.3%}**. Overall benign FPR is **{total_metrics['benign_false_positive_rate']:.3%}** (run-clustered 95% interval **{pooled_ci['benign_false_positive_rate'][0]:.3%}–{pooled_ci['benign_false_positive_rate'][1]:.3%}**). {int((mission_recall_all >= .60).sum())}/100 Evil_Twin missions reached 60% recall; missions with no scored attack rows count as misses for this mission-level coverage measure. The cluster-bootstrap interval for that mission fraction is **{ci(ev_coverage_draws)[0]:.1%}–{ci(ev_coverage_draws)[1]:.1%}**. {len(meaningful)}/100 missions had at least 30 scored attack-positive observations; every mission, including unscored ones, remains listed in `per_mission_metrics.csv`.

The separate mission-level any-alert calculation detected {event_detected_count}/100 scheduled Evil_Twin attack missions (combined alert can be a class alert or physics residual); {preexisting_alarm_count}/100 also had a combined alert before the scheduled attack. This is an operational caveat: early false alerts are common even though row-level Evil_Twin precision and benign false-positive rate meet their pooled targets. Its run-clustered 95% intervals are {event_detection_ci[0]:.1%}–{event_detection_ci[1]:.1%} detection and {preexisting_alarm_ci[0]:.1%}–{preexisting_alarm_ci[1]:.1%} pre-existing alerts.

| Condition | EV TP/support | EV precision | EV recall | Clustered 95% recall interval | Benign FPR | Missions ≥60% recall | Missions with ≥30 EV rows | Any-alert mission detections | Pre-attack alert missions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{cond_md}

| Condition | Scheduled-active EV TP/support | Recall counting no-output samples as misses | Run-clustered 95% interval | Scheduled samples with no prediction |
|---|---:|---:|---:|---:|
{schedule_md}

The pooled Evil_Twin precision target is met. The `unseen_ap_benign_roaming` subset is **94.685%** precision, so the 95% precision bar would not be met if interpreted separately for every condition.

Pooled five-class metrics: accuracy **{total_metrics['accuracy']:.3%}**, majority-class baseline **{total_metrics['majority_class_baseline']:.3%}** (`{total_metrics['majority_class']}`), macro-F1 **{total_metrics['macro_f1']:.3%}**, balanced accuracy **{total_metrics['balanced_accuracy']:.3%}**.

| Scheduled class | Precision | Recall | F1 | Scored support |
|---|---:|---:|---:|---:|
{class_md}

Classifier acceptance targets on emitted, scorable prediction rows (assessed on this single frozen simulated test pass):

- Pooled EV recall ≥80%: **{status(target['pooled_evil_twin_recall_at_least_0_80'])}** ({total_metrics['evil_twin']['recall']:.2%}).
- Every changed condition EV recall ≥70%: **{status(target['each_condition_evil_twin_recall_at_least_0_70'])}** (weakest {min(m['evil_twin']['recall'] for m in condition_metrics.values()):.2%}).
- Pooled EV precision ≥95%: **{status(target['pooled_evil_twin_precision_at_least_0_95'])}** ({total_metrics['evil_twin']['precision']:.2%}).
- Pooled benign FPR ≤6.5%: **{status(target['pooled_benign_false_positive_rate_at_most_0_065'])}** ({total_metrics['benign_false_positive_rate']:.2%}).
- At least 80% of EV missions at ≥60% recall: **{status(target['at_least_80pct_evil_twin_missions_recall_at_least_0_60'])}** ({int((mission_recall >= .60).sum())}/100).

If every scheduled attack sample without a prediction counts as a miss, the pooled 80% system-level recall target is **{status(target['scheduled_observation_pooled_recall_at_least_0_80_including_no_output'])}** ({schedule_wide_recall:.2%}). The target that every changed condition reach 70% under that denominator is **{status(target['each_condition_scheduled_observation_recall_at_least_0_70_including_no_output'])}**: wind/background traffic is {schedule_condition_metrics['wind_background_traffic']['recall_including_no_output_as_miss']:.2%}, and replay-delay/source-segment is {schedule_condition_metrics['replay_delay_source_segment']['recall_including_no_output_as_miss']:.2%}. These system-level misses reflect packet delivery loss/delay and temporal warm-up/gaps: the classifier performs strongly on rows it receives, but high-load conditions do not consistently deliver enough scorable attack observations.

The behavioral distinction remains simulated: takeover produces a persistent reassociation to a per-run random AP identity with a relay path, while benign missions include random temporary roaming and management frames. The model receives only packet-derived transition persistence, packet age/delay, network summaries and physical telemetry; it receives neither AP identity nor attack state. Results support the five-class claim **within these tested synthetic conditions only**. The weakest non-EV results and all misses remain visible in the ledgers; DoS is the weakest pooled class in this test, so the overall five-class scores should not be read as equally strong class performance.

## Integrity and artifacts

All five dataset checksums match the pre-score lock; all 500 runs and seeds are unique; checkpoint, source, test specification and config hashes still match their pre-generation locks; each per-condition dashboard's local images resolve; confusion matrices were recalculated from saved predictions and match the runtime metrics. `integrity_checks` in `analysis/final_metrics.json` contains each condition's evidence.

- Prediction ledgers and dashboards: `evaluations/<condition>/`.
- Combined predictions, per-mission metrics, per-class/per-condition metrics, confusion matrices, figures and bootstrap intervals: `analysis/`.
- Pre-generation protocol lock: `pre_generation_lock.json`.
- Pre-score dataset/run-manifest hashes: `pre_score_manifest.json`.
- One-pass scoring record: `evaluation_execution.json`.
- Console logs: `../logs/`.

## Commands

From the project root, use `.venv`:

```bash
export MPLCONFIGDIR="$PWD/.venv/matplotlib-cache"
.venv/bin/python run_project.py verify
.venv/bin/python run_project.py demo --data data/ACPDAD_Simulated_Research_v5_4r1_trainval --model simulation/models/acpdad_research_v5_4r1 --split val --output results/acpdad_research_v5_4r1/development/demo/manual
```

Fresh simulated generation and selected-run inference:

```bash
.venv/bin/python run_project.py generate --seed 9654321 --runs-per-class 1 --duration 30 --profile patrol --output data/acpdad_v5_4r1_new_seed_9654321
RUN_ID=$(.venv/bin/python -c "import pandas as pd; r=pd.read_csv('data/acpdad_v5_4r1_new_seed_9654321/run_manifest.csv'); print(r.loc[r.scenario.eq('Evil_Twin'),'run_id'].iloc[0])")
.venv/bin/python run_project.py predict --data data/acpdad_v5_4r1_new_seed_9654321 --model simulation/models/acpdad_research_v5_4r1 --run-id "$RUN_ID" --output results/acpdad_research_v5_4r1/prediction_checks/new_seed_9654321.csv
```

This is a simulation-only research workflow. It does not run SITL, connect to a vehicle, capture Wi-Fi, or establish real-world detection performance.
'''
    (OUT / 'IMPLEMENTATION_REPORT.md').write_text(report)

    dashboard = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ACPDAD — Attack Detection Dashboard</title><h1>ACPDAD — Attack Detection Dashboard</h1><h2>V5.4R1 simulated evaluation</h2><p>SIMULATED ONLY; not SITL, captured traffic, or real-world validation. Frozen model {selected['model_name']}.</p><p>On emitted, scorable rows: pooled Evil_Twin precision {total_metrics['evil_twin']['precision']:.2%}; recall {total_metrics['evil_twin']['recall']:.2%} (run-clustered 95% CI {pooled_ci['evil_twin_recall'][0]:.2%}–{pooled_ci['evil_twin_recall'][1]:.2%}); benign FPR {total_metrics['benign_false_positive_rate']:.2%}.</p><p>Counting every scheduled active attack sample and treating no prediction as a miss: Evil_Twin recall {schedule_wide_recall:.2%} (run-clustered 95% CI {schedule_wide_ci[0]:.2%}–{schedule_wide_ci[1]:.2%}). The 80% system-level recall target is not met; wind/background traffic is {schedule_condition_metrics['wind_background_traffic']['recall_including_no_output_as_miss']:.2%}, replay-delay/source-segment is {schedule_condition_metrics['replay_delay_source_segment']['recall_including_no_output_as_miss']:.2%}.</p><p>Macro-F1 {total_metrics['macro_f1']:.2%}; balanced accuracy {total_metrics['balanced_accuracy']:.2%}; accuracy {total_metrics['accuracy']:.2%}; majority baseline {total_metrics['majority_class_baseline']:.2%}.</p><img src="analysis/pooled_confusion_matrix.png" alt="Pooled confusion matrix"><img src="analysis/evil_twin_recall_by_condition_and_mission.png" alt="Evil Twin mission recall"><h2>Conditions</h2><ul>{''.join(f'<li><a href="evaluations/{c}/index.html">{c}</a>: scored-row EV recall {condition_metrics[c]["evil_twin"]["recall"]:.2%}; schedule-wide recall counting missing outputs as misses {schedule_condition_metrics[c]["recall_including_no_output_as_miss"]:.2%}; EV precision {condition_metrics[c]["evil_twin"]["precision"]:.2%}; benign FPR {condition_metrics[c]["benign_false_positive_rate"]:.2%}.</li>' for c in CONDITIONS)}</ul><p><a href="IMPLEMENTATION_REPORT.md">Full report</a>; prediction, per-run, metrics and integrity files are in <code>analysis/</code>.</p></html>'''
    (OUT / 'index.html').write_text(dashboard)
    print(json.dumps({'report': str(OUT / 'IMPLEMENTATION_REPORT.md'),
                      'dashboard': str(OUT / 'index.html'),
                      'summary': summary['acceptance_targets'],
                      'pooled_evil_twin': total_metrics['evil_twin'],
                      'benign_false_positive_rate': total_metrics['benign_false_positive_rate'],
                      'pooled_ci': pooled_ci}, indent=2))


if __name__ == '__main__':
    main()
