"""Validation and one-pass simulated test protocol."""
import argparse
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'config'
RESULTS=ROOT/'results'
MODEL=ROOT/'models/selected'
TRAIN=ROOT/'data/train_validation'
FINAL_DATA=ROOT/'data/frozen_test'
FINAL_DIR=RESULTS/'frozen_test'

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj,exclusive=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);mode='x' if exclusive else 'w'
    with p.open(mode) as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def render(stage):
    final=stage=='final'
    matrix=read(SRC/('frozen_test.json' if final else 'validation_matrix.json'))
    base=read(SRC/'simulation.json');output=RESULTS/'protocol_configs'/stage;output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i,item in enumerate(matrix['conditions']):
        c=dict(base);c.update(item.get('overrides',{}));c['runtime_revision']='acpdad-sim-research-v5-4r2'
        c['runs_per_class']={'train':0,'val':0,'test':0};c['runs_per_class']['test' if final else 'val']=int(matrix['runs_per_class'])
        c['master_seeds']={'train':0,'val':0,'test':0};c['master_seeds']['test' if final else 'val']=int(item['master_seed'])
        c['train_ap_base']=50000000;c['val_ap_base']=int(item['ap_base']) if not final else 51000000+i*10000
        c['test_ap_base']=int(item['ap_base']) if final else 52000000+i*10000
        c['run_prefix']=f"v5_4r2_{'final' if final else 'dev'}_{item['id']}"
        c['matrix_condition_id']=item['id'];c['matrix_protocol_sha256']=digest(SRC/('frozen_test.json' if final else 'validation_matrix.json'))
        c['matrix_protocol_stage']='final_test' if final else 'validation'
        if final:
            c['prediction_deadline_s']=float(matrix['prediction_deadline_s'])
            c['frozen_model_settings']=matrix['frozen_model_settings']
        p=output/f"{item['id']}.json"
        rendered=json.dumps(c,indent=2)+'\n'
        if p.exists():
            if p.read_text()!=rendered:raise FileExistsError(f'Existing protocol config differs: {p}')
        else:p.write_text(rendered)
        paths.append(p)
    return paths

def finalize_spec_from_validation():
    spec_path=SRC/'frozen_test.json'
    spec=read(spec_path);manifest=read(MODEL/'manifest.json');selection=read(RESULTS/'selection/training/validation_selection.json')
    calibration_path=RESULTS/'selection/alert_calibration/alert_policy_calibration.json'
    calibration=read(calibration_path)
    if calibration['new_checkpoint_sha256']!=manifest['checkpoint_sha256'] or calibration['new_checkpoint_sha256']!=digest(MODEL/'checkpoint.pkl'):
        raise ValueError('Validation alert calibration does not match the active checkpoint')
    spec['frozen_model_settings']={
        'checkpoint_sha256':manifest['checkpoint_sha256'],
        'fallback_selected':manifest['fallback_model'],
        'primary_threshold':manifest['primary_threshold'],
        'fallback_threshold':manifest['fallback_threshold'],
        'prediction_deadline_s':manifest['prediction_deadline_s'],
        'alert_policy':calibration['selected_policy'],
        'validation_alert_acceptance_met':calibration['validation_acceptance_met']
    }
    spec['acceptance_targets']={
        'pooled_end_to_end_evil_twin_recall_min':.80,
        'per_condition_end_to_end_evil_twin_recall_min':.70,
        'pooled_evil_twin_precision_min':.95,
        'per_condition_evil_twin_precision_min':.95,
        'benign_scored_row_fpr_max':.065,
        'benign_mission_alert_rate_max':.20,
        'false_alert_episodes_per_benign_minute_max':calibration['acceptance_thresholds']['false_alert_episodes_per_benign_minute_max'],
        'evil_twin_pre_attack_alert_mission_rate_max':.20,
        'mean_attack_alert_detection_delay_s_max':calibration['acceptance_thresholds']['mean_attack_alert_detection_delay_s_max'],
    }
    spec['contract']='acpdad-sim-research-v5-4r2'
    spec['validation_selection_sha256']=digest(RESULTS/'selection/training/validation_selection.json')
    spec['validation_alert_calibration_sha256']=digest(calibration_path)
    spec['protocol_freeze_note']='These model, threshold, fallback, alert, deadline and acceptance settings are frozen before fresh test generation. If a target fails, report it; do not revise or regenerate the test.'
    spec_path.write_text(json.dumps(spec,indent=2)+'\n')
    return spec

def freeze():
    spec_path=SRC/'frozen_test.json';spec=read(spec_path);model=read(MODEL/'manifest.json')
    if FINAL_DATA.exists() or FINAL_DIR.exists():raise FileExistsError('Final test data/output already exists; refusing to replace or rescore')
    if spec.get('frozen_model_settings',{}).get('checkpoint_sha256')!=model['checkpoint_sha256']:
        raise ValueError('Final protocol does not freeze the selected checkpoint')
    configs=render('final')
    source_files=['src/acpdad/simulation.py','src/acpdad/workflow.py',
        'src/acpdad/evaluation_protocol.py','config/simulation.json',
        'config/validation_matrix.json','config/frozen_test.json',
        'run_project.py','models/selected/checkpoint.pkl','models/selected/manifest.json']
    lock={'protocol':'ACPDAD V5.4R2 frozen fresh simulated test; score once',
        'provenance':'SIMULATED_ONLY_NOT_SITL_NOT_CAPTURED','created_utc':datetime.now(timezone.utc).isoformat(),
        'test_spec_sha256':digest(spec_path),'test_config_sha256':{p.name:digest(p) for p in configs},
        'checkpoint_sha256':model['checkpoint_sha256'],'model_manifest_sha256':digest(MODEL/'manifest.json'),
        'source_sha256':{p:digest(ROOT/p) for p in source_files},
        'trainval_checksums_sha256':digest(TRAIN/'checksums.json'),
        'validation_selection_sha256':digest(RESULTS/'selection/training/validation_selection.json'),
        'validation_alert_calibration_sha256':digest(RESULTS/'selection/alert_calibration/alert_policy_calibration.json'),
        'conditions':spec['conditions'],'runs_per_class_per_condition':spec['runs_per_class'],
        'prediction_deadline_s':spec['prediction_deadline_s'],'frozen_model_settings':spec['frozen_model_settings'],
        'acceptance_targets':spec['acceptance_targets'],'dataset_root':str(FINAL_DATA.relative_to(ROOT)),
        'evaluation_root':'results/frozen_test','scored_once':False,
        'labels_joined_after_inference':True,'scheduled_observations_accounted':True}
    path=FINAL_DIR/'pre_generation_lock.json';write(path,lock,exclusive=True);return path

def lock_generated():
    lock_path=FINAL_DIR/'pre_generation_lock.json';lock=read(lock_path);spec_path=SRC/'frozen_test.json'
    if digest(spec_path)!=lock['test_spec_sha256']:raise ValueError('Frozen final test specification changed')
    if digest(RESULTS/'selection/alert_calibration/alert_policy_calibration.json')!=lock['validation_alert_calibration_sha256']:
        raise ValueError('Validation alert calibration changed after the final test was frozen')
    for rel,h in lock['source_sha256'].items():
        if digest(ROOT/rel)!=h:raise ValueError(f'Frozen source changed: {rel}')
    if digest(MODEL/'manifest.json')!=lock['model_manifest_sha256'] or read(MODEL/'manifest.json')['checkpoint_sha256']!=lock['checkpoint_sha256']:
        raise ValueError('Frozen model changed')
    configs=sorted((RESULTS/'protocol_configs/final').glob('*.json'))
    if {p.name:digest(p) for p in configs}!=lock['test_config_sha256']:raise ValueError('Final condition configs differ from the lock')
    condition_manifest={};all_ids=[];all_seeds=[];all_aps=[];condition_ap_sets={}
    expected_runs=int(lock['runs_per_class_per_condition'])*5
    for item in lock['conditions']:
        path=FINAL_DATA/item['id'];checks=read(path/'checksums.json')
        for filename,h in checks.items():
            if digest(path/filename)!=h:raise ValueError(f'Dataset checksum mismatch: {item["id"]}/{filename}')
        runs=pd.read_csv(path/'run_manifest.csv')
        if len(runs)!=expected_runs or set(runs.scenario.astype(str))!=set(['Benign','DoS','Replay','Evil_Twin','FDI']):
            raise ValueError(f'Unexpected runs/classes in final condition: {item["id"]}')
        if runs.run_id.duplicated().any() or runs.seed.duplicated().any():raise ValueError('Run ID/seed overlap within final condition')
        all_ids.extend(runs.run_id.astype(str));all_seeds.extend(runs.seed.astype(int));condition_ap_sets[item['id']]=set()
        for col in ('ap_identity','benign_roam_ap_identity','takeover_ap_identity'):
            values=set(runs[col].astype(int).tolist());all_aps.extend(values);condition_ap_sets[item['id']].update(values)
        condition_manifest[item['id']]={'path':str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),'run_manifest_sha256':digest(path/'run_manifest.csv'),
            'checksums_sha256':digest(path/'checksums.json'),'checksums':checks,'runs':len(runs),
            'ground_truth_rows':sum(1 for _ in (path/'ground_truth.csv').open())-1}
    if len(set(all_ids))!=len(all_ids) or len(set(all_seeds))!=len(all_seeds):raise ValueError('Final runs/seeds overlap across conditions')
    ap_overlaps={}
    for i,left in enumerate(condition_ap_sets):
        for right in list(condition_ap_sets)[i+1:]:
            overlap=condition_ap_sets[left]&condition_ap_sets[right]
            if overlap:ap_overlaps[f'{left}::{right}']=sorted(overlap)
    if ap_overlaps:raise ValueError(f'Final AP identities overlap across conditions: {ap_overlaps}')
    # Explicitly verify held-out seeds and AP identities are disjoint from prior datasets, including V4/V5.3/R1.
    historical_seeds=set();historical_aps=set()
    for p in (ROOT/'data').rglob('run_manifest.csv'):
        if FINAL_DATA in p.parents:continue
        try:runs=pd.read_csv(p)
        except Exception:continue
        if 'seed' in runs:historical_seeds.update(pd.to_numeric(runs.seed,errors='coerce').dropna().astype(int).tolist())
        for col in ('ap_identity','benign_roam_ap_identity','takeover_ap_identity'):
            if col in runs:historical_aps.update(pd.to_numeric(runs[col],errors='coerce').dropna().astype(int).tolist())
    final_seedset=set(all_seeds);final_apset=set(all_aps)
    isolation={'final_run_ids_unique':len(set(all_ids))==len(all_ids),'final_seeds_unique':len(final_seedset)==len(all_seeds),
        'prior_seed_overlap':sorted(final_seedset&historical_seeds),'prior_ap_overlap':sorted(final_apset&historical_aps),
        'condition_ap_overlap_checked':not bool(ap_overlaps),'condition_ap_overlap':ap_overlaps}
    if isolation['prior_seed_overlap'] or isolation['prior_ap_overlap']:raise ValueError(f'Final isolation failure: {isolation}')
    manifest={'protocol':'V5.4R2 final test data pre-score manifest','provenance':'SIMULATED_ONLY_NOT_SITL_NOT_CAPTURED',
        'pre_generation_lock_sha256':digest(lock_path),'test_spec_sha256':lock['test_spec_sha256'],
        'checkpoint_sha256':lock['checkpoint_sha256'],'conditions':condition_manifest,'isolation':isolation,'evaluation_started':False}
    p=FINAL_DIR/'pre_score_manifest.json';write(p,manifest,exclusive=True);return p

def _load_metrics(condition):return read(FINAL_DIR/'evaluations'/condition/'metrics.json')

def evaluate_final():
    from .workflow import load_checkpoint,load_dataset,_save_evaluation,score_ledger
    lock_path=FINAL_DIR/'pre_generation_lock.json';pre_path=FINAL_DIR/'pre_score_manifest.json'
    lock=read(lock_path);pre=read(pre_path)
    if pre['evaluation_started']:raise RuntimeError('This frozen test has already started scoring; refusing a second evaluation')
    if digest(lock_path)!=pre['pre_generation_lock_sha256']:raise ValueError('Final lock hash changed after generation')
    if digest(MODEL/'checkpoint.pkl')!=lock['checkpoint_sha256']:raise ValueError('Checkpoint differs from the frozen test lock')
    if pre['test_spec_sha256']!=digest(SRC/'frozen_test.json'):raise ValueError('Final protocol differs from the pre-score manifest')
    execution={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'started','scored_once':True,
        'pre_generation_lock_sha256':digest(lock_path),'checkpoint_sha256':lock['checkpoint_sha256'],
        'test_spec_sha256':lock['test_spec_sha256'],'condition_results':{}}
    # Persist the one-pass guard before any model inference is run.
    write(FINAL_DIR/'evaluation_execution.json',execution,exclusive=True);pre['evaluation_started']=True;write(pre_path,pre)
    ck=load_checkpoint(MODEL);condition_names=[]
    for item in lock['conditions']:
        name=item['id'];d=load_dataset(FINAL_DATA/name)
        out=FINAL_DIR/'evaluations'/name
        metrics,_=_save_evaluation(d,ck,out,stream=True);condition_names.append(name)
        execution['condition_results'][name]={'metrics_sha256':digest(out/'metrics.json'),'prediction_ledger_sha256':digest(out/'predictions_label_free.csv'),
            'observation_audit_sha256':digest(out/'observation_audit.csv'),'run_count':len(d.runs)}
        write(FINAL_DIR/'evaluation_execution.json',execution)
    audits=[];predictions=[];episodes=[];physics=[];ev_episodes=[];missions=[];cond_metrics={}
    for name in condition_names:
        out=FINAL_DIR/'evaluations'/name
        a=pd.read_csv(out/'observation_audit.csv');a['condition']=name;audits.append(a)
        p=pd.read_csv(out/'predictions_label_free.csv');p['condition']=name;predictions.append(p)
        ep=pd.read_csv(out/'attack_alert_episodes.csv');
        if len(ep):ep['condition']=name;episodes.append(ep)
        path=out/'evil_twin_alert_episodes.csv'
        if path.exists():
            x=pd.read_csv(path)
            if len(x):x['condition']=name;ev_episodes.append(x)
        path=out/'physics_warning_episodes.csv'
        if path.exists():
            x=pd.read_csv(path)
            if len(x):x['condition']=name;physics.append(x)
        mm=pd.read_csv(out/'per_mission_metrics.csv');mm['condition']=name;missions.append(mm)
        cond_metrics[name]=_load_metrics(name)
    audit=pd.concat(audits,ignore_index=True);audit.attrs['sampling_hz']=5.0;audit.attrs['duration_s']=60.
    all_episodes=pd.concat(episodes,ignore_index=True) if episodes else pd.DataFrame()
    overall=score_ledger(audit,all_episodes.to_dict('records') if len(all_episodes) else [])
    overall['frozen_protocol_sha256']=lock['test_spec_sha256'];overall['pre_generation_lock_sha256']=digest(lock_path)
    overall['checkpoint_sha256']=lock['checkpoint_sha256'];overall['test_condition_count']=len(condition_names)
    overall['test_mission_count']=int(audit.run_id.nunique());overall['test_seed_count']=sum(v['runs'] for v in read(pre_path)['conditions'].values())
    overall['per_condition']={name:{**{k:cond_metrics[name].get(k) for k in (
        'end_to_end_evil_twin_recall','end_to_end_evil_twin_recall_run_clustered_95_ci',
        'conditional_evil_twin_recall','evil_twin_precision','evil_twin_precision_run_clustered_95_ci',
        'benign_scored_row_false_positive_rate','prediction_coverage','benign_mission_alert_rate',
        'false_alert_episodes_per_benign_minute','evil_twin_pre_attack_alert_rate',
        'evil_twin_attack_alert_detection_delay_mean_s','gap_breakdown','evil_twin_confusion')},
        'accuracy':cond_metrics[name]['row_level']['accuracy'],
        'macro_f1':cond_metrics[name]['row_level']['macro_f1'],
        'balanced_accuracy':cond_metrics[name]['row_level']['balanced_accuracy'],
        'per_class':cond_metrics[name]['row_level']['per_class'],
        'scored_confusion_matrix':cond_metrics[name]['row_level']['confusion_matrix']}
        for name in condition_names}
    targets=lock['acceptance_targets'];conditions=overall['per_condition']
    target_results={
        'pooled_end_to_end_evil_twin_recall':overall['end_to_end_evil_twin_recall']>=targets['pooled_end_to_end_evil_twin_recall_min'],
        'every_condition_end_to_end_evil_twin_recall':all(x['end_to_end_evil_twin_recall']>=targets['per_condition_end_to_end_evil_twin_recall_min'] for x in conditions.values()),
        'pooled_evil_twin_precision':overall['evil_twin_precision']>=targets['pooled_evil_twin_precision_min'],
        'every_condition_evil_twin_precision':all(x['evil_twin_precision']>=targets['per_condition_evil_twin_precision_min'] for x in conditions.values()),
        'benign_scored_row_fpr':overall['benign_scored_row_false_positive_rate']<=targets['benign_scored_row_fpr_max'],
        'benign_alert_mission_rate':overall['benign_mission_alert_rate']<=targets['benign_mission_alert_rate_max'],
        'false_alert_episode_rate':overall['false_alert_episodes_per_benign_minute']<=targets['false_alert_episodes_per_benign_minute_max'],
        'evil_twin_pre_attack_alert_rate':overall['evil_twin_pre_attack_alert_rate']<=targets['evil_twin_pre_attack_alert_mission_rate_max'],
        'mean_attack_alert_detection_delay':overall['evil_twin_attack_alert_detection_delay_mean_s'] is not None and overall['evil_twin_attack_alert_detection_delay_mean_s']<=targets['mean_attack_alert_detection_delay_s_max']}
    overall['acceptance_targets']=targets;overall['target_results']=target_results;overall['all_targets_met']=all(target_results.values())
    analysis=FINAL_DIR/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    audit.to_csv(analysis/'all_observation_audit.csv',index=False)
    all_prediction_frame=pd.concat(predictions,ignore_index=True)
    all_prediction_frame.to_csv(analysis/'all_predictions_label_free.csv',index=False)
    latency_ms=all_prediction_frame.inference_latency_s.astype(float)*1000
    overall['prediction_inference_latency_p50_ms']=float(latency_ms.quantile(.5))
    overall['prediction_inference_latency_p95_ms']=float(latency_ms.quantile(.95))
    overall['prediction_inference_latency_max_ms']=float(latency_ms.max())
    overall['prediction_deadline_s']=float(lock['prediction_deadline_s'])
    overall['physics_warning_episodes']=sum(int(m.get('physics_warning_episodes',0)) for m in cond_metrics.values())
    physics_models={}
    for m in cond_metrics.values():
        for name,count in m.get('physics_warning_support_models',{}).items():physics_models[name]=physics_models.get(name,0)+int(count)
    overall['physics_warning_support_models']=physics_models
    missions_frame=pd.concat(missions,ignore_index=True);missions_frame.to_csv(analysis/'per_mission_metrics.csv',index=False)
    ev_missions=missions_frame[missions_frame.scenario.eq('Evil_Twin')]
    overall['evil_twin_missions_at_least_60pct_end_to_end_recall']=int((ev_missions.evil_twin_end_to_end_recall>=.60).sum())
    overall['evil_twin_missions_count']=int(len(ev_missions))
    if len(all_episodes):all_episodes.to_csv(analysis/'attack_alert_episodes.csv',index=False)
    if ev_episodes:pd.concat(ev_episodes,ignore_index=True).to_csv(analysis/'evil_twin_alert_episodes.csv',index=False)
    if physics:pd.concat(physics,ignore_index=True).to_csv(analysis/'physics_warning_episodes.csv',index=False)
    cond_table=[]
    for name,m in cond_metrics.items():
        cond_table.append(dict(condition=name,scheduled_observations=m['scheduled_observations'],prediction_coverage=m['prediction_coverage'],
            conditional_evil_twin_recall=m['conditional_evil_twin_recall'],end_to_end_evil_twin_recall=m['end_to_end_evil_twin_recall'],
            end_to_end_evil_twin_recall_ci_low=m['end_to_end_evil_twin_recall_run_clustered_95_ci'][0],
            end_to_end_evil_twin_recall_ci_high=m['end_to_end_evil_twin_recall_run_clustered_95_ci'][1],
            evil_twin_precision=m['evil_twin_precision'],evil_twin_precision_ci_low=m['evil_twin_precision_run_clustered_95_ci'][0],
            evil_twin_precision_ci_high=m['evil_twin_precision_run_clustered_95_ci'][1],
            macro_f1=m['row_level']['macro_f1'],balanced_accuracy=m['row_level']['balanced_accuracy'],
            benign_scored_row_fpr=m['benign_scored_row_false_positive_rate'],
            benign_alert_mission_rate=m['benign_mission_alert_rate'],false_alert_episodes_per_benign_minute=m['false_alert_episodes_per_benign_minute'],
            evil_twin_pre_attack_alert_rate=m['evil_twin_pre_attack_alert_rate'],evil_twin_alert_delay_mean_s=m['evil_twin_attack_alert_detection_delay_mean_s'],
            gap_breakdown=json.dumps(m['gap_breakdown'],sort_keys=True)))
    cdf=pd.DataFrame(cond_table);cdf.to_csv(analysis/'per_condition_metrics.csv',index=False)
    class_rows=[]
    for name,m in cond_metrics.items():
        for label,metrics in m['row_level']['per_class'].items():
            class_rows.append(dict(condition=name,scenario=label,**metrics))
    pd.DataFrame(class_rows).to_csv(analysis/'per_condition_class_metrics.csv',index=False)
    gap_rows=[]
    for name,m in cond_metrics.items():
        for reason,count in m['gap_breakdown'].items():gap_rows.append(dict(condition=name,reason=reason,count=count))
    pd.DataFrame(gap_rows).to_csv(analysis/'gap_breakdown.csv',index=False)
    cm=np.asarray(overall['scheduled_confusion_matrix']);pd.DataFrame(cm,index=overall['scheduled_confusion_labels'],columns=overall['scheduled_confusion_labels']).to_csv(analysis/'confusion_matrix_end_to_end.csv')
    class_names=list(overall['row_level']['per_class'])
    pd.DataFrame(overall['row_level']['confusion_matrix'],index=class_names,columns=class_names).to_csv(analysis/'confusion_matrix_scored.csv')
    for key in ('per_mission','alert_episodes','scheduled_confusion_matrix','scheduled_confusion_labels'):overall.pop(key,None)
    write(analysis/'metrics.json',overall)
    # Rebuild traceable plots from the saved condition metrics and prediction timelines.
    names=list(cond_metrics);x=np.arange(len(names));fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(x-.18,[cond_metrics[n]['end_to_end_evil_twin_recall'] for n in names],.36,label='End-to-end recall')
    ax.bar(x+.18,[cond_metrics[n]['prediction_coverage'] for n in names],.36,label='Prediction coverage');ax.axhline(.7,color='red',ls='--',lw=1,label='Per-condition recall target')
    ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylim(0,1);ax.legend();ax.set_ylabel('Rate');fig.tight_layout();fig.savefig(analysis/'condition_recall_coverage.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(names,[cond_metrics[n]['benign_mission_alert_rate'] for n in names],color='#bb5544');ax.axhline(targets['benign_mission_alert_rate_max'],color='black',ls='--');ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel('Benign missions with alert');fig.tight_layout();fig.savefig(analysis/'benign_false_alert_rate.png');plt.close(fig)
    condition_html=''.join(f"<tr><td>{r['condition']}</td><td>{r['prediction_coverage']:.1%}</td><td>{r['end_to_end_evil_twin_recall']:.1%}</td><td>{r['evil_twin_precision']:.1%}</td><td>{r['benign_scored_row_fpr']:.1%}</td><td>{r['benign_alert_mission_rate']:.1%}</td><td><a href=\"evaluations/{r['condition']}/index.html\">condition dashboard</a></td></tr>" for r in cond_table)
    dashboard=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ACPDAD — Attack Detection Dashboard</title><style>body{{font:16px system-ui;margin:2rem;max-width:1200px;color:#17212b}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.45rem .65rem;border:1px solid #aab4bf;text-align:left}}img{{max-width:100%}}.sim{{background:#fff2cf;padding:1rem}}.metrics{{display:flex;gap:1rem;flex-wrap:wrap}}.metric{{padding:.8rem 1rem;background:#eef4f7;border-radius:.5rem}}</style></head><body><h1>ACPDAD — Attack Detection Dashboard</h1><p class="sim"><b>Simulated telemetry only.</b> This report is not ArduPilot SITL, captured Wi-Fi, physical-drone, or real-world validation.</p><h2>End-to-end test results</h2><p>Scheduled Evil_Twin observations without a timely classification count as misses. Current prediction coverage is {overall['prediction_coverage']:.1%}; schedule-wide Evil_Twin recall is {overall['end_to_end_evil_twin_recall']:.1%} (run-clustered 95% CI {overall['end_to_end_evil_twin_recall_run_clustered_95_ci'][0]:.1%}–{overall['end_to_end_evil_twin_recall_run_clustered_95_ci'][1]:.1%}). Scored-row conditional recall is {overall['conditional_evil_twin_recall']:.1%}; Evil_Twin precision is {overall['evil_twin_precision']:.1%}; benign scored-row false-positive rate is {overall['benign_scored_row_false_positive_rate']:.1%}.</p><div class="metrics"><div class="metric"><b>Timely predictions</b><br>{overall['timely_predictions']:,} / {overall['scheduled_observations']:,}</div><div class="metric"><b>Evil_Twin TP / FN / FP</b><br>{overall['evil_twin_confusion']['tp']} / {overall['evil_twin_confusion']['fn']} / {overall['evil_twin_confusion']['fp']}</div><div class="metric"><b>Benign alert missions</b><br>{overall['benign_missions_with_alert']} / {overall['benign_mission_count']}</div><div class="metric"><b>False classifier episodes/min</b><br>{overall['false_alert_episodes_per_benign_minute']:.3f}</div><div class="metric"><b>Freshness deadline</b><br>{lock['prediction_deadline_s']:.2f}s after observation time</div></div><h2>Changed-condition results</h2><table><thead><tr><th>Condition</th><th>Coverage</th><th>End-to-end EV recall</th><th>EV precision</th><th>Benign FPR</th><th>Benign alert missions</th><th>Detail</th></tr></thead><tbody>{condition_html}</tbody></table><h2>Traceable evidence</h2><ul><li><a href="IMPLEMENTATION_REPORT.md">Implementation report and frozen target results</a></li><li><a href="analysis/metrics.json">Pooled recalculated metrics</a></li><li><a href="analysis/per_condition_metrics.csv">Condition metrics and run-clustered recall intervals</a></li><li><a href="analysis/per_mission_metrics.csv">Per-mission results</a></li><li><a href="analysis/all_observation_audit.csv">Complete scheduled-observation timeline and gap reasons</a></li><li><a href="analysis/all_predictions_label_free.csv">Label-free prediction ledger</a></li><li><a href="analysis/gap_breakdown.csv">Mutually exclusive prediction-gap counts</a></li><li><a href="analysis/attack_alert_episodes.csv">Confirmed classifier alert episodes with supporting model path</a></li><li><a href="analysis/confusion_matrix_end_to_end.csv">End-to-end confusion matrix including UNAVAILABLE</a></li><li><a href="analysis/cross_verification.csv">Cross-verification checklist</a></li></ul><h2>Plots</h2><img src="analysis/condition_recall_coverage.png" alt="Recall and coverage by condition"><img src="analysis/benign_false_alert_rate.png" alt="Benign false alert rate by condition"></body></html>'''
    (FINAL_DIR/'index.html').write_text(dashboard)
    # Preserve a complete written report with both row-conditional and end-to-end metrics.
    lines=['# ACPDAD V5.4R2 focused system improvement','',
      'All records are simulated. This work does not establish ArduPilot SITL, physical-drone, captured-Wi-Fi, or real-world behavior.','',
      '## Frozen test and implementation','',
      f"The fresh test was generated and evaluated once after freezing the V5.4R2 source, selected checkpoint, settings, seeds and conditions. Test specification SHA-256: `{lock['test_spec_sha256']}`. Pre-generation lock SHA-256: `{digest(lock_path)}`. Checkpoint SHA-256: `{lock['checkpoint_sha256']}`. The V5.4R1 frozen test remains untouched and historical.",'',
      'The primary 22-base/88-temporal feature contract is unchanged. A separately trained three-observation fallback is used only until the primary window is ready after startup or a delivery gap. The validation-selected policy uses fresh K-of-N class evidence, benign hysteresis clearing, a bounded evidence expiry, separate physics warnings and separate communication-health warnings. Physics residuals alone no longer assert an attack. A telemetry expedited-priority queue reduces simulated queueing delay while leaving offered traffic, configured capacity, random loss, jitter, relay/handoff delay and the 450 ms maximum delivery lag unchanged. This is a modeled scheduling assumption, not an observed wireless guarantee.','',
      'Prediction deadline: 0.5 seconds after the current emitted observation time. A delayed offline reconstruction is retained in the ledger but is not counted as timely or allowed to keep an alert active.','',
      '## Gap accounting','',
      f"V5.4R1 historical scheduled Evil_Twin gaps: 1,612 random network losses, 1,386 queue/lag timeouts, 684 post-gap temporal warm-up observations; initial startup accounted for none of the active attack rows. V5.4R1 pre-attack combined-alert sources: physics-only 47 missions, classifier-only 9, both 6.",'',
      f"V5.4R2 gap reasons are mutually exclusive per scheduled sample: `{json.dumps(overall['gap_breakdown'],sort_keys=True)}`. The total unavailable/late count is {sum(overall['gap_breakdown'].values())}; it reconciles against all {overall['scheduled_observations']} scheduled timeline rows. Each observation audit records observation/arrival/prediction time, evidence age, delivery reason and timely/unavailable status. The complete breakdown is in `analysis/gap_breakdown.csv`.",'',
      '## Results','',
      f"Pooled prediction coverage: {overall['prediction_coverage']:.2%} (run-clustered 95% CI {overall['prediction_coverage_run_clustered_95_ci'][0]:.2%}–{overall['prediction_coverage_run_clustered_95_ci'][1]:.2%}). Conditional Evil_Twin recall on timely scored rows: {overall['conditional_evil_twin_recall']:.2%}. End-to-end Evil_Twin recall with every scheduled sample included: {overall['end_to_end_evil_twin_recall']:.2%} (run-clustered 95% CI {overall['end_to_end_evil_twin_recall_run_clustered_95_ci'][0]:.2%}–{overall['end_to_end_evil_twin_recall_run_clustered_95_ci'][1]:.2%}). Confusion counts: TP {overall['evil_twin_confusion']['tp']}, FN {overall['evil_twin_confusion']['fn']}, FP {overall['evil_twin_confusion']['fp']}. Evil_Twin precision: {overall['evil_twin_precision']:.2%} (95% CI {overall['evil_twin_precision_run_clustered_95_ci'][0]:.2%}–{overall['evil_twin_precision_run_clustered_95_ci'][1]:.2%}). Benign scored-row FPR: {overall['benign_scored_row_false_positive_rate']:.2%} (95% CI {overall['benign_fpr_run_clustered_95_ci'][0]:.2%}–{overall['benign_fpr_run_clustered_95_ci'][1]:.2%}).",'',
      '| Condition | Coverage | Conditional EV recall | End-to-end EV recall (run-clustered 95% CI) | EV precision (run-clustered 95% CI) | Benign scored FPR | Benign mission alert rate | False episodes/min | EV pre-alert rate | EV alert delay |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in cond_table:
        lines.append(f"| {row['condition']} | {row['prediction_coverage']:.2%} | {row['conditional_evil_twin_recall']:.2%} | {row['end_to_end_evil_twin_recall']:.2%} ({row['end_to_end_evil_twin_recall_ci_low']:.2%}–{row['end_to_end_evil_twin_recall_ci_high']:.2%}) | {row['evil_twin_precision']:.2%} ({row['evil_twin_precision_ci_low']:.2%}–{row['evil_twin_precision_ci_high']:.2%}) | {row['benign_scored_row_fpr']:.2%} | {row['benign_alert_mission_rate']:.2%} | {row['false_alert_episodes_per_benign_minute']:.3f} | {row['evil_twin_pre_attack_alert_rate']:.2%} | {row['evil_twin_alert_delay_mean_s'] if row['evil_twin_alert_delay_mean_s'] is not None else float('nan'):.3f}s |")
    lines+=['','## Frozen acceptance results','']
    target_values={
        'pooled_end_to_end_evil_twin_recall':overall['end_to_end_evil_twin_recall'],
        'every_condition_end_to_end_evil_twin_recall':{n:v['end_to_end_evil_twin_recall'] for n,v in conditions.items()},
        'pooled_evil_twin_precision':overall['evil_twin_precision'],
        'every_condition_evil_twin_precision':{n:v['evil_twin_precision'] for n,v in conditions.items()},
        'benign_scored_row_fpr':overall['benign_scored_row_false_positive_rate'],
        'benign_alert_mission_rate':overall['benign_mission_alert_rate'],
        'false_alert_episode_rate':overall['false_alert_episodes_per_benign_minute'],
        'evil_twin_pre_attack_alert_rate':overall['evil_twin_pre_attack_alert_rate'],
        'mean_attack_alert_detection_delay':overall['evil_twin_attack_alert_detection_delay_mean_s']}
    target_limits={
        'pooled_end_to_end_evil_twin_recall':f">= {targets['pooled_end_to_end_evil_twin_recall_min']:.1%}",
        'every_condition_end_to_end_evil_twin_recall':f">= {targets['per_condition_end_to_end_evil_twin_recall_min']:.1%} each",
        'pooled_evil_twin_precision':f">= {targets['pooled_evil_twin_precision_min']:.1%}",
        'every_condition_evil_twin_precision':f">= {targets['per_condition_evil_twin_precision_min']:.1%} each",
        'benign_scored_row_fpr':f"<= {targets['benign_scored_row_fpr_max']:.1%}",
        'benign_alert_mission_rate':f"<= {targets['benign_mission_alert_rate_max']:.1%}",
        'false_alert_episode_rate':f"<= {targets['false_alert_episodes_per_benign_minute_max']:.3f}/min",
        'evil_twin_pre_attack_alert_rate':f"<= {targets['evil_twin_pre_attack_alert_mission_rate_max']:.1%}",
        'mean_attack_alert_detection_delay':f"<= {targets['mean_attack_alert_detection_delay_s_max']:.2f}s"}
    for name,ok in target_results.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`: measured `{json.dumps(target_values[name],sort_keys=True)}`; frozen target {target_limits[name]}.")
    lines += ['', '### Five-class scored metrics', '',
        '| Class | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for label,vals in overall['row_level']['per_class'].items():
        lines.append(f"| {label} | {vals['precision']:.2%} | {vals['recall']:.2%} | {vals['f1-score']:.3f} | {int(vals['support'])} |")
    lines += ['', f"Scored-row accuracy {overall['row_level']['accuracy']:.2%}; macro-F1 {overall['row_level']['macro_f1']:.3f}; balanced accuracy {overall['row_level']['balanced_accuracy']:.3f}. Inference p50/p95/max latency: {overall.get('prediction_inference_latency_p50_ms',0):.2f}/{overall.get('prediction_inference_latency_p95_ms',0):.2f}/{overall.get('prediction_inference_latency_max_ms',0):.2f} ms under the recorded sequential streaming run.",
        '', '### Alert source attribution', '',
        f"False classifier alert episode support by model path: `{json.dumps(overall.get('false_alert_episode_support_models',{}),sort_keys=True)}`. Pre-attack alert support by model path: `{json.dumps(overall.get('pre_attack_alert_support_models',{}),sort_keys=True)}`. Physics warnings are logged separately and did not create classifier attack alerts; their episode count is {overall.get('physics_warning_episodes',0)} with support `{json.dumps(overall.get('physics_warning_support_models',{}),sort_keys=True)}`.",
        '']
    lines+=['',f"All targets met: **{'YES' if overall['all_targets_met'] else 'NO'}**. Failed targets remain reported as measured; no test conditions or predictions were changed.",'',
       '## Alert and class details','',
       f"Benign missions with any confirmed classifier attack alert: {overall['benign_missions_with_alert']}/{overall['benign_mission_count']} ({overall['benign_mission_alert_rate']:.2%}); false classifier alert episodes per benign mission-minute: {overall['false_alert_episodes_per_benign_minute']:.3f}; mean false episode duration: {overall['mean_false_alert_episode_duration_s']:.3f}s. Evil_Twin missions with pre-attack alert: {overall['evil_twin_missions_with_pre_attack_alerts']}/{overall['evil_twin_mission_count']} ({overall['evil_twin_pre_attack_alert_rate']:.2%}). Physics warning episodes remain separate and do not independently trigger attack classification. Communication warnings are separate from attack alerts.",'',
       f"{overall['evil_twin_missions_at_least_60pct_end_to_end_recall']}/{overall['evil_twin_missions_count']} Evil_Twin missions reached at least 60% schedule-wide recall; per-mission measurements for all classes and missions, including zero-prediction missions, are retained in `analysis/per_mission_metrics.csv`.",'',
       f"The scheduled-observation confusion matrix includes an `UNAVAILABLE` prediction column: `analysis/confusion_matrix_end_to_end.csv`. Scored-only confusion matrix: `analysis/confusion_matrix_scored.csv`. Per-mission recall, precision, coverage, missing-observation counts and alert measures are in `analysis/per_mission_metrics.csv`.","",
       '## Integrity, verification and limitations','',
       'Training/threshold/alert settings used only train and validation missions. The lock records source, config, model and manifest hashes. All five prespecified conditions and every mission remain in saved artifacts. The simulator is simplified; the priority policy is an assumption, not empirical radio scheduling evidence.','',
       'See `pre_generation_lock.json`, `pre_score_manifest.json`, `evaluation_execution.json`, `cross_verification.csv`, `verification/verify.json`, condition-level audits, label-free ledgers and dashboards.']
    (FINAL_DIR/'IMPLEMENTATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    # Cross verification records checks from the final artifact set and the
    # documented development smoke checks. It is part of the review package.
    cross_rows=[]
    def record(item,passed,evidence,issue=''):
        cross_rows.append(dict(item=item,status='PASS' if passed else 'FAIL',evidence=str(evidence),issue='' if passed else (issue or 'condition was false')))
    record('Frozen source/model/test protocol hashes',
        digest(lock_path)==pre['pre_generation_lock_sha256'] and digest(MODEL/'checkpoint.pkl')==lock['checkpoint_sha256'],
        f"lock={digest(lock_path)}; checkpoint={digest(MODEL/'checkpoint.pkl')}; protocol={lock['test_spec_sha256']}")
    record('Complete scheduled timeline and exclusive gap accounting',
        len(audit)==overall['scheduled_observations'] and overall['gap_breakdown_reconciles'],
        f"audit_rows={len(audit)}; scheduled={overall['scheduled_observations']}; unavailable={sum(overall['gap_breakdown'].values())}; reasons={overall['gap_breakdown']}")
    forbidden={'Target_Label','attack_active','scenario','seed','split','attack_start_index','attack_end_index','ap_identity','true_height_cm','true_vz_down_mps'}
    order_ok=all(g.arrival_time_ns.is_monotonic_increasing for _,g in all_prediction_frame.groupby('run_id',sort=False))
    record('Executed predictions are label-free and arrival ordered',not (forbidden&set(all_prediction_frame.columns)) and order_ok,
        f"forbidden_columns={sorted(forbidden&set(all_prediction_frame.columns))}; arrival_order={order_ok}; rows={len(all_prediction_frame)}")
    asset_paths=[FINAL_DIR/'index.html',analysis/'condition_recall_coverage.png',analysis/'benign_false_alert_rate.png',analysis/'all_observation_audit.csv',analysis/'per_mission_metrics.csv']
    record('Aggregate dashboard and linked image/ledger assets exist',all(p.is_file() for p in asset_paths),'; '.join(str(p.relative_to(ROOT)) if ROOT in p.parents else str(p) for p in asset_paths))
    verification_files=sorted((RESULTS/'verification').glob('verify_*.json'),key=lambda p:p.stat().st_mtime)
    verification=read(verification_files[-1]) if verification_files else {}
    verify_checks=all(verification.get(k) is True for k in ('primary_reconstruction','batch_stream_agreement','run_state_isolation','out_of_order_arrival_rejected','long_gap_window_reset','stale_replay_payload_detected','benign_roaming_present','stale_alert_expiry','stale_evil_twin_evidence_excluded'))
    record('Causal feature, gap, replay, roaming and alert verification',verification.get('status')=='passed' and verify_checks and verification.get('checkpoint_sha256')==lock['checkpoint_sha256'],
        str(verification_files[-1]) if verification_files else 'no V5.4R2 verification JSON found')
    prediction_smoke=RESULTS/'verification'/'prediction_smoke.json'
    smoke=read(prediction_smoke) if prediction_smoke.exists() else {}
    smoke_pass=(smoke.get('status')=='passed' and smoke.get('active_checkpoint_sha256')==lock['checkpoint_sha256'] and
        smoke.get('selected_run_exact') and smoke.get('all_run_ids_match') and smoke.get('unknown_run_rejected') and smoke.get('label_free'))
    record('Selected/all-run CLI mapping and unknown-run rejection',smoke_pass,
        str(prediction_smoke.relative_to(ROOT)) if prediction_smoke.exists() else 'prediction smoke record missing')
    record('All five classes retained in scored metrics',len(class_names)==5 and set(overall['row_level']['per_class'])==set(class_names),
        f"classes={list(overall['row_level']['per_class'])}")
    isolated=pre.get('isolation',{})
    checksums_ok=True
    for condition,details in pre.get('conditions',{}).items():
        data_path=FINAL_DATA/condition
        checksums_ok &= digest(data_path/'run_manifest.csv')==details.get('run_manifest_sha256') and digest(data_path/'checksums.json')==details.get('checksums_sha256')
    record('Final data manifests/checksums and seed/AP isolation',checksums_ok and not isolated.get('prior_seed_overlap') and not isolated.get('prior_ap_overlap'),
        f"seed_overlap={isolated.get('prior_seed_overlap')}; ap_overlap={isolated.get('prior_ap_overlap')}; condition_manifests={checksums_ok}")
    preserved=[ROOT/'results/historical/pre_fallback_evaluation/pre_generation_lock.json',
        ROOT/'results/historical/early_takeover_evaluation/freeze_lock.json']
    record('Historical comparison locks remain present',all(p.is_file() for p in preserved),
        '; '.join(str(p.relative_to(ROOT)) for p in preserved))
    cross=pd.DataFrame(cross_rows);cross.to_csv(FINAL_DIR/'analysis/cross_verification.csv',index=False)
    execution['status']='completed';execution['completed_utc']=datetime.now(timezone.utc).isoformat()
    execution['aggregate_metrics_sha256']=digest(analysis/'metrics.json');execution['report_sha256']=digest(FINAL_DIR/'IMPLEMENTATION_REPORT.md')
    execution['dashboard_sha256']=digest(FINAL_DIR/'index.html');execution['cross_verification_sha256']=digest(FINAL_DIR/'analysis/cross_verification.csv')
    execution['condition_results']={**execution['condition_results'],'aggregate':{'metrics_sha256':digest(analysis/'metrics.json'),'report_sha256':execution['report_sha256']}}
    write(FINAL_DIR/'evaluation_execution.json',execution)
    return {'status':'completed_once','report':str(FINAL_DIR/'IMPLEMENTATION_REPORT.md'),'dashboard':str(FINAL_DIR/'index.html'),
            'metrics':str(analysis/'metrics.json'),'target_results':target_results,'all_targets_met':overall['all_targets_met']}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('render-validation');sub.add_parser('render-final');sub.add_parser('finalize-spec');sub.add_parser('freeze-final');sub.add_parser('lock-generated-final')
    args=parser.parse_args(argv)
    if args.command.startswith('render'):
        print('\n'.join(map(str,render('final' if args.command.endswith('final') else 'validation'))))
    elif args.command=='finalize-spec':print(SRC/'frozen_test.json');finalize_spec_from_validation()
    elif args.command=='freeze-final':print(freeze())
    else:print(lock_generated())

if __name__=='__main__':main()
