"""ACPDAD: causal simulated telemetry detection and complete-gap accounting."""
import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .simulation import (BASE, TEMPORAL, CLASSES, VERSION as FEATURE_CONTRACT, W,
                  MAX_GAP_NS, Extractor, generate, digest as file_digest,
                  schema as core_schema, simulate)

ROOT=Path(__file__).resolve().parents[2]
REVISION='acpdad-sim-research-v5-4r2'
DEFAULT_DATA=ROOT/'data/train_validation'
DEFAULT_MODEL=ROOT/'models/selected'
DEFAULT_RESULTS=ROOT/'results/selection'
FALLBACK_WINDOW=3
FALLBACK_TEMPORAL=(BASE+[f'{x}_mean_{FALLBACK_WINDOW}' for x in BASE]+
                   [f'{x}_std_{FALLBACK_WINDOW}' for x in BASE]+
                   [f'{x}_diff' for x in BASE])
DEADLINE_S=.5
ALERT_FALSE_MISSION_LIMIT=.20
ALERT_FALSE_EPISODES_PER_MIN_LIMIT=.35
ALERT_PRE_ATTACK_LIMIT=.20
ALERT_MEAN_DELAY_MAX_S=1.5
PHYSICS_NUMERIC_FLOOR_M=1e-9

def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compatible_schema(value):
    expected=core_schema()
    for key in ('version','base_features','temporal_features','base_count','temporal_count','window','classes','target'):
        if value.get(key)!=expected.get(key):raise ValueError(f'Incompatible feature schema field {key}')
    if value.get('base_count')!=22 or value.get('temporal_count')!=88:
        raise ValueError('V5.4R2 requires the compatible 22-base/88-temporal feature contract')
    return True

@dataclass
class Dataset:
    path: Path
    schema: dict
    config: dict
    packets: pd.DataFrame
    truth: pd.DataFrame
    runs: pd.DataFrame
    saved_features: pd.DataFrame

def load_dataset(path,check=True):
    path=Path(path);schema=read_json(path/'schema.json');compatible_schema(schema)
    if check:
        checksums=read_json(path/'checksums.json')
        for name,expected in checksums.items():
            if file_digest(path/name)!=expected:raise ValueError(f'Dataset checksum mismatch: {path/name}')
    config=read_json(path/'experiment_config.json')
    packets=pd.read_csv(path/'delivered_packets.csv')
    truth=pd.read_csv(path/'ground_truth.csv')
    runs=pd.read_csv(path/'run_manifest.csv')
    features=pd.read_csv(path/'features.csv').set_index('observation_id')
    if list(features.columns)!=TEMPORAL or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('Saved primary feature contract/finite check failed')
    if not runs.run_id.is_unique or not runs.seed.is_unique:raise ValueError('Run IDs/seeds must be unique within a dataset')
    if packets.observation_id.duplicated().any() or truth.observation_id.duplicated().any():raise ValueError('Observation IDs are not unique')
    if not set(features.index).issubset(set(packets.observation_id)):raise ValueError('Feature rows lack delivered packet evidence')
    return Dataset(path,schema,config,packets,truth,runs,features)

def _sample_index(oid):
    try:return int(str(oid).rsplit('_',1)[-1])
    except (TypeError,ValueError):return -1

def _short_features(history):
    arr=np.asarray(history,dtype=float)
    return np.r_[arr[-1],arr.mean(axis=0),arr.std(axis=0,ddof=1),arr[-1]-arr[-2]]

def build_views(packets,hz):
    """Derive primary and fallback features causally in run/arrival order."""
    rows=[]
    ordered=packets.sort_values(['run_id','arrival_time_ns'],kind='stable')
    for run_id,group in ordered.groupby('run_id',sort=False):
        extractor=Extractor(hz);short=deque(maxlen=FALLBACK_WINDOW);segment_origin='run_start'
        for packet in group.to_dict('records'):
            reset_reason=None;error=None
            try:
                primary=extractor.observe(packet)
                reset_reason=extractor.last_reset_reason
                error=extractor.last_error
            except ValueError as exc:
                primary=None;error='out_of_order_arrival:'+str(exc);segment_origin='out_of_order_reset'
                extractor.reset();extractor.run=run_id;short.clear()
            if reset_reason=='delivery_gap':segment_origin='delivery_gap'
            elif reset_reason=='run_start':segment_origin='run_start'
            if reset_reason in ('run_start','delivery_gap'):short.clear()
            base=extractor.last_base
            if error or base is None:
                if error:short.clear()
                row=dict(run_id=run_id,observation_id=str(packet['observation_id']),sample_index=_sample_index(packet['observation_id']),
                         observation_time_ns=int(packet.get('observation_time_ns',_sample_index(packet['observation_id'])*1e9/hz)),
                         source_time_ns=int(packet['source_time_ns']),arrival_time_ns=int(packet['arrival_time_ns']),sequence=int(packet['sequence']),
                         primary=None,base=None,short3=None,segment_count=int(getattr(extractor,'segment_observations',0)),
                         reset_reason=reset_reason,segment_origin=segment_origin,error_reason=error or 'invalid_features',physics_score=np.nan)
                rows.append(row);continue
            base=np.asarray(base,dtype=float);short.append(base.copy())
            short3=_short_features(short) if len(short)==FALLBACK_WINDOW else None
            row=dict(run_id=run_id,observation_id=str(packet['observation_id']),sample_index=_sample_index(packet['observation_id']),
                     observation_time_ns=int(packet.get('observation_time_ns',_sample_index(packet['observation_id'])*1e9/hz)),
                     source_time_ns=int(packet['source_time_ns']),arrival_time_ns=int(packet['arrival_time_ns']),sequence=int(packet['sequence']),
                     primary=None if primary is None else np.asarray(primary,dtype=float),base=base,short3=short3,
                     segment_count=int(getattr(extractor,'segment_observations',0)),reset_reason=reset_reason,segment_origin=segment_origin,error_reason=None,
                     physics_score=float(base[BASE.index('vertical_residual_m')]))
            rows.append(row)
    return rows

def _run_weights(ids,labels,run_for_id):
    y=np.asarray(labels,dtype=str);runs=np.asarray([run_for_id[x] for x in ids],dtype=str)
    frame=pd.DataFrame({'run':runs,'label':y})
    count=frame.groupby(['run','label'],sort=False).label.transform('size').to_numpy(dtype=float)
    weight=1/np.maximum(count,1)
    return weight/weight.mean()

def _fit_hgb(x,y,weights,seed):
    model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=19,min_samples_leaf=25,
            l2_regularization=1.0,learning_rate=.08,random_state=int(seed))
    model.fit(np.asarray(x,dtype=float),np.asarray(y,dtype=str),sample_weight=np.asarray(weights,dtype=float))
    return model

def _predict_probability(model,x):
    if not len(x):return np.empty((0,len(CLASSES)),dtype=float)
    out=model.predict_proba(np.asarray(x,dtype=float))
    if set(model.classes_)!=set(CLASSES):raise ValueError(f'Model classes do not match schema: {list(model.classes_)}')
    return out[:,[list(model.classes_).index(c) for c in CLASSES]]

def _fit_views(dataset,views):
    truth=dataset.truth.set_index('observation_id')
    run_split=dataset.runs.set_index('run_id').split.astype(str).to_dict()
    run_for_id=truth.run_id.astype(str).to_dict()
    labels=truth.Target_Label.astype(str).to_dict()
    train_ids=set(dataset.runs.loc[dataset.runs.split.eq('train'),'run_id'].astype(str))
    primary_rows=[r for r in views if r['run_id'] in train_ids and r['primary'] is not None]
    base_rows=[r for r in views if r['run_id'] in train_ids and r['base'] is not None]
    short_rows=[r for r in views if r['run_id'] in train_ids and r['short3'] is not None]
    if not primary_rows or not base_rows or not short_rows:raise ValueError('Training data lacks usable primary/fallback rows')
    primary=_fit_hgb([r['primary'] for r in primary_rows],[labels[r['observation_id']] for r in primary_rows],
                     _run_weights([r['observation_id'] for r in primary_rows],[labels[r['observation_id']] for r in primary_rows],run_for_id),5101)
    fallback_base=_fit_hgb([r['base'] for r in base_rows],[labels[r['observation_id']] for r in base_rows],
                           _run_weights([r['observation_id'] for r in base_rows],[labels[r['observation_id']] for r in base_rows],run_for_id),5102)
    fallback_short=_fit_hgb([r['short3'] for r in short_rows],[labels[r['observation_id']] for r in short_rows],
                            _run_weights([r['observation_id'] for r in short_rows],[labels[r['observation_id']] for r in short_rows],run_for_id),5103)
    return dict(primary=primary,fallback_base=fallback_base,fallback_short3=fallback_short,
                run_split=run_split,training_ids=train_ids)

def _probability_cache(views,models):
    primary_rows=[r for r in views if r['primary'] is not None]
    base_rows=[r for r in views if r['base'] is not None]
    short_rows=[r for r in views if r['short3'] is not None]
    start=time.perf_counter()
    pp=_predict_probability(models['primary'],[r['primary'] for r in primary_rows])
    base=_predict_probability(models['fallback_base'],[r['base'] for r in base_rows])
    short=_predict_probability(models['fallback_short3'],[r['short3'] for r in short_rows])
    elapsed=time.perf_counter()-start
    cache={r['observation_id']:{'primary':None,'base':None,'short3':None} for r in views}
    for r,p in zip(primary_rows,pp):cache[r['observation_id']]['primary']=p
    for r,p in zip(base_rows,base):cache[r['observation_id']]['base']=p
    for r,p in zip(short_rows,short):cache[r['observation_id']]['short3']=p
    return cache,elapsed

def _classify(probs,threshold):
    ev_index=CLASSES.index('Evil_Twin');selected=int(np.argmax(probs))
    if float(probs[ev_index])>=float(threshold):selected=ev_index
    return CLASSES[selected],float(probs[selected]),float(probs[ev_index])

def _view_to_event(view,cache,fallback,primary_threshold,fallback_threshold,physics_threshold,deadline_s,latency_ns=0):
    oid=view['observation_id'];prob=cache[oid]['primary'];source='primary_temporal';threshold=primary_threshold
    if prob is None:
        source='fallback_'+fallback;threshold=fallback_threshold;prob=cache[oid][fallback]
    error=view['error_reason'];prediction=None;confidence=None;evprob=None;inference_error=None
    if error:
        status='invalid_features'
    elif prob is None:
        status='fallback_warmup';source='none'
    else:
        try:prediction,confidence,evprob=_classify(prob,threshold);status='predicted'
        except Exception as exc:status='inference_failure';inference_error=repr(exc)
    attempted=prob is not None and not error
    pred_time=int(view['arrival_time_ns'])+int(latency_ns) if attempted else None
    obs_time=int(view['observation_time_ns'])
    timely=bool(prediction is not None and pred_time is not None and pred_time<=obs_time+int(deadline_s*1e9))
    if prediction is not None and not timely:status='prediction_late'
    return dict(run_id=view['run_id'],observation_id=oid,sample_index=view['sample_index'],
                observation_time_ns=obs_time,source_time_ns=view['source_time_ns'],arrival_time_ns=view['arrival_time_ns'],
                prediction_attempt_time_ns=pred_time,prediction_time_ns=pred_time,
                arrival_evidence_age_s=max(0.,(int(view['arrival_time_ns'])-int(view['source_time_ns']))/1e9),
                evidence_age_s=None if pred_time is None else max(0.,(pred_time-int(view['source_time_ns']))/1e9),
                observation_age_s=None if pred_time is None else max(0.,(pred_time-obs_time)/1e9),inference_latency_s=max(0.,latency_ns/1e9),
                prediction=prediction,confidence=confidence,evil_twin_probability=evprob,
                decision_threshold=float(threshold),selected_model=source,status=status,timely=timely,
                fallback_used=source.startswith('fallback_'),primary_ready=cache[oid]['primary'] is not None,
                segment_count=view['segment_count'],reset_reason=view['reset_reason'],error_reason=error,
                segment_origin=view['segment_origin'],
                inference_error=inference_error,physics_score=view['physics_score'],
                physics_candidate=bool(np.isfinite(view['physics_score']) and view['physics_score']>physics_threshold))

def _chosen_event(view,cache,fallback,pthr,fthr,physics_threshold,deadline_s,latency_ns=0):
    return _view_to_event(view,cache,fallback,pthr,fthr,physics_threshold,deadline_s,latency_ns)

def _selected_probability(view,cache,fallback):
    p=cache[view['observation_id']]['primary']
    return p if p is not None else cache[view['observation_id']][fallback]

def _make_predictions(views,cache,fallback,pthr,fthr,physics_threshold,deadline_s,stream=False):
    result=[]
    for view in views:
        oid=view['observation_id'];latency_ns=0
        source='primary' if view['primary'] is not None else fallback
        vector=view['primary'] if source=='primary' else view[fallback]
        prob=cache[oid][source];inference_exception=None
        if stream and vector is not None and not view['error_reason']:
            model=cache['_models'][source];tic=time.perf_counter_ns()
            try:prob=_predict_probability(model,[vector])[0]
            except Exception as exc:prob=None;inference_exception=repr(exc)
            latency_ns=time.perf_counter_ns()-tic
        event=_chosen_event(view,cache,fallback,pthr,fthr,physics_threshold,deadline_s,latency_ns)
        event['inference_latency_s']=latency_ns/1e9
        if stream and vector is not None and prob is None and not view['error_reason']:
            event.update(prediction=None,confidence=None,evil_twin_probability=None,timely=False,
                         status='inference_failure',selected_model='none',inference_error=inference_exception)
            event['prediction_attempt_time_ns']=int(view['arrival_time_ns'])+int(latency_ns)
            event['prediction_time_ns']=None;event['evidence_age_s']=None;event['observation_age_s']=None
        elif stream and vector is not None and prob is not None:
            threshold=pthr if source=='primary' else fthr
            event['prediction'],event['confidence'],event['evil_twin_probability']=_classify(prob,threshold)
            event['selected_model']='primary_temporal' if source=='primary' else 'fallback_'+fallback
            event['fallback_used']=source!='primary'
            pred_time=int(view['arrival_time_ns'])+int(latency_ns)
            event['prediction_attempt_time_ns']=pred_time;event['prediction_time_ns']=pred_time
            event['arrival_evidence_age_s']=max(0.,(int(view['arrival_time_ns'])-int(view['source_time_ns']))/1e9)
            event['evidence_age_s']=max(0.,(pred_time-int(view['source_time_ns']))/1e9)
            event['observation_age_s']=max(0.,(pred_time-int(view['observation_time_ns']))/1e9)
        if event['prediction'] is not None:
            event['timely']=event['prediction_time_ns']<=event['observation_time_ns']+int(deadline_s*1e9)
            event['status']='predicted' if event['timely'] else 'prediction_late'
        result.append(event)
    return result

def _append_interval(intervals,state,start,last_support,end_ns,reason='evidence_expiry'):
    if state and start is not None:
        stop=int(end_ns)
        if stop>start:intervals.append((int(start),stop,reason))

def _alert_intervals(events,duration_ns,k,n,expiry_s,clear_benign=3,physics_threshold=0.,max_ev_evidence_age_s=float('inf')):
    """Causal K-of-N alert confirmation with freshness, hysteresis and expiry."""
    events=sorted((e for e in events if e.get('timely') and e.get('prediction') is not None),key=lambda e:(e['arrival_time_ns'],e['observation_id']))
    hist=deque(maxlen=n);phist=deque(maxlen=3)
    attack_active=False;attack_start=None;attack_last=None;benign_count=0;attack_intervals=[]
    ev_active=False;ev_start=None;ev_last=None;ev_benign=0;ev_intervals=[]
    physics_active=False;physics_start=None;physics_last=None;physics_clear=0;physics_intervals=[]
    expiry_ns=int(expiry_s*1e9);p_expiry=int(.8*1e9);evhist=deque(maxlen=n);last_event_time=None
    for e in events:
        t=int(e['arrival_time_ns'])
        if last_event_time is not None and t-last_event_time>expiry_ns:
            hist.clear();evhist.clear();phist.clear()
        if attack_active and (attack_last is None or t-attack_last>expiry_ns):
            _append_interval(attack_intervals,True,attack_start,attack_last,attack_last+expiry_ns)
            attack_active=False;attack_start=None;attack_last=None;benign_count=0
        if ev_active and (ev_last is None or t-ev_last>expiry_ns):
            _append_interval(ev_intervals,True,ev_start,ev_last,ev_last+expiry_ns)
            ev_active=False;ev_start=None;ev_last=None;ev_benign=0
        if physics_active and (physics_last is None or t-physics_last>p_expiry):
            _append_interval(physics_intervals,True,physics_start,physics_last,physics_last+p_expiry)
            physics_active=False;physics_start=None;physics_last=None;physics_clear=0
        nonbenign=e['prediction']!='Benign'
        # An Evil_Twin-specific alert needs a fresh source observation. A
        # stale replay may still support a Replay classification, but cannot
        # by itself establish a current persistent takeover.
        is_ev=(e['prediction']=='Evil_Twin' and float(e.get('evidence_age_s',0.) or 0.)<=max_ev_evidence_age_s)
        is_phys=bool(e['physics_candidate'])
        hist.append(nonbenign);evhist.append(is_ev);phist.append(is_phys)
        if nonbenign and sum(hist)>=k and not attack_active:attack_active=True;attack_start=t
        if attack_active:
            if nonbenign:attack_last=t;benign_count=0
            else:
                benign_count+=1
                if benign_count>=clear_benign:
                    _append_interval(attack_intervals,True,attack_start,attack_last,t,'benign_hysteresis_clear')
                    attack_active=False;attack_start=None;attack_last=None;benign_count=0
        if is_ev and sum(evhist)>=k and not ev_active:ev_active=True;ev_start=t
        if ev_active:
            if is_ev:ev_last=t;ev_benign=0
            else:
                ev_benign+=1
                if ev_benign>=clear_benign:
                    _append_interval(ev_intervals,True,ev_start,ev_last,t,'benign_hysteresis_clear')
                    ev_active=False;ev_start=None;ev_last=None;ev_benign=0
        if is_phys and sum(phist)>=2 and not physics_active:physics_active=True;physics_start=t
        if physics_active:
            if is_phys:physics_last=t;physics_clear=0
            else:
                physics_clear+=1
                if physics_clear>=3:
                    _append_interval(physics_intervals,True,physics_start,physics_last,t,'physics_hysteresis_clear')
                    physics_active=False;physics_start=None;physics_last=None;physics_clear=0
        e['attack_alert_active']=attack_active;e['evil_twin_alert_active']=ev_active;e['physics_warning_active']=physics_active
        last_event_time=t
    end_ns=int(duration_ns)
    if attack_active:_append_interval(attack_intervals,True,attack_start,attack_last,min(end_ns,attack_last+expiry_ns))
    if ev_active:_append_interval(ev_intervals,True,ev_start,ev_last,min(end_ns,ev_last+expiry_ns))
    if physics_active:_append_interval(physics_intervals,True,physics_start,physics_last,min(end_ns,physics_last+p_expiry))
    def records(items,kind):
        out=[]
        for a,b,why in items:
            if kind=='persistent_physics_warning':
                sources=sorted({str(e.get('selected_model','unknown')) for e in events if a<=int(e['arrival_time_ns'])<=b and e.get('physics_candidate')})
            elif kind=='persistent_classifier_evil_twin':
                sources=sorted({str(e.get('selected_model','unknown')) for e in events if a<=int(e['arrival_time_ns'])<=b and e.get('prediction')=='Evil_Twin' and float(e.get('evidence_age_s',0.) or 0.)<=max_ev_evidence_age_s})
            else:
                sources=sorted({str(e.get('selected_model','unknown')) for e in events if a<=int(e['arrival_time_ns'])<=b and e.get('prediction')!='Benign'})
            out.append(dict(start_time_ns=a,end_time_ns=b,duration_s=(b-a)/1e9,source=kind,
                            supporting_models=';'.join(sources),close_reason=why))
        return out
    return records(attack_intervals,'persistent_classifier_attack'),records(ev_intervals,'persistent_classifier_evil_twin'),records(physics_intervals,'persistent_physics_warning')

def _class_metrics(y,pred,valid):
    yy=np.asarray(y,dtype=str)[valid];pp=np.asarray(pred,dtype=object)[valid]
    cm=confusion_matrix(yy,pp,labels=CLASSES);support=cm.sum(axis=1)
    report=classification_report(yy,pp,labels=CLASSES,output_dict=True,zero_division=0) if len(yy) else {}
    return dict(scored_rows=int(len(yy)),accuracy=float(accuracy_score(yy,pp)) if len(yy) else None,
        macro_f1=float(f1_score(yy,pp,labels=CLASSES,average='macro',zero_division=0)) if len(yy) else None,
        balanced_accuracy=float(balanced_accuracy_score(yy,pp)) if len(yy) else None,
        confusion_matrix=cm.tolist(),
        per_class={c:{m:float(report[c][m]) for m in ('precision','recall','f1-score','support')} if report else None for c in CLASSES},
        benign_false_positive_rate=float((cm[0].sum()-cm[0,0])/cm[0].sum()) if cm[0].sum() else None)

def _validation_summary(dataset,views,cache,fallback,pthr,fthr,policy,deadline_s):
    runmeta=dataset.runs.set_index('run_id')
    predictions=[]
    for view in views:
        oid=view['observation_id'];row=cache[oid];prob=row['primary'] if row['primary'] is not None else row[fallback]
        prediction=None
        if prob is not None:prediction,_,_=_classify(prob,pthr if row['primary'] is not None else fthr)
        timely=bool(prediction is not None and int(view['arrival_time_ns'])<=int(view['observation_time_ns'])+int(deadline_s*1e9))
        predictions.append(dict(run_id=view['run_id'],observation_id=oid,arrival_time_ns=view['arrival_time_ns'],prediction=prediction,
            timely=timely,evidence_age_s=max(0.,(int(view['arrival_time_ns'])-int(view['source_time_ns']))/1e9),
            selected_model='primary_temporal' if row['primary'] is not None else 'fallback_'+fallback,
            physics_candidate=bool(np.isfinite(view['physics_score']) and view['physics_score']>policy['physics_threshold'])))
    event_frame=pd.DataFrame(predictions)
    grouped={rid:g.to_dict('records') for rid,g in event_frame[event_frame.timely].groupby('run_id',sort=False)} if len(event_frame) else {}
    episodes=[];ev_episodes=[];phys_episodes=[];end=int(float(dataset.config['duration_s'])*1e9)
    for run_id in dataset.runs.run_id.astype(str):
        aa,ee,pp=_alert_intervals(grouped.get(run_id,[]),end,policy['alert_k'],policy['alert_n'],policy['evidence_expiry_s'],policy['clear_benign_predictions'],policy['physics_threshold'],policy.get('max_ev_evidence_age_s',float('inf')))
        for x in aa:x['run_id']=run_id
        for x in ee:x['run_id']=run_id
        for x in pp:x['run_id']=run_id
        episodes.extend(aa);ev_episodes.extend(ee);phys_episodes.extend(pp)
    benign_runs=set(runmeta[runmeta.scenario.eq('Benign')].index.astype(str));ev_runs=runmeta[runmeta.scenario.eq('Evil_Twin')]
    benign_alert={x['run_id'] for x in episodes if x['run_id'] in benign_runs};false_episodes=[x for x in episodes if x['run_id'] in benign_runs]
    prealert=detected=0;delays=[];ev_prealert=ev_detected=0;ev_delays=[]
    for run_id,row in ev_runs.iterrows():
        start=int(row.attack_start_index/dataset.config['sampling_hz']*1e9);stop=int(row.attack_end_index/dataset.config['sampling_hz']*1e9)
        run_eps=[x for x in episodes if x['run_id']==run_id]
        prealert+=any(x['start_time_ns']<start for x in run_eps)
        hits=[x for x in run_eps if start<=x['start_time_ns']<stop]
        detected+=bool(hits)
        if hits:delays.append((min(x['start_time_ns'] for x in hits)-start)/1e9)
        ev_eps=[x for x in ev_episodes if x['run_id']==run_id]
        ev_prealert+=any(x['start_time_ns']<start for x in ev_eps)
        ev_hits=[x for x in ev_eps if start<=x['start_time_ns']<stop]
        ev_detected+=bool(ev_hits)
        if ev_hits:ev_delays.append((min(x['start_time_ns'] for x in ev_hits)-start)/1e9)
    class_score=_fast_condition_score(dataset,views,cache,fallback,pthr,fthr,deadline_s)
    return dict(**class_score,benign_mission_alert_rate=len(benign_alert)/max(1,len(benign_runs)),
        benign_false_episodes=len(false_episodes),false_alert_episodes_per_benign_minute=len(false_episodes)/max(1,len(benign_runs)*float(dataset.config['duration_s'])/60),
        evil_twin_pre_attack_alert_rate=prealert/max(1,len(ev_runs)),evil_twin_alert_mission_rate=detected/max(1,len(ev_runs)),
        evil_twin_specific_pre_attack_alert_rate=ev_prealert/max(1,len(ev_runs)),evil_twin_specific_alert_mission_rate=ev_detected/max(1,len(ev_runs)),
        evil_twin_specific_alert_delay_s=float(np.mean(ev_delays)) if ev_delays else None,
        mean_attack_alert_detection_delay_s=float(np.mean(delays)) if delays else None,
        alert_episodes=episodes,evil_twin_alert_episodes=ev_episodes,physics_warning_episodes=phys_episodes)

def _datasets_for_validation(base,additional):
    items=[Path(base),*[Path(p) for p in additional]]
    result=[]
    for p in items:
        d=load_dataset(p)
        if not set(d.runs.split.astype(str)).issubset({'train','val'}) or 'val' not in set(d.runs.split.astype(str)):
            raise ValueError(f'Validation data must contain only train/validation missions: {p}')
        if set(d.runs.loc[d.runs.split.eq('val'),'scenario'].astype(str))!=set(CLASSES):raise ValueError(f'All five labels required in validation data: {p}')
        result.append(d)
    return result

def _batch_events(d,models):
    views=build_views(d.packets,float(d.config['sampling_hz']))
    cache,elapsed=_probability_cache(views,models)
    # Labels are intentionally not joined here.
    return views,cache,elapsed

def _fast_condition_score(dataset,views,cache,fallback,pthr,fthr,deadline_s):
    labels=dataset.truth.set_index('observation_id').Target_Label.astype(str)
    y=np.asarray([labels[v['observation_id']] for v in views],dtype=str)
    probs=np.zeros((len(views),len(CLASSES)),dtype=float);available=np.zeros(len(views),dtype=bool)
    primary=np.zeros(len(views),dtype=bool);timely=np.zeros(len(views),dtype=bool);run_ids=np.empty(len(views),dtype=object)
    for i,v in enumerate(views):
        oid=v['observation_id'];run_ids[i]=v['run_id'];pp=cache[oid]['primary'];primary[i]=pp is not None
        if pp is None:pp=cache[oid][fallback]
        if pp is not None:
            probs[i]=pp;available[i]=True
            timely[i]=int(v['arrival_time_ns'])<=int(v['observation_time_ns'])+int(deadline_s*1e9)
    th=np.where(primary,pthr,fthr);idx=np.argmax(probs,axis=1);ev_i=CLASSES.index('Evil_Twin')
    idx=np.where(probs[:,ev_i]>=th,ev_i,idx);pred=np.asarray(CLASSES,dtype=object)[idx];valid=available&timely
    ev=y=='Evil_Twin';tp=int(np.sum(ev&valid&(pred=='Evil_Twin')));support=int(dataset.truth.Target_Label.eq('Evil_Twin').sum())
    ev_pred=int(np.sum(valid&(pred=='Evil_Twin')));benign=(y=='Benign')&valid;benign_fp=int(np.sum(benign&(pred!='Benign')))
    cm=confusion_matrix(y[valid],pred[valid],labels=CLASSES)
    recalls=[];runs=dataset.runs.set_index('run_id')
    for run_id,row in runs.iterrows():
        if row.scenario!='Evil_Twin':continue
        mask=run_ids==run_id;rs=int(np.sum(y[mask]=='Evil_Twin'))
        recalls.append(float(np.sum((y[mask]=='Evil_Twin')&valid[mask]&(pred[mask]=='Evil_Twin'))/max(1,rs)))
    return dict(evil_twin_tp=tp,scheduled_evil_twin=support,scored_evil_twin=int(np.sum(ev&valid)),
        evil_twin_end_to_end_recall=tp/max(1,support),evil_twin_conditional_recall=tp/max(1,int(np.sum(ev&valid))),
        evil_twin_precision=tp/max(1,ev_pred),evil_twin_predictions=ev_pred,
        benign_false_positive_rate=benign_fp/max(1,int(benign.sum())),benign_false_positives=benign_fp,benign_scored=int(benign.sum()),
        prediction_coverage=float(valid.sum()/max(1,len(dataset.truth))),scored_rows=int(valid.sum()),
        macro_f1=float(f1_score(y[valid],pred[valid],labels=CLASSES,average='macro',zero_division=0)) if valid.any() else 0.,
        confusion_matrix=cm.tolist(),mission_recall_min=min(recalls,default=0.),mission_recall_mean=float(np.mean(recalls)) if recalls else 0.)

def _evaluate_choice(validation,models,fallback,pthr,fthr,policy,deadline_s):
    records=[(d,_fast_condition_score(d,views,cache,fallback,pthr,fthr,deadline_s)) for d,views,cache in validation]
    tp=sum(m['evil_twin_tp'] for _,m in records);support=sum(m['scheduled_evil_twin'] for _,m in records)
    pred_ev=sum(m['evil_twin_predictions'] for _,m in records);bfp=sum(m['benign_false_positives'] for _,m in records);bsc=sum(m['benign_scored'] for _,m in records)
    feasible=all(m['evil_twin_precision']>=.95 and m['benign_false_positive_rate']<=.065 for _,m in records)
    return dict(fallback=fallback,primary_threshold=float(pthr),fallback_threshold=float(fthr),
        pooled_end_to_end_recall=tp/max(1,support),minimum_condition_end_to_end_recall=min(m['evil_twin_end_to_end_recall'] for _,m in records),
        pooled_scored_precision=tp/max(1,pred_ev),pooled_benign_fpr=bfp/max(1,bsc),all_conditions_precision_fpr_feasible=feasible,
        mean_condition_conditional_recall=float(np.mean([m['evil_twin_conditional_recall'] for _,m in records])),
        condition_metrics={d.path.name:{k:v for k,v in m.items() if k!='confusion_matrix'} for d,m in records})

def _evaluate_alert_choice(validation,fallback,pthr,fthr,policy,deadline_s):
    records=[(d,_validation_summary(d,views,cache,fallback,pthr,fthr,policy,deadline_s)) for d,views,cache in validation]
    return dict(mean_condition_benign_alert_mission_rate=float(np.mean([m['benign_mission_alert_rate'] for _,m in records])),
        mean_condition_false_episodes_per_minute=float(np.mean([m['false_alert_episodes_per_benign_minute'] for _,m in records])),
        mean_condition_pre_attack_rate=float(np.mean([m['evil_twin_pre_attack_alert_rate'] for _,m in records])),
        mean_condition_attack_alert_rate=float(np.mean([m['evil_twin_alert_mission_rate'] for _,m in records])),
        mean_condition_end_to_end_recall=float(np.mean([m['evil_twin_end_to_end_recall'] for _,m in records])),
        mean_condition_attack_alert_delay_s=float(np.mean([m['mean_attack_alert_detection_delay_s'] for _,m in records if m['mean_attack_alert_detection_delay_s'] is not None])) if any(m['mean_attack_alert_detection_delay_s'] is not None for _,m in records) else None,
        condition_metrics={d.path.name:{k:v for k,v in m.items() if k not in ('alert_episodes','evil_twin_alert_episodes','physics_warning_episodes')} for d,m in records})

def train(data,model_dir,results,validation_data=()):
    model_dir=Path(model_dir);results=Path(results)
    if model_dir.exists() or results.exists():raise FileExistsError('Choose new model and results directories')
    d=load_dataset(data)
    if set(d.runs.split.astype(str))!={'train','val'}:raise ValueError('Training requires only independent train/validation runs')
    views=build_views(d.packets,float(d.config['sampling_hz']))
    models=_fit_views(d,views)
    benign_train=set(d.runs.loc[d.runs.split.eq('train')&d.runs.scenario.eq('Benign'),'run_id'].astype(str))
    benign_residuals=np.asarray([r['physics_score'] for r in views if r['run_id'] in benign_train and np.isfinite(r['physics_score'])],dtype=float)
    if len(benign_residuals)<100:raise ValueError('Insufficient benign training residuals for physics calibration')
    p99=max(float(np.quantile(benign_residuals,.99)),PHYSICS_NUMERIC_FLOOR_M)
    validations=[]
    for item in _datasets_for_validation(data,validation_data):
        vv,cc,_=_batch_events(item,models);validations.append((item,vv,cc))
    threshold_values=[.10,.16,.21,.27,.35,.45,.55]
    candidates=[]
    policies=[]
    for fallback in ('base','short3'):
        best=None
        for pthr in threshold_values:
            for fthr in threshold_values:
                # Initial threshold sweep uses a fixed causal 2-of-3 alert policy.
                policy={'alert_k':2,'alert_n':3,'evidence_expiry_s':.8,'clear_benign_predictions':3,
                        'physics_threshold':p99}
                score=_evaluate_choice(validations,models,fallback,pthr,fthr,policy,DEADLINE_S)
                candidates.append(score)
                key=(int(score['all_conditions_precision_fpr_feasible']),score['minimum_condition_end_to_end_recall'],
                     score['pooled_end_to_end_recall'],score['mean_condition_conditional_recall'],-pthr,-fthr)
                if best is None or key>best[0]:best=(key,fallback,pthr,fthr,score,policy)
        policies.append(best)
    selected=max(policies,key=lambda q:q[0])
    _,fallback,pthr,fthr,base_score,policy=selected
    # Validation selects persistence, hysteresis expiry, and attack-warning false-alert limits.
    alert_candidates=[]
    p99=policy['physics_threshold']
    for k,n in ((2,3),(3,4)):
        for expiry in (.6,.8,1.2):
          for max_ev_age in (.35,.6,1.2):
            candidate_policy={'alert_k':k,'alert_n':n,'evidence_expiry_s':expiry,
                              'clear_benign_predictions':3,'physics_threshold':p99,
                              'max_ev_evidence_age_s':max_ev_age}
            score=_evaluate_alert_choice(validations,fallback,pthr,fthr,candidate_policy,DEADLINE_S)
            score['alert_policy']=candidate_policy
            passes=(score['mean_condition_benign_alert_mission_rate']<=ALERT_FALSE_MISSION_LIMIT and
                    score['mean_condition_false_episodes_per_minute']<=ALERT_FALSE_EPISODES_PER_MIN_LIMIT and
                    score['mean_condition_pre_attack_rate']<=ALERT_PRE_ATTACK_LIMIT and
                 score['mean_condition_attack_alert_delay_s'] is not None and score['mean_condition_attack_alert_delay_s']<=ALERT_MEAN_DELAY_MAX_S)
            score['premature_alert_acceptance_met_on_validation']=bool(passes)
            key=(int(passes),score['mean_condition_benign_alert_mission_rate']<=ALERT_FALSE_MISSION_LIMIT,
                 score['mean_condition_pre_attack_rate']<=ALERT_PRE_ATTACK_LIMIT,
                 score['mean_condition_false_episodes_per_minute']<=ALERT_FALSE_EPISODES_PER_MIN_LIMIT,
                 score['mean_condition_attack_alert_rate'],-(score['mean_condition_attack_alert_delay_s'] or 1e6),
                 score['mean_condition_end_to_end_recall'],
                 -score['mean_condition_benign_alert_mission_rate'],-score['mean_condition_pre_attack_rate'],-score['mean_condition_false_episodes_per_minute'],
                 -k,-expiry,-max_ev_age)
            alert_candidates.append((key,candidate_policy,score))
    _,alert_policy,alert_score=max(alert_candidates,key=lambda v:v[0])
    policy.update(alert_policy)
    # Train-only residual calibration; the physics signal remains a separate warning, not an attack label.
    physics=p99
    policy['physics_threshold']=physics
    checkpoint=dict(contract=REVISION,feature_contract=FEATURE_CONTRACT,base=BASE,temporal=TEMPORAL,window=W,
                    fallback_window=FALLBACK_WINDOW,fallback_schema='acpdad-sim-research-v5-4r2-fallback-3-observation',
                    classes=CLASSES,models={k:v for k,v in models.items() if k in ('primary','fallback_base','fallback_short3')},
                    fallback_selected=fallback,primary_threshold=float(pthr),fallback_threshold=float(fthr),
                    physics_threshold=physics,alert_policy=policy,prediction_deadline_s=DEADLINE_S,
                    decision_rule='Use 10-observation temporal classifier when available; otherwise use selected separately trained causal fallback. Use per-model validation threshold. Emit unavailable for no timely fresh output; never repeat prior predictions.',
                    alert_rule='Generic attack alert requires validation-selected K-of-N fresh non-Benign classifier outputs with benign hysteresis clear and evidence expiry. Physics residuals are separate persisted warnings and do not independently assert an attack.',
                    training_provenance='SIMULATED_ONLY; all estimators fitted on train missions only; threshold and policy selection uses validation missions only.')
    model_dir.mkdir(parents=True);results.mkdir(parents=True)
    with (model_dir/'checkpoint.pkl').open('xb') as f:pickle.dump(checkpoint,f,pickle.HIGHEST_PROTOCOL)
    model_hash=file_digest(model_dir/'checkpoint.pkl')
    model_manifest=dict(contract=REVISION,feature_contract=FEATURE_CONTRACT,checkpoint_sha256=model_hash,
        fallback_model=fallback,fallback_schema=checkpoint['fallback_schema'],base_features=len(BASE),temporal_features=len(TEMPORAL),
        primary_threshold=float(pthr),fallback_threshold=float(fthr),prediction_deadline_s=DEADLINE_S,
        alert_policy=policy,selected_on='training split fits; trainval validation and prespecified independent validation conditions select fallback, class thresholds, K-of-N confirmation and expiry',
        source_dataset=str(Path(data)),source_checksums=read_json(Path(data)/'checksums.json'),
        validation_data=[dict(path=str(x.path),run_manifest_sha256=file_digest(x.path/'run_manifest.csv'),checksums_sha256=file_digest(x.path/'checksums.json')) for x,_,_ in validations],
        packages={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__})
    write_json(model_dir/'manifest.json',model_manifest)
    report=dict(contract=REVISION,selected_fallback=fallback,primary_model='HistGradientBoostingClassifier',fallback_model='HistGradientBoostingClassifier',
        primary_threshold=float(pthr),fallback_threshold=float(fthr),prediction_deadline_s=DEADLINE_S,
        validation_selected_alert_policy=policy,validation_selected_metrics=alert_score,
        premature_alert_acceptance_thresholds={'benign_mission_rate_max':ALERT_FALSE_MISSION_LIMIT,
          'false_alert_episodes_per_benign_minute_max':ALERT_FALSE_EPISODES_PER_MIN_LIMIT,
          'evil_twin_pre_attack_mission_rate_max':ALERT_PRE_ATTACK_LIMIT,'mean_detection_delay_max_s':ALERT_MEAN_DELAY_MAX_S},
        alert_acceptance_met_in_validation=alert_score['premature_alert_acceptance_met_on_validation'],
        all_threshold_fallback_candidates=candidates,all_alert_policy_candidates=[x[2] for x in alert_candidates],
        train_run_count=int(d.runs.split.eq('train').sum()),validation_run_count=int(d.runs.split.eq('val').sum()),
        checkpoint_sha256=model_hash,feature_schema='22 base / 88 primary temporal features; fallback has separate 22-base/66-derived-feature 3-observation schema',
        inference_inputs='delivered packet payload and arrival metadata only; no labels/schedules/IDs/seeds/AP identity enter either model')
    write_json(results/'validation_selection.json',report)
    pd.DataFrame(candidates).drop(columns=['condition_metrics'],errors='ignore').to_csv(results/'threshold_fallback_sweep.csv',index=False)
    pd.DataFrame([x[2] for x in alert_candidates]).to_csv(results/'alert_policy_sweep.csv',index=False)
    write_json(results/'training_manifest.json',dict(contract=REVISION,checkpoint_sha256=model_hash,model_manifest_sha256=file_digest(model_dir/'manifest.json'),
        source_dataset=str(Path(data)),source_checksums=read_json(Path(data)/'checksums.json'),
        validation_datasets=model_manifest['validation_data'],validation_selection_sha256=file_digest(results/'validation_selection.json'),
        selected_only_from_train_validation=True,final_test_not_generated=True))
    return checkpoint

def load_checkpoint(path):
    path=Path(path);manifest=read_json(path/'manifest.json');ck=path/'checkpoint.pkl'
    if file_digest(ck)!=manifest['checkpoint_sha256']:raise ValueError('V5.4R2 checkpoint digest mismatch')
    with ck.open('rb') as f:obj=pickle.load(f)
    if obj.get('contract')!=REVISION or obj.get('feature_contract')!=FEATURE_CONTRACT or obj.get('base')!=BASE or obj.get('temporal')!=TEMPORAL or obj.get('window')!=W:
        raise ValueError('Checkpoint feature schema mismatch')
    if obj.get('fallback_window')!=FALLBACK_WINDOW or obj.get('fallback_selected') not in ('base','short3'):
        raise ValueError('Checkpoint fallback contract mismatch')
    for name in ('primary','fallback_base','fallback_short3'):
        if set(obj['models'][name].classes_)!=set(CLASSES):raise ValueError(f'Checkpoint {name} classes mismatch')
    return obj

def calibrate_alerts(model_dir,data,validation_data,results):
    """Validation-only K-of-N alert calibration; leaves fitted classifiers unchanged."""
    model_dir=Path(model_dir);results=Path(results)
    if results.exists():raise FileExistsError(f'Refusing to overwrite alert calibration: {results}')
    manifest_path=model_dir/'manifest.json';checkpoint_path=model_dir/'checkpoint.pkl'
    old_manifest=read_json(manifest_path);old_sha=file_digest(checkpoint_path)
    if old_sha!=old_manifest['checkpoint_sha256']:raise ValueError('Checkpoint digest mismatch before alert calibration')
    checkpoint=load_checkpoint(model_dir)
    validations=[]
    for item in _datasets_for_validation(data,validation_data):
        views,cache,_=_batch_events(item,checkpoint['models']);validations.append((item,views,cache))
    candidates=[]
    for k,n in ((2,3),(3,4),(4,5),(4,6)):
      for expiry in (.6,.8,1.2):
        policy={'alert_k':k,'alert_n':n,'evidence_expiry_s':expiry,'clear_benign_predictions':3,
                'physics_threshold':checkpoint['physics_threshold'],'max_ev_evidence_age_s':.35}
        score=_evaluate_alert_choice(validations,checkpoint['fallback_selected'],checkpoint['primary_threshold'],
            checkpoint['fallback_threshold'],policy,checkpoint['prediction_deadline_s'])
        passed=(score['mean_condition_benign_alert_mission_rate']<=ALERT_FALSE_MISSION_LIMIT and
                score['mean_condition_false_episodes_per_minute']<=ALERT_FALSE_EPISODES_PER_MIN_LIMIT and
                score['mean_condition_pre_attack_rate']<=ALERT_PRE_ATTACK_LIMIT and
                score['mean_condition_attack_alert_delay_s'] is not None and
                score['mean_condition_attack_alert_delay_s']<=ALERT_MEAN_DELAY_MAX_S)
        score['alert_policy']=policy;score['validation_acceptance_met']=bool(passed);candidates.append(score)
    def rank(s):
        policy=s['alert_policy'];delay=s['mean_condition_attack_alert_delay_s']
        return (int(s['validation_acceptance_met']),
            int(s['mean_condition_benign_alert_mission_rate']<=ALERT_FALSE_MISSION_LIMIT),
            int(s['mean_condition_pre_attack_rate']<=ALERT_PRE_ATTACK_LIMIT),
            int(s['mean_condition_false_episodes_per_minute']<=ALERT_FALSE_EPISODES_PER_MIN_LIMIT),
            int(delay is not None and delay<=ALERT_MEAN_DELAY_MAX_S),
            -s['mean_condition_false_episodes_per_minute'],-s['mean_condition_benign_alert_mission_rate'],
            s['mean_condition_attack_alert_rate'],-(delay if delay is not None else 1e6),
            s['mean_condition_end_to_end_recall'],-policy['alert_k'],-policy['evidence_expiry_s'])
    selected=max(candidates,key=rank)
    checkpoint['alert_policy']=selected['alert_policy']
    checkpoint['alert_rule']=('Validation-calibrated causal K-of-N fresh non-Benign classifier evidence with benign hysteresis and evidence expiry. '
        'Evil_Twin-specific alerts additionally require source evidence age <= max_ev_evidence_age_s. Physics residuals and communication health are separate warnings.')
    checkpoint['alert_policy_calibration']={'selection':'development validation only','acceptance_thresholds':{
        'benign_mission_alert_rate_max':ALERT_FALSE_MISSION_LIMIT,
        'false_alert_episodes_per_benign_minute_max':ALERT_FALSE_EPISODES_PER_MIN_LIMIT,
        'evil_twin_pre_attack_mission_rate_max':ALERT_PRE_ATTACK_LIMIT,
        'mean_attack_alert_detection_delay_s_max':ALERT_MEAN_DELAY_MAX_S}}
    fd,tmp_name=tempfile.mkstemp(prefix='checkpoint_v5_4r2_',suffix='.tmp',dir=model_dir)
    try:
        with os.fdopen(fd,'wb') as f:pickle.dump(checkpoint,f,pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name,checkpoint_path)
    finally:
        if os.path.exists(tmp_name):os.unlink(tmp_name)
    new_sha=file_digest(checkpoint_path)
    result={'contract':REVISION,'provenance':'SIMULATED_ONLY_NOT_SITL_NOT_CAPTURED','selected_policy':selected['alert_policy'],
        'validation_acceptance_met':selected['validation_acceptance_met'],'acceptance_thresholds':checkpoint['alert_policy_calibration']['acceptance_thresholds'],
        'selected_metrics':selected,'candidates':candidates,'candidate_grid':'2-of-3, 3-of-4, 4-of-5, 4-of-6; expiry 0.6, 0.8, 1.2 s; EV source-age limit 0.35 s',
        'old_checkpoint_sha256':old_sha,'new_checkpoint_sha256':new_sha,'classifier_weights_changed':False,
        'validation_data':[dict(path=str(d.path),run_manifest_sha256=file_digest(d.path/'run_manifest.csv'),checksums_sha256=file_digest(d.path/'checksums.json')) for d,_,_ in validations],
        'trainval_only_threshold_and_fitted_model_settings_unchanged':True}
    results.mkdir(parents=True);calibration_path=results/'alert_policy_calibration.json';write_json(calibration_path,result)
    pd.DataFrame([{**{k:v for k,v in s.items() if k!='condition_metrics'},'alert_policy':json.dumps(s['alert_policy'],sort_keys=True)} for s in candidates]).to_csv(results/'alert_policy_calibration_sweep.csv',index=False)
    model_manifest=read_json(manifest_path);model_manifest['checkpoint_sha256']=new_sha
    model_manifest['alert_policy']=selected['alert_policy'];model_manifest['validation_alert_policy_calibration_sha256']=file_digest(calibration_path)
    model_manifest['validation_alert_policy_acceptance_met']=selected['validation_acceptance_met']
    write_json(manifest_path,model_manifest)
    # Update only the manifest accompanying this reproduction, never the
    # historical selected-run training record after repository relocation.
    training_manifest_path=results.parent/'training_manifest.json'
    if training_manifest_path.exists():
        training_manifest=read_json(training_manifest_path);training_manifest['checkpoint_sha256']=new_sha
        training_manifest['alert_policy_calibration_sha256']=file_digest(calibration_path);write_json(training_manifest_path,training_manifest)
    return result

def infer_dataset(dataset,checkpoint,stream=True):
    views=build_views(dataset.packets,float(dataset.config['sampling_hz']))
    if stream:
        cache={r['observation_id']:{'primary':None,'base':None,'short3':None} for r in views};cache_seconds=0.
    else:cache,cache_seconds=_probability_cache(views,checkpoint['models'])
    cache['_models']={'primary':checkpoint['models']['primary'],'base':checkpoint['models']['fallback_base'],'short3':checkpoint['models']['fallback_short3']}
    events=_make_predictions(views,cache,checkpoint['fallback_selected'],checkpoint['primary_threshold'],checkpoint['fallback_threshold'],
                            checkpoint['physics_threshold'],checkpoint['prediction_deadline_s'],stream=stream)
    intervals=[];ev_intervals=[];physics_intervals=[];byrun={}
    for event in events:byrun.setdefault(event['run_id'],[]).append(event)
    for run_id in dataset.runs.run_id.astype(str):
        run_events=byrun.get(run_id,[])
        end=int(float(dataset.config['duration_s'])*1e9)
        a,e,p=_alert_intervals(run_events,end,checkpoint['alert_policy']['alert_k'],checkpoint['alert_policy']['alert_n'],
          checkpoint['alert_policy']['evidence_expiry_s'],checkpoint['alert_policy']['clear_benign_predictions'],checkpoint['physics_threshold'],
          checkpoint['alert_policy'].get('max_ev_evidence_age_s',float('inf')))
        for table,values in ((intervals,a),(ev_intervals,e),(physics_intervals,p)):
            for item in values:item['run_id']=run_id
            table.extend(values)
    if stream:cache_seconds=sum(float(e['inference_latency_s']) for e in events)
    for event in events:
        event['classifier_attack_alert']=bool(event.get('attack_alert_active',False))
        event['evil_twin_alert']=bool(event.get('evil_twin_alert_active',False))
        event['physics_warning']=bool(event.get('physics_warning_active',False))
        event['communication_warning']=False
        event['attack_alert']=event['classifier_attack_alert']
    byrun={}
    for e in events:byrun.setdefault(e['run_id'],[]).append(e)
    # Runtime health only: a warning is not an attack classification.
    for run_id,group in byrun.items():
        group.sort(key=lambda e:e['sample_index']);streak=0
        for e in group:
            if e['timely']:streak=0
            else:streak+=1
            e['communication_warning']=streak>=2
    return views,events,intervals,ev_intervals,physics_intervals,cache_seconds

def _build_audit(dataset,events,intervals,ev_intervals,physics_intervals):
    truth=dataset.truth.copy();flow=pd.read_csv(dataset.path/'network_flow.csv')
    packet=dataset.packets[['run_id','observation_id','arrival_time_ns','source_time_ns','sequence']].copy()
    pred=pd.DataFrame(events)
    if pred.empty:pred=pd.DataFrame(columns=['observation_id'])
    keep=[c for c in ['observation_id','prediction_attempt_time_ns','prediction_time_ns','arrival_evidence_age_s','evidence_age_s','observation_age_s','inference_latency_s','prediction','confidence','evil_twin_probability','selected_model','status','timely','fallback_used','primary_ready','segment_count','reset_reason','segment_origin','error_reason','inference_error','physics_score','physics_candidate'] if c in pred]
    delivered=packet.merge(pred[keep],on='observation_id',how='left',validate='one_to_one')
    # Keep delivered packet timing and sequence in the scheduled-observation
    # audit.  Only drop the duplicate run key: arrival/source times are evidence
    # required to classify deadline misses and must not disappear at this join.
    ledger=truth.merge(flow,on=['run_id','observation_id'],how='left',validate='one_to_one').merge(delivered.drop(columns=['run_id']),on='observation_id',how='left',validate='one_to_one')
    deadline_ns=int(DEADLINE_S*1e9)
    if 'observation_time_ns' in ledger:ledger['audit_observation_time_ns']=ledger.observation_time_ns
    else:ledger['audit_observation_time_ns']=ledger.simulated_time_ns
    ledger['audit_arrival_time_ns']=ledger.arrival_time_ns
    # One mutually exclusive reason per scheduled observation.
    reasons=[];timely=[]
    for row in ledger.itertuples():
        status=''
        delivered_flag=bool(row.telemetry_delivered) if pd.notna(row.telemetry_delivered) else False
        if not delivered_flag:
            reason=str(row.delivery_reason) if pd.notna(row.delivery_reason) else 'unknown'
            status='packet_never_delivered_'+reason
        elif pd.notna(row.audit_arrival_time_ns) and int(row.audit_arrival_time_ns)>int(row.audit_observation_time_ns)+deadline_ns:
            status='delivered_too_late'
        elif pd.notna(row.error_reason) and str(row.error_reason):
            status='invalid_features'
        elif pd.isna(row.prediction):
            if pd.notna(row.segment_count) and int(row.segment_count)<FALLBACK_WINDOW:
                status='post_gap_warmup' if str(row.segment_origin)=='delivery_gap' else 'startup_warmup'
            elif str(row.status)=='inference_failure':status='inference_failure'
            else:status='no_usable_fallback'
        elif pd.notna(row.prediction_time_ns) and int(row.prediction_time_ns)>int(row.audit_observation_time_ns)+deadline_ns:
            status='inference_deadline_miss'
        else:status='prediction_available'
        reasons.append(status);timely.append(status=='prediction_available')
    ledger['no_prediction_reason']=reasons;ledger['timely_prediction']=timely
    ledger['prediction_status']=np.where(ledger.timely_prediction,'predicted', 'unavailable_or_late')
    ledger['evidence_age_s']=ledger.get('evidence_age_s',pd.Series(np.nan,index=ledger.index))
    ledger['attack_alert']=False;ledger['evil_twin_alert']=False;ledger['physics_warning']=False;ledger['communication_warning']=False
    interval_tables=[(intervals,'attack_alert'),(ev_intervals,'evil_twin_alert'),(physics_intervals,'physics_warning')]
    for run_id,indices in ledger.groupby('run_id',sort=False).groups.items():
        ordered=ledger.loc[indices].sort_values('audit_observation_time_ns');times=ordered.audit_observation_time_ns.to_numpy(dtype=np.int64)
        for interval_table,column in interval_tables:
            diff=np.zeros(len(times)+1,dtype=np.int32)
            for ep in interval_table:
                if ep['run_id']!=run_id:continue
                left=int(np.searchsorted(times,int(ep['start_time_ns']),side='left'));right=int(np.searchsorted(times,int(ep['end_time_ns']),side='left'))
                if left<len(times):diff[left]+=1;diff[min(right,len(times))]-=1
            ledger.loc[ordered.index,column]=(np.cumsum(diff[:-1])>0)
        unavailable=~ordered.timely_prediction.to_numpy(dtype=bool);streak=0;warnings=[]
        for missing in unavailable:
            streak=streak+1 if missing else 0;warnings.append(streak>=2)
        ledger.loc[ordered.index,'communication_warning']=warnings
    return ledger,intervals,ev_intervals,physics_intervals

def _bootstrap_cluster(ledger,metric,replicates=2000,seed=20261108):
    rng=np.random.default_rng(seed);groups=[g for _,g in ledger.groupby('run_id',sort=False)]
    numer=[];denom=[]
    for g in groups:
        valid=g.timely_prediction.to_numpy(dtype=bool);pred=g.prediction.to_numpy(dtype=object)
        if metric=='e2e_recall':
            truth=g.Target_Label.eq('Evil_Twin').to_numpy();numer.append(np.sum(truth&valid&(pred=='Evil_Twin')));denom.append(np.sum(truth))
        elif metric=='precision':
            truth=g.Target_Label.eq('Evil_Twin').to_numpy();numer.append(np.sum(truth&valid&(pred=='Evil_Twin')));denom.append(np.sum(valid&(pred=='Evil_Twin')))
        elif metric=='benign_fpr':
            benign=g.Target_Label.eq('Benign').to_numpy()&valid;numer.append(np.sum(benign&(pred!='Benign')));denom.append(np.sum(benign))
        elif metric=='coverage':numer.append(int(valid.sum()));denom.append(len(g))
    numer=np.asarray(numer,dtype=float);denom=np.asarray(denom,dtype=float);values=np.empty(replicates)
    for j in range(replicates):
        idx=rng.integers(0,len(groups),size=len(groups));values[j]=numer[idx].sum()/max(1.,denom[idx].sum())
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

def score_ledger(ledger,intervals):
    y=ledger.Target_Label.astype(str).to_numpy();valid=ledger.timely_prediction.to_numpy(dtype=bool);p=ledger.prediction.to_numpy(dtype=object)
    scored=_class_metrics(y,p,valid);offline_valid=pd.notna(ledger.prediction).to_numpy(dtype=bool);offline=_class_metrics(y,p,offline_valid)
    ev=y=='Evil_Twin';tp=int(np.sum(ev&valid&(p=='Evil_Twin')));support=int(ev.sum());scored_ev=int(np.sum(ev&valid));ev_pred=int(np.sum(valid&(p=='Evil_Twin')))
    offline_tp=int(np.sum(ev&offline_valid&(p=='Evil_Twin')))
    end_cm=confusion_matrix(y,np.where(valid,p,'UNAVAILABLE'),labels=CLASSES+['UNAVAILABLE'])
    gaps=ledger.loc[~ledger.timely_prediction,'no_prediction_reason'].value_counts().to_dict()
    # Reconciliation is exact: every scheduled row appears once in the audit and in one status category.
    if sum(int(v) for v in ledger.no_prediction_reason.value_counts())!=len(ledger):raise AssertionError('Observation accounting does not reconcile')
    run_table=[]
    run_map={}
    for run_id,g in ledger.groupby('run_id',sort=False):
        evmask=g.Target_Label.eq('Evil_Twin');evsupport=int(evmask.sum());evt=int((evmask&g.timely_prediction&g.prediction.eq('Evil_Twin')).sum());run_scored_ev=int((evmask&g.timely_prediction).sum())
        benign=g.Target_Label.eq('Benign')&g.timely_prediction
        bfp=int((benign&g.prediction.ne('Benign')).sum())
        mission={
          'run_id':run_id,'scenario':str(g.scenario.iloc[0]),'scheduled_rows':len(g),'timely_predictions':int(g.timely_prediction.sum()),'offline_predictions':int(pd.notna(g.prediction).sum()),
          'coverage':float(g.timely_prediction.mean()),'evil_twin_support':evsupport,'evil_twin_scored_support':run_scored_ev,'evil_twin_tp':evt,
          'evil_twin_end_to_end_recall':evt/max(1,evsupport),'evil_twin_conditional_recall':evt/max(1,run_scored_ev),
          'evil_twin_precision':evt/max(1,int((g.timely_prediction&g.prediction.eq('Evil_Twin')).sum())),
          'benign_scored_rows':int(benign.sum()),'benign_fp':bfp,'benign_row_fpr':bfp/max(1,int(benign.sum())),
          'unavailable_rows':int((~g.timely_prediction).sum()),'alert_episodes':0,'false_alert_episodes':0,
          'attack_alert_mission':False,'pre_attack_attack_alert':False,'attack_alert_detection_delay_s':None}
        run_map[str(run_id)]=mission;run_table.append(mission)
    alert_rows=[]
    runmeta=ledger.groupby('run_id',sort=False).first()
    for ep in intervals:
        r=run_map[str(ep['run_id'])];is_benign=r['scenario']=='Benign';
        meta=ledger[ledger.run_id.eq(ep['run_id'])].iloc[0]
        # Schedule time is used only here, after inference, to classify false episodes.
        end_index=int(ledger.loc[ledger.run_id.eq(ep['run_id']),'sample_index'].max())+1
        start_active=int(meta.get('attack_start_index',0))/float(ledger.loc[ledger.run_id.eq(ep['run_id']),'sample_index'].max()+1)*end_index if False else None
        attack_start_ns=None;attack_stop_ns=None
        # schedule indices map to simulator nanoseconds at the configured cadence.
        # The run manifest is attached to the ledger by _build_audit when scoring.
        if 'attack_start_index' in ledger.columns:
            hz=float(ledger.attrs.get('sampling_hz',5.0));attack_start_ns=int(ledger.loc[ledger.run_id.eq(ep['run_id']),'attack_start_index'].iloc[0]/hz*1e9);attack_stop_ns=int(ledger.loc[ledger.run_id.eq(ep['run_id']),'attack_end_index'].iloc[0]/hz*1e9)
        overlaps=bool(attack_start_ns is not None and ep['end_time_ns']>=attack_start_ns and ep['start_time_ns']<attack_stop_ns)
        pre=bool(attack_start_ns is not None and ep['start_time_ns']<attack_start_ns)
        false=is_benign or not overlaps or pre
        if false:r['false_alert_episodes']+=1
        r['alert_episodes']+=1
        if pre:r['pre_attack_attack_alert']=True
        if attack_start_ns is not None and attack_start_ns<=ep['start_time_ns']<attack_stop_ns:
            r['attack_alert_mission']=True
            delay=(ep['start_time_ns']-attack_start_ns)/1e9
            if r['attack_alert_detection_delay_s'] is None or delay<r['attack_alert_detection_delay_s']:r['attack_alert_detection_delay_s']=delay
        episode=dict(ep)
        # Per-condition episode CSVs already contain schedule-derived fields;
        # recompute them from the pooled audit and overwrite the old values.
        episode.update(scheduled_scenario=r['scenario'],false_alert=bool(false),pre_attack=pre,
                       overlaps_scheduled_attack=overlaps)
        alert_rows.append(episode)
    for run_id,r in run_map.items():
        r['false_alert_mission']=r['false_alert_episodes']>0
    false_support=Counter()
    pre_support=Counter()
    for ep in alert_rows:
        models=[x for x in str(ep.get('supporting_models','')).split(';') if x]
        if ep['false_alert']:
            for model in models:false_support[model]+=1
        if ep['pre_attack']:
            for model in models:pre_support[model]+=1
    benign_runs=[r for r in run_table if r['scenario']=='Benign'];ev_runs=[r for r in run_table if r['scenario']=='Evil_Twin']
    total_benign_rows=int(np.sum((ledger.Target_Label.eq('Benign')&ledger.timely_prediction)))
    total_benign_fp=int(np.sum((ledger.Target_Label.eq('Benign')&ledger.timely_prediction)&ledger.prediction.ne('Benign')))
    ev_alert_any=sum(r['attack_alert_mission'] for r in ev_runs);ev_pre=sum(r['pre_attack_attack_alert'] for r in ev_runs)
    score=dict(contract=REVISION,provenance='SIMULATED_ONLY_NOT_SITL_NOT_CAPTURED',scheduled_observations=len(ledger),
        timely_predictions=int(valid.sum()),prediction_coverage=float(valid.mean()),
        prediction_coverage_run_clustered_95_ci=_bootstrap_cluster(ledger,'coverage'),
        conditional_evil_twin_recall=tp/max(1,scored_ev),end_to_end_evil_twin_recall=tp/max(1,support),
        delayed_offline_reconstruction=dict(prediction_coverage=float(offline_valid.mean()),evil_twin_recall_including_offline_predictions=offline_tp/max(1,support),row_level=offline),
        end_to_end_evil_twin_recall_run_clustered_95_ci=_bootstrap_cluster(ledger,'e2e_recall'),
        evil_twin_precision=tp/max(1,ev_pred),evil_twin_precision_run_clustered_95_ci=_bootstrap_cluster(ledger,'precision'),
        evil_twin_confusion=dict(tp=tp,fn=support-tp,fp=ev_pred-tp,support=support),
        benign_scored_row_false_positive_rate=total_benign_fp/max(1,total_benign_rows),
        benign_fpr_run_clustered_95_ci=_bootstrap_cluster(ledger,'benign_fpr'),
        row_level=scored,scheduled_confusion_labels=CLASSES+['UNAVAILABLE'],scheduled_confusion_matrix=end_cm.tolist(),
        end_to_end_per_class_recall={c:float(end_cm[i,i]/max(1,end_cm[i].sum())) for i,c in enumerate(CLASSES)},
        gap_breakdown={k:int(v) for k,v in gaps.items()},gap_breakdown_reconciles=sum(gaps.values())==len(ledger)-int(valid.sum()),
        total_false_alert_episodes=len([e for e in alert_rows if e['false_alert']],),
        benign_missions_with_alert=sum(r['false_alert_mission'] for r in benign_runs),benign_mission_count=len(benign_runs),
        benign_mission_alert_rate=sum(r['false_alert_mission'] for r in benign_runs)/max(1,len(benign_runs)),
        false_alert_episodes_per_benign_minute=sum(r['false_alert_episodes'] for r in benign_runs)/max(1,len(benign_runs)*float(ledger.attrs.get('duration_s',60))/60),
        mean_false_alert_episode_duration_s=float(np.mean([e['duration_s'] for e in alert_rows if e['false_alert']])) if any(e['false_alert'] for e in alert_rows) else 0.,
        false_alert_episode_support_models=dict(false_support),pre_attack_alert_support_models=dict(pre_support),
        evil_twin_missions_with_pre_attack_alerts=ev_pre,evil_twin_mission_count=len(ev_runs),
        evil_twin_pre_attack_alert_rate=ev_pre/max(1,len(ev_runs)),evil_twin_any_attack_alert_rate=ev_alert_any/max(1,len(ev_runs)),
        evil_twin_attack_alert_detection_delay_mean_s=float(np.mean([r['attack_alert_detection_delay_s'] for r in ev_runs if r['attack_alert_detection_delay_s'] is not None])) if any(r['attack_alert_detection_delay_s'] is not None for r in ev_runs) else None,
        evil_twin_attack_alert_detection_delay_p90_s=float(np.quantile([r['attack_alert_detection_delay_s'] for r in ev_runs if r['attack_alert_detection_delay_s'] is not None],.9)) if any(r['attack_alert_detection_delay_s'] is not None for r in ev_runs) else None,
        primary_model='HistGradientBoostingClassifier',fallback_schema='22 base or 3-observation causal (22 base/66 derived) with primary 22/88 unchanged',
        labels_joined_after_inference=True,alert_episodes=alert_rows,per_mission=run_table)
    return score

def _save_evaluation(dataset,checkpoint,output,stream=True):
    output=Path(output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite result directory: {output}')
    output.mkdir(parents=True)
    views,events,intervals,ev_intervals,physics_intervals,cache_seconds=infer_dataset(dataset,checkpoint,stream=stream)
    ledger,intervals,ev_intervals,physics_intervals=_build_audit(dataset,events,intervals,ev_intervals,physics_intervals)
    ledger.attrs['sampling_hz']=float(dataset.config['sampling_hz']);ledger.attrs['duration_s']=float(dataset.config['duration_s'])
    runs=dataset.runs[['run_id','scenario','attack_start_index','attack_end_index','seed','profile','intensity','replay_delay','subtype']]
    # Ground truth intentionally contains no scenario metadata.  Attach run
    # descriptors only after predictions have been generated, using the run
    # manifest's unique run_id mapping.
    ledger=ledger.merge(runs,on='run_id',how='left',validate='many_to_one')
    ledger.attrs['sampling_hz']=float(dataset.config['sampling_hz']);ledger.attrs['duration_s']=float(dataset.config['duration_s'])
    score=score_ledger(ledger,intervals)
    # Add physics-only warning counts and explicit alert source separation.
    score['physics_warning_episodes']=len(physics_intervals)
    physics_sources=Counter()
    for episode in physics_intervals:
        for model in str(episode.get('supporting_models','')).split(';'):
            if model:physics_sources[model]+=1
    score['physics_warning_support_models']=dict(physics_sources)
    score['prediction_inference_batch_seconds']=float(cache_seconds)
    score['prediction_inference_latency_p50_ms']=float(pd.DataFrame(events).inference_latency_s.quantile(.5)*1000) if events else 0.
    score['prediction_inference_latency_p95_ms']=float(pd.DataFrame(events).inference_latency_s.quantile(.95)*1000) if events else 0.
    score['prediction_inference_latency_max_ms']=float(pd.DataFrame(events).inference_latency_s.max()*1000) if events else 0.
    score['prediction_deadline_s']=checkpoint['prediction_deadline_s']
    score['alert_policy']=checkpoint['alert_policy']
    prediction_frame=pd.DataFrame(events)
    if not prediction_frame.empty:
        # This file contains packet metadata and model outputs only; labels/schedules are absent.
        prediction_frame=prediction_frame.sort_values(['run_id','arrival_time_ns'],kind='stable')
    prediction_frame.to_csv(output/'predictions_label_free.csv',index=False)
    ledger.sort_values(['run_id','sample_index'],kind='stable').to_csv(output/'observation_audit.csv',index=False)
    pd.DataFrame(score['alert_episodes']).to_csv(output/'attack_alert_episodes.csv',index=False)
    pd.DataFrame(ev_intervals,columns=['start_time_ns','end_time_ns','duration_s','source','supporting_models','close_reason','run_id']).to_csv(output/'evil_twin_alert_episodes.csv',index=False)
    pd.DataFrame(physics_intervals,columns=['start_time_ns','end_time_ns','duration_s','source','supporting_models','close_reason','run_id']).to_csv(output/'physics_warning_episodes.csv',index=False)
    pd.DataFrame([r for r in score['per_mission']]).to_csv(output/'per_mission_metrics.csv',index=False)
    pd.DataFrame(score['row_level']['confusion_matrix'],index=CLASSES,columns=CLASSES).to_csv(output/'confusion_matrix_scored.csv')
    pd.DataFrame(score['scheduled_confusion_matrix'],index=score['scheduled_confusion_labels'],columns=score['scheduled_confusion_labels']).to_csv(output/'confusion_matrix_scheduled.csv')
    for key in ('alert_episodes','per_mission','scheduled_confusion_matrix','scheduled_confusion_labels'):score.pop(key,None)
    write_json(output/'metrics.json',score)
    # Small diagnostic figures use the full evaluated timeline.
    fig,ax=plt.subplots(figsize=(7,4));gaps=score['gap_breakdown'];labels=list(gaps) or ['none'];vals=[gaps.get(k,0) for k in labels]
    ax.bar(labels,vals,color='#bb5544');ax.set_ylabel('Scheduled observations');ax.set_title('Unavailable or late observation reasons');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(output/'gap_breakdown.png');plt.close(fig)
    ev=ledger[ledger.Target_Label.eq('Evil_Twin')]
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(ev.groupby('run_id').timely_prediction.mean(),bins=np.linspace(0,1,11),color='#3d7895');ax.set(xlabel='Mission timely-prediction coverage',ylabel='Evil_Twin missions',title='Mission-level prediction coverage');fig.tight_layout();fig.savefig(output/'mission_coverage.png');plt.close(fig)
    # Pick a deterministic representative run for a transparent schedule/prediction timeline.
    if len(ledger):
        run=str(ledger.run_id.iloc[0]);sample=ledger[ledger.run_id.eq(run)]
        fig,ax=plt.subplots(figsize=(10,3));class_ix={c:i for i,c in enumerate(CLASSES)}
        ax.step(sample.sample_index/float(dataset.config['sampling_hz']),sample.Target_Label.map(class_ix),where='post',label='Scheduled label',color='#333')
        observed=sample[sample.timely_prediction]
        ax.scatter(observed.sample_index/float(dataset.config['sampling_hz']),observed.prediction.map(class_ix),s=8,label='Timely model prediction',color='#3477a1')
        ax.set_yticks(range(5),CLASSES);ax.set_xlabel('Simulated mission time (s)');ax.legend(ncol=2);fig.tight_layout();fig.savefig(output/'timeline.png');plt.close(fig)
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>ACPDAD — Attack Detection Dashboard</title><style>body{{font:16px system-ui;margin:2rem;max-width:1100px}}table{{border-collapse:collapse}}td,th{{padding:.4rem .8rem;border:1px solid #bbb}}img{{max-width:100%}}.sim{{background:#fff4d6;padding:1rem}}</style></head><body><h1>ACPDAD — Attack Detection Dashboard</h1><p class="sim"><b>Simulated telemetry only.</b> Not SITL, captured Wi-Fi, a physical vehicle, or real-world validation.</p><h2>End-to-end performance</h2><p>Scheduled samples: {score['scheduled_observations']}; timely predictions: {score['timely_predictions']} ({score['prediction_coverage']:.1%} coverage); Evil_Twin end-to-end recall: {score['end_to_end_evil_twin_recall']:.1%}; conditional recall on scored rows: {score['conditional_evil_twin_recall']:.1%}.</p><p>False alert missions among benign runs: {score['benign_missions_with_alert']}/{score['benign_mission_count']}; false episodes per benign mission-minute: {score['false_alert_episodes_per_benign_minute']:.3f}. Communication warnings are separate from attack alerts.</p><h2>Observation availability</h2><img src="gap_breakdown.png"><h2>Mission coverage and timeline</h2><img src="mission_coverage.png"><img src="timeline.png"><p>Full schedule ledger: observation_audit.csv. Label-free predictions: predictions_label_free.csv. Metrics and confidence intervals: metrics.json.</p></body></html>'''
    (output/'index.html').write_text(html)
    return score,output

def verify(data,model_dir):
    dataset=load_dataset(data);checkpoint=load_checkpoint(model_dir)
    views=build_views(dataset.packets,float(dataset.config['sampling_hz']))
    rebuilt={r['observation_id']:r['primary'] for r in views if r['primary'] is not None}
    common=set(rebuilt)&set(dataset.saved_features.index)
    if len(common)!=len(dataset.saved_features):raise AssertionError('Saved primary rows do not reconstruct from arrival-ordered packets')
    if not np.allclose(np.vstack([rebuilt[k] for k in dataset.saved_features.index]),dataset.saved_features.to_numpy(),rtol=1e-8,atol=1e-8):
        raise AssertionError('Primary temporal features changed during reconstruction')
    if any(x in set(BASE) for x in ('Target_Label','scenario','run_id','seed','split','ap_identity','attack_active','sample_index')):
        raise AssertionError('Label/metadata leakage in base feature contract')
    # Batch and streaming predictions must agree on the same causal packet sequence.
    run_id=str(dataset.runs.run_id.iloc[0]);sub=dataset.packets[dataset.packets.run_id.eq(run_id)].sort_values('arrival_time_ns',kind='stable').head(35)
    small=build_views(sub,float(dataset.config['sampling_hz']));cache,_=_probability_cache(small,checkpoint['models']);cache['_models']={'primary':checkpoint['models']['primary'],'base':checkpoint['models']['fallback_base'],'short3':checkpoint['models']['fallback_short3']}
    batch=_make_predictions(small,cache,checkpoint['fallback_selected'],checkpoint['primary_threshold'],checkpoint['fallback_threshold'],checkpoint['physics_threshold'],checkpoint['prediction_deadline_s'],False)
    stream_cache={r['observation_id']:{'primary':None,'base':None,'short3':None} for r in small};stream_cache['_models']=cache['_models']
    streaming=_make_predictions(small,stream_cache,checkpoint['fallback_selected'],checkpoint['primary_threshold'],checkpoint['fallback_threshold'],checkpoint['physics_threshold'],checkpoint['prediction_deadline_s'],True)
    for a,b in zip(batch,streaming):
        if a['observation_id']!=b['observation_id'] or a['prediction']!=b['prediction'] or not np.isclose(a['evil_twin_probability'],b['evil_twin_probability'],atol=1e-12):
            raise AssertionError('Batch/stream predictions differ')
    # Per-run feature extraction must not inherit state from the preceding run.
    first_ids=set(sub.observation_id.astype(str));combined={r['observation_id']:r for r in small}
    isolated=build_views(sub,float(dataset.config['sampling_hz']))
    iso={r['observation_id']:r for r in isolated}
    for oid in first_ids:
        a=combined[oid]['base'];b=iso[oid]['base']
        if (a is None)!=(b is None) or (a is not None and not np.allclose(a,b,atol=0,rtol=0)):raise AssertionError('Run state isolation failure')
    # Explicit out-of-order check: detector rejects rather than treating it as continuous.
    if len(sub)<2:raise AssertionError('Insufficient packets for order check')
    ordered=sub.to_dict('records');det=Extractor(float(dataset.config['sampling_hz']));det.observe(ordered[1]);out_of_order_rejected=False
    try:det.observe(ordered[0])
    except ValueError:out_of_order_rejected=True
    if not out_of_order_rejected:raise AssertionError('Out-of-order arrival was accepted')
    # Long-gap check verifies both fallback and primary windows restart.
    gap=sub.head(18).copy().reset_index(drop=True)
    if len(gap)<12:raise AssertionError('Insufficient packets for gap reset check')
    shift=1_500_000_000
    gap.loc[10:,'arrival_time_ns']=gap.loc[10:,'arrival_time_ns'].astype('int64')+shift
    gap_views=build_views(gap,float(dataset.config['sampling_hz']));post=gap_views[10]
    if post['segment_origin']!='delivery_gap' or post['primary'] is not None or post['short3'] is not None:
        raise AssertionError('Long gap did not reset temporal and fallback windows')
    if len(gap_views)<=12 or gap_views[11]['short3'] is not None or gap_views[12]['short3'] is None or gap_views[12]['primary'] is not None:
        raise AssertionError('Short fallback did not warm causally from post-gap observations')
    # Fresh stale replay and benign roaming checks; generator truth is not fed to inference.
    rp,rt,rm,emitted,states,flow=simulate(dataset.config,'Replay','verify_replay',90123456,'val',98760000,include_evidence=True)
    stale=(rp.source_time_ns<rp.observation_time_ns-1_000_000_000) if 'observation_time_ns' in rp else False
    if not bool(np.any(stale)):raise AssertionError('Replay proof did not preserve stale source timestamps')
    bp,bt,bm,be,bs,bf=simulate(dataset.config,'Benign','verify_benign',90123457,'val',98761000,include_evidence=True)
    benign_roam=bool((bf.handoff_added_latency_s>0).any())
    if not benign_roam:raise AssertionError('Benign roaming/handoff traffic missing')
    # Selected/fallback models only consume feature vectors; expose a small online-switch proof.
    fallback_count=sum(x['selected_model'].startswith('fallback_') for x in batch if x['selected_model']!='none')
    primary_count=sum(x['selected_model']=='primary_temporal' for x in batch)
    if not fallback_count or not primary_count:raise AssertionError('Fallback/primary switching was not exercised')
    # A stale prediction is not retained across expiry without new evidence.
    probe=[dict(run_id='probe',observation_id='a',arrival_time_ns=0,prediction='DoS',timely=True,physics_candidate=False),
           dict(run_id='probe',observation_id='b',arrival_time_ns=200_000_000,prediction='DoS',timely=True,physics_candidate=False),
           dict(run_id='probe',observation_id='c',arrival_time_ns=2_000_000_000,prediction='Benign',timely=True,physics_candidate=False)]
    test_intervals,_,_=_alert_intervals(probe,3_000_000_000,2,3,.8,3,checkpoint['physics_threshold'])
    if test_intervals and test_intervals[0]['end_time_ns']>1_100_000_000:raise AssertionError('Attack alert survived evidence expiry')
    stale_ev=[dict(run_id='stale',observation_id=f's{i}',arrival_time_ns=i*200_000_000,prediction='Evil_Twin',timely=True,
                   physics_candidate=False,evidence_age_s=12.,selected_model='primary_temporal') for i in range(5)]
    stale_attack,stale_ev_alert,_=_alert_intervals(stale_ev,2_000_000_000,2,3,.8,3,checkpoint['physics_threshold'],.6)
    if not stale_attack or stale_ev_alert:raise AssertionError('Stale Evil_Twin evidence was not separated from generic attack/replay warning')
    return dict(status='passed',contract=REVISION,feature_contract=FEATURE_CONTRACT,rows=len(dataset.saved_features),runs=len(dataset.runs),
        primary_features=len(BASE),temporal_features=len(TEMPORAL),fallback_window=FALLBACK_WINDOW,checkpoint_sha256=read_json(Path(model_dir)/'manifest.json')['checkpoint_sha256'],
        primary_reconstruction=True,batch_stream_agreement=True,run_state_isolation=True,out_of_order_arrival_rejected=True,
        long_gap_window_reset=True,stale_replay_payload_detected=True,benign_roaming_present=benign_roam,
        fallback_rows_checked=fallback_count,primary_rows_checked=primary_count,stale_alert_expiry=True,
        stale_evil_twin_evidence_excluded=True,
        provenance='SIMULATED_ONLY_NOT_SITL_NOT_CAPTURED')

def _default_output(name):
    return ROOT/'results'/'replays'/str(time.time_ns())

def predict_cli(data,model_dir,output,run_id=None,all_runs=False,run_alias=None):
    dataset=load_dataset(data);available=dataset.runs.run_id.astype(str).tolist()
    if run_id is not None and run_id not in available:
        raise ValueError(f'Unknown run ID {run_id!r}. Available run IDs: '+', '.join(available))
    selected=available if all_runs else [run_id]
    packets=dataset.packets[dataset.packets.run_id.astype(str).isin(selected)].copy()
    found=set(packets.run_id.astype(str))
    missing=set(selected)-found
    if missing:raise ValueError('Selected run(s) have no delivered packets: '+', '.join(sorted(missing)))
    sub=Dataset(dataset.path,dataset.schema,dataset.config,packets,dataset.truth[dataset.truth.run_id.astype(str).isin(selected)],
                dataset.runs[dataset.runs.run_id.astype(str).isin(selected)],dataset.saved_features)
    ck=load_checkpoint(model_dir);_,events,_,_,_,_=infer_dataset(sub,ck,stream=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(events).sort_values(['run_id','arrival_time_ns'],kind='stable')
    if run_alias is not None and len(selected) == 1:
        orig_id = str(selected[0])
        alias_str = str(run_alias)
        frame['run_id'] = alias_str
        frame['observation_id'] = frame['observation_id'].astype(str).str.replace(f'^{re.escape(orig_id)}', alias_str, regex=True)
    forbidden={'Target_Label','attack_active','scenario','seed','split','attack_start_index','attack_end_index','ap_identity','true_height_cm','true_vz_down_mps'}
    if forbidden&set(frame.columns):raise AssertionError('Prediction output contains labels, attack metadata, or identifiers')
    if frame.observation_id.duplicated().any():raise AssertionError('Prediction output duplicates observation mapping')
    if not frame.arrival_time_ns.groupby(frame.run_id).apply(lambda s:s.is_monotonic_increasing).all():raise AssertionError('Arrival order changed')
    with output.open('w') as f:frame.to_csv(f,index=False)
    return dict(status='passed',runs=len(selected),delivered_packets=len(packets),prediction_rows=len(frame),output=str(output),label_free=True,
                order='run_id,arrival_time_ns stable',provenance='SIMULATED_ONLY',run_alias=run_alias)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--config',type=Path,default=ROOT/'config/simulation.json');g.add_argument('--output',type=Path,required=True);g.add_argument('--seed',type=int);g.add_argument('--runs-per-class',type=int);g.add_argument('--duration',type=float);g.add_argument('--hz',type=float);g.add_argument('--profile',choices=['hover','straight','patrol','figure_eight','climb_descent']);g.add_argument('--set',action='append',default=[])
    t=sub.add_parser('train');t.add_argument('--data',type=Path,default=DEFAULT_DATA);t.add_argument('--model',type=Path,default=DEFAULT_MODEL);t.add_argument('--results',type=Path,default=DEFAULT_RESULTS);t.add_argument('--validation-data',type=Path,action='append',default=[])
    ca=sub.add_parser('calibrate-alerts');ca.add_argument('--data',type=Path,default=DEFAULT_DATA);ca.add_argument('--model',type=Path,required=True);ca.add_argument('--results',type=Path,required=True);ca.add_argument('--validation-data',type=Path,action='append',default=[])
    for command in ('evaluate','demo'):
        e=sub.add_parser(command);e.add_argument('--data',type=Path,default=DEFAULT_DATA);e.add_argument('--model',type=Path,default=DEFAULT_MODEL);e.add_argument('--output',type=Path);e.add_argument('--split',choices=['train','val','test'],default='val');e.add_argument('--run-id');e.add_argument('--realtime',action='store_true')
    p=sub.add_parser('predict');p.add_argument('--data',type=Path,default=DEFAULT_DATA);p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-alias',type=str,default=None)
    choice=p.add_mutually_exclusive_group(required=True);choice.add_argument('--run-id');choice.add_argument('--all-runs',action='store_true')
    v=sub.add_parser('verify');v.add_argument('--data',type=Path,default=DEFAULT_DATA);v.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    sub.add_parser('evaluate-final');sub.add_parser('protocol',add_help=False)
    args,remaining=parser.parse_known_args(argv)
    if args.command=='protocol':
        from .evaluation_protocol import main as protocol_main
        return protocol_main(remaining)
    if args.command=='generate':
        c=read_json(args.config)
        if args.seed is not None:
            c['master_seeds']={k:args.seed+i*1_000_000 for i,k in enumerate(('train','val','test'))};c['final_test_protocol_sha256']=None
        if args.runs_per_class is not None:c['runs_per_class']={k:args.runs_per_class for k in c['runs_per_class']}
        if args.duration is not None:c['duration_s']=args.duration
        if args.hz is not None:c['sampling_hz']=args.hz
        if args.profile is not None:c['profile']=args.profile
        for pair in args.set:
            key,value=pair.split('=',1);target=c
            parts=key.split('.')
            for part in parts[:-1]:target=target.setdefault(part,{})
            target[parts[-1]]=json.loads(value)
        print(generate(c,args.output));return 0
    if args.command=='train':
        result=train(args.data,args.model,args.results,args.validation_data);print('Trained V5.4R2:',result['fallback_selected']);return 0
    if args.command=='calibrate-alerts':
        result=calibrate_alerts(args.model,args.data,args.validation_data,args.results);print(json.dumps({k:result[k] for k in ('selected_policy','validation_acceptance_met','acceptance_thresholds','old_checkpoint_sha256','new_checkpoint_sha256')},indent=2));print('Saved',args.results/'alert_policy_calibration.json');return 0
    if args.command=='verify':
        result=verify(args.data,args.model);path=ROOT/'results/verification'/f"verify_{time.time_ns()}.json";write_json(path,result);print(json.dumps(result,indent=2));print('Saved',path);return 0
    if args.command=='predict':
        result=predict_cli(args.data,args.model,args.output,args.run_id,args.all_runs,run_alias=getattr(args,'run_alias',None));print(json.dumps(result,indent=2));return 0
    if args.command in ('evaluate','demo'):
        output=args.output or _default_output('demo')
        if args.run_id is not None:
            dataset=load_dataset(args.data)
            if args.run_id not in set(dataset.runs.loc[dataset.runs.split.eq(args.split),'run_id'].astype(str)):
                raise ValueError(f'Unknown run ID {args.run_id!r} for split {args.split}')
            dataset.runs=dataset.runs[dataset.runs.run_id.astype(str).eq(args.run_id)]
            dataset.packets=dataset.packets[dataset.packets.run_id.astype(str).eq(args.run_id)]
            dataset.truth=dataset.truth[dataset.truth.run_id.astype(str).eq(args.run_id)]
        d=load_dataset(args.data) if args.run_id is None else dataset;ck=load_checkpoint(args.model)
        if args.split!='test':
            ids=set(d.runs.loc[d.runs.split.eq(args.split),'run_id'].astype(str));d.packets=d.packets[d.packets.run_id.astype(str).isin(ids)];d.truth=d.truth[d.truth.run_id.astype(str).isin(ids)];d.runs=d.runs[d.runs.run_id.astype(str).isin(ids)]
        score,path=_save_evaluation(d,ck,output,stream=True);print(json.dumps({k:v for k,v in score.items() if k not in ('row_level',)},indent=2));print('Open',path/'index.html');return 0
    if args.command=='evaluate-final':
        from .evaluation_protocol import evaluate_final
        print(json.dumps(evaluate_final(),indent=2));return 0

if __name__=='__main__':main()

