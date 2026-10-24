"""Separate A01 registry: immutable imports, evidence, decisions and report revisions."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from henachel import but_v2_participation_a01 as part
from henachel.but_v2_prospective import summarize as compare, write_daily
from henachel.but_v2_validation import paired


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=payload:raise ValueError('Immutable collision: '+str(path))
    else:
        with path.open('x') as f:f.write(payload)


def seal(state):
    state=Path(state)
    files={str(p.relative_to(state)):sha(p) for p in state.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    (state/'archive_manifest.json').write_text(json.dumps(dict(files=files),sort_keys=True))


def verify(state):
    state=Path(state)
    if any(p.is_symlink() for p in state.rglob('*')):raise ValueError('Registry symlink')
    expected=json.loads((state/'archive_manifest.json').read_text())['files']
    actual={str(p.relative_to(state)):sha(p) for p in state.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    if expected!=actual:raise ValueError('Registry integrity failure')


def validate_start(start, release, as_of):
    if not start.get('enabled') or start.get('mode')!='PROSPECTIVE':raise ValueError('A01 disabled')
    registered,t0=part.utc(start['registered_at_utc']),part.utc(start['t0_utc'])
    if not registered<t0<=part.utc(as_of):raise ValueError('T0 not preregistered or not reached')
    if not start.get('approved_by') or not start.get('approval_reference'):raise ValueError('Approval required')
    if start['first_nhl_date']<=start['excluded_through_nhl_date']:raise ValueError('Examined dates not excluded')
    if pd.Timestamp(start['first_nhl_date']).date()<t0.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).date():
        raise ValueError('First NHL date predates T0')
    if start['release_sha256']!=part.digest(release):raise ValueError('Release not preregistered')
    for name,digest in release['files'].items():
        if sha(name)!=digest:raise ValueError('Code/protocol drift: '+name)


def eligible(rows, start):
    if start['mode']=='EXPLORATORY':return True
    return bool((pd.to_datetime(rows.generated_at_utc,utc=True)>=pd.Timestamp(start['t0_utc'])).all()
                and (rows.date_match>=start['first_nhl_date']).all()
                and (rows.date_match>start['excluded_through_nhl_date']).all())


def semantic(evidence):
    return part.digest({k:dict(source=v.get('source'),document=v.get('document')) for k,v in evidence.items()})


def final_report(rows, coverage, protocol, as_of, first_date):
    end=pd.Timestamp(first_date)+pd.Timedelta(days=179)
    if pd.Timestamp(as_of).tz_convert('UTC').tz_localize(None).normalize()<=end:
        return dict(status='WINDOW_NOT_COMPLETE',promotion_authorized=False)
    cohort=rows[(rows.date_match>=first_date)&(rows.date_match<=str(end.date()))]
    valid=cohort[cohort.settlement_status.isin(['WIN','LOSS'])].copy()
    if valid.empty:return dict(status='INSUFFICIENT_DATA',promotion_authorized=False)
    ref=valid.copy();ref['probability']=ref.v1_probability
    comparison=paired(ref,valid,replicates=10000,seed=20261020,family_size=2)
    descriptive=compare(cohort)
    def ece(name):
        return sum(b['count']*abs(b['mean_probability']-b['observed_rate']) for b in descriptive[name]['calibration'] if b['count'])/len(valid)
    p=protocol['inherited_original_requirements']
    return dict(status='HUMAN_REVIEW_REQUIRED',promotion_authorized=False,
        sample_floors_met=bool(len(valid)>=p['min_rows'] and valid.date_match.nunique()>=p['min_dates'] and valid.y.sum()>=p['min_positive_outcomes']),
        information_gate=coverage['information_gate_98'],participant_results_gate=coverage['participant_result_gate_98'],
        score_gate=bool(comparison['ll']['delta']<=-p['minimum_log_loss_gain'] and comparison['br']['delta']<=-p['minimum_brier_gain'] and
                        comparison['ll']['family_adjusted_ci'][1]<0 and comparison['br']['family_adjusted_ci'][1]<0),
        calibration_gate=bool(ece('v2')<=.03 and ece('v2')-ece('v1')<=.005),comparison=comparison,
        remaining_gates=['external eligible-day coverage >=95%', 'segment robustness review', 'explicit human approval'])


def cycle(original, state, evidence, as_of, run_id, mode='EXPLORATORY', start=None, release=None, decisions=None):
    original,state=Path(original).resolve(),Path(state).resolve()
    if original==state or original in state.parents or state in original.parents:raise ValueError('Separate registries required')
    verify(original)
    original_hash=sha(original/'archive_manifest.json')
    if mode not in {'EXPLORATORY','PROSPECTIVE'}:raise ValueError('Invalid mode')
    if mode=='PROSPECTIVE':validate_start(start,release,as_of)
    else:start=dict(mode='EXPLORATORY',study_id='A01_EXPLORATORY',t0_utc=None)
    if not str(run_id).replace('-','').replace('_','').isalnum():raise ValueError('Unsafe run identity')
    if state.exists():verify(state)
    else:state.mkdir(parents=True)
    write(state/'registration.json',start)
    if release and mode=='PROSPECTIVE':write(state/'release.json',release)
    batch=state/'batches'/str(run_id)
    if batch.exists():
        if not (batch/'complete.json').exists():raise ValueError('Incomplete prior attempt: retain it and retry with a new run ID')
        old=json.loads((batch/'request.json').read_text())
        request=dict(evidence=part.digest(evidence),as_of=as_of,original_manifest=original_hash,decisions=part.digest(decisions or []))
        if old!=request:raise ValueError('Run ID reused for different request')
        return json.loads((batch/'report.json').read_text())
    baseline_path=state/'baselines.json'
    baselines=json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    try:
        batch.mkdir(parents=True)
        if release:write(batch/'release.json',release)
        write(batch/'request.json',dict(evidence=part.digest(evidence),as_of=as_of,original_manifest=original_hash,decisions=part.digest(decisions or [])))
        imported=[];excluded=[]
        for cap in sorted((original/'captures').iterdir()):
            prediction=cap/'predictions.csv'
            if not prediction.exists():continue
            manifest=json.loads((cap/'manifest.json').read_text())
            if sha(prediction)!=manifest['predictions_sha256']:raise ValueError('Original capture changed')
            rows=pd.read_csv(prediction)
            if rows.duplicated(['id_match','id_joueur']).any():raise ValueError('Duplicate original capture')
            if not rows.archive_group.eq('CANONICAL').all() or not rows.scope.eq('BUT_V2_SHADOW').all():raise ValueError('Wrong source universe')
            if not (pd.to_datetime(rows.generated_at_utc,utc=True)<pd.to_datetime(rows.start_time_utc,utc=True)).all():raise ValueError('Not pregame')
            if (pd.to_datetime(rows.generated_at_utc,utc=True)>pd.Timestamp(as_of)).any():raise ValueError('Future capture')
            if not eligible(rows,start):excluded.append(cap.name);continue
            target=state/'captures'/cap.name
            if target.exists():
                if sha(target/'predictions.csv')!=sha(prediction):raise ValueError('First capture immutable')
            else:
                target.mkdir(parents=True);shutil.copyfile(prediction,target/'predictions.csv');shutil.copyfile(cap/'manifest.json',target/'manifest.json')
            imported.append(rows)
        write(batch/'imports.json',dict(excluded_dates=excluded,original_manifest_sha256=original_hash))
        if not imported:
            result=dict(mode=mode,n=0,status='NO_ADMISSIBLE_CAPTURE',promotion_authorized=False)
            write(batch/'report.json',result);write(batch/'complete.json',dict(complete=True));return result
        rows=pd.concat(imported,ignore_index=True)
        if rows.duplicated(['id_match','id_joueur']).any():raise ValueError('Duplicate registry identity')
        protocol=json.loads(Path('configs/but_v2_a01_proposed.json').read_text())
        first_date=start.get('first_nhl_date',rows.date_match.min())
        end=str((pd.Timestamp(first_date)+pd.Timedelta(days=179)).date())
        closed=mode=='PROSPECTIVE' and pd.Timestamp(as_of).date()>pd.Timestamp(end).date()
        decision_map={}
        for path in sorted((state/'decisions').glob('*.json')) if (state/'decisions').exists() else []:
            d=json.loads(path.read_text());decision_map[d['case_id']]=d
        for d in decisions or []:
            if d['choice'] not in {'ACCEPT_CURRENT','KEEP_PREVIOUS'} or not d.get('reviewer') or not d.get('rationale') or not d.get('review_reference'):
                raise ValueError('Explicit signed review required')
            if part.utc(d['reviewed_at_utc'])>part.utc(as_of):raise ValueError('Future review')
            if d['case_id'] in decision_map and decision_map[d['case_id']]!=d:raise ValueError('Decision cannot be overwritten')
            decision_map[d['case_id']]=d
        classifications=[];used=set();evidence_hashes={};previous_cache={}
        for row in rows.to_dict('records'):
            key=f"{row['id_match']}-{row['id_joueur']}";gid=str(row['id_match']);current=evidence.get(gid,{})
            if gid not in evidence_hashes:evidence_hashes[gid]=part.archive_evidence(state/'evidence',current)
            current_hash=evidence_hashes[gid]
            if key in baselines and baselines[key] not in previous_cache:
                previous_cache[baselines[key]]=part.read_evidence(state/'evidence'/(baselines[key]+'.json'))
            previous=previous_cache[baselines[key]] if key in baselines else None
            c=part.audit([row],{gid:current},as_of,{gid:previous} if previous else None,closure=closed)[0]
            if c['reason']=='OFFICIAL_REVISION_REQUIRES_REVIEW':
                case_id=part.digest(dict(key=key,previous=semantic(previous),current=semantic(current)))
                write(state/'cases'/case_id/(current_hash+'.json'),dict(case_id=case_id,classification=c))
                c['case_id']=case_id
                if case_id in decision_map:
                    d=decision_map[case_id]
                    if any(part.utc(v['observed_at_utc'])>part.utc(d['reviewed_at_utc']) for v in current.values()):
                        # Later polls of the same semantic document are allowed only for an already recorded decision.
                        if not (state/'decisions'/(case_id+'.json')).exists():raise ValueError('Review predates evidence')
                    write(state/'decisions'/(case_id+'.json'),d);used.add(case_id)
                    selected=c['proposed_classification'] if d['choice']=='ACCEPT_CURRENT' else c['previous_classification']
                    c=dict(selected,review_decision=case_id,reason='REVIEWED_'+d['choice'])
                    if d['choice']=='ACCEPT_CURRENT':baselines[key]=current_hash
            elif c['category'] not in {part.UNKNOWN,part.PENDING}:
                baselines[key]=current_hash
            classifications.append(c)
        if any(d['case_id'] not in used for d in decisions or []):raise ValueError('Decision does not match a pending case')
        pointer=json.loads((original/'latest_batch.json').read_text())['path']
        official=pd.read_csv(original/pointer/'cumulative/settlement.csv')
        original_rows=rows[['id_match','id_joueur']].merge(official[['id_match','id_joueur','settlement_status']],validate='one_to_one',how='left')
        original_settled=int(original_rows.settlement_status.isin(['WIN','LOSS']).sum())
        coverage=part.summarize(classifications,original_settled,len(rows))
        merged=rows.drop(columns=['status'],errors='ignore').merge(pd.DataFrame(classifications),on=['id_match','id_joueur'],validate='one_to_one')
        merged['y']=np.where(merged.category.eq(part.KNOWN),pd.to_numeric(merged.goals,errors='coerce').gt(0).astype(float),np.nan)
        merged['settlement_status']=np.where(merged.y.isna(),'UNRESOLVED',np.where(merged.y.eq(1),'WIN','LOSS'))
        original_global_settled=int(official.settlement_status.isin(['WIN','LOSS']).sum())
        result=dict(mode=mode,study_id=start['study_id'],coverage=coverage,comparison=compare(merged),promotion_authorized=False,
                    original_study=dict(n=len(official),settled=original_global_settled,
                                        coverage=original_global_settled/len(official) if len(official) else None),
                    by_date={str(day):compare(group) for day,group in merged.groupby('date_match')},
                    segments={str(team):compare(group) for team,group in merged.groupby('team_player_match')},
                    segment_limitations='Home/away and season type require a separately verified schedule join; mandatory human robustness review remains open.')
        if mode=='PROSPECTIVE':
            window=merged[(merged.date_match>=first_date)&(merged.date_match<=end)]
            selected_keys=set(zip(window.id_match,window.id_joueur))
            classes=[c for c in classifications if (c['id_match'],c['id_joueur']) in selected_keys]
            original_window=original_rows.merge(window[['id_match','id_joueur']],on=['id_match','id_joueur'])
            cov=part.summarize(classes,int(original_window.settlement_status.isin(['WIN','LOSS']).sum()),len(window))
            assessment=state/'confirmatory/assessment.json'
            if assessment.exists():
                result['confirmatory']=json.loads(assessment.read_text())
                result['post_closure_revision_note']='Original confirmatory assessment retained; current daily metrics are descriptive corrections only.'
            else:
                result['confirmatory']=final_report(window,cov,protocol,as_of,first_date)
                if closed:
                    result['confirmatory']['basis']=dict(run_id=str(run_id),as_of=as_of,first_date=first_date,end_date=end,
                                                         classification_sha256=part.digest(classes))
                    write(assessment,result['confirmatory'])
        write(batch/'classifications.json',classifications);merged.to_csv(batch/'settlement.csv',index=False)
        write(batch/'report.json',result);write_daily(merged,result['comparison'],batch)
        with (batch/'README.md').open('a') as stream:
            stream.write('\nOriginal study (all dates): '+json.dumps(result['original_study'])+'\n')
            stream.write('\nA01 '+mode+' — dénominateurs et catégories :\n```json\n'+json.dumps(coverage,indent=2)+'\n```\n')
        baseline_path.write_text(json.dumps(baselines,sort_keys=True))
        (state/'latest_batch.json').write_text(json.dumps(dict(path=str(batch.relative_to(state)))))
        if sha(original/'archive_manifest.json')!=original_hash:raise ValueError('Original registry was modified')
        write(batch/'complete.json',dict(complete=True))
        return result
    finally:
        seal(state)
