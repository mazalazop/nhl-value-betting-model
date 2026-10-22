"""Isolated daily shadow engine. Sources are copied, predictions never backfilled."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from henachel.but_v2_inputs import prepare
from henachel.but_v2_shadow import capture,report,utc
from henachel.but_v2_prospective import summarize,write_daily
from henachel.but_v2_validation import write

OUTCOME_COLUMNS=['id_match','id_joueur','status','buts','source','observed_at_utc','toi_min','team_player_match']

def outcome_rows(box,game_id,observed,url):
    if str(box.get('id'))!=str(game_id):raise ValueError('NHL boxscore ID mismatch')
    if box.get('gameState') not in {'FINAL','OFF'}:return []
    result=[]
    for side in ['homeTeam','awayTeam']:
        team=box[side]['abbrev'];stats=box.get('playerByGameStats',{}).get(side,{})
        for player in stats.get('forwards',[])+stats.get('defense',[]):
            value=player.get('toi');toi=None
            if value is not None:
                minutes,seconds=value.split(':');toi=int(minutes)+int(seconds)/60
            result.append(dict(id_match=game_id,id_joueur=player['playerId'],status=box['gameState'],buts=player.get('goals'),
                source=url,observed_at_utc=observed,toi_min=toi,team_player_match=team))
    return result

def collect_outcomes(rows,state,session):
    raw=state/'raw';raw.mkdir(exist_ok=True)
    pointers=state/'official';pointers.mkdir(exist_ok=True)
    result=[];errors=[];clock=pd.Timestamp.now(tz='UTC')
    for gid,group in rows.groupby('id_match'):
        start=utc(group.start_time_utc)[0]
        if start>=clock:continue
        pointer=pointers/f'{gid}.json';envelope=None
        if pointer.exists():envelope=json.loads(pointer.read_text())
        # Recheck final data for seven calendar days, preserving every response revision.
        if envelope is None or envelope['document'].get('gameState') not in {'OFF','FINAL'} or clock-start<pd.Timedelta(days=7):
            url=f'https://api-web.nhle.com/v1/gamecenter/{int(gid)}/boxscore'
            try:
                response=session.get(url,timeout=35);response.raise_for_status();doc=response.json()
                observed=pd.Timestamp.now(tz='UTC').isoformat()
                outcome_rows(doc,int(gid),observed,url)  # validate BEFORE replacing a cache pointer
                envelope=dict(document=doc,observed_at_utc=observed,source=url)
                encoded=json.dumps(envelope,sort_keys=True).encode();digest=hashlib.sha256(encoded).hexdigest()
                path=raw/(digest+'.json')
                if not path.exists():path.write_bytes(encoded)
                write(pointer,envelope)
            except Exception as exc:errors.append(dict(game_id=int(gid),error=str(exc)))
        if envelope:result.extend(outcome_rows(envelope['document'],int(gid),envelope['observed_at_utc'],envelope['source']))
    return pd.DataFrame(result,columns=OUTCOME_COLUMNS),errors

def seal(state):
    files={str(p.relative_to(state)):hashlib.sha256(p.read_bytes()).hexdigest() for p in state.rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
    write(state/'archive_manifest.json',dict(files=files))

def daily(upstream,state,run,session,protocol_path):
    state=Path(state);state.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(protocol_path).read_text())
    # Freeze algorithms/config and protocol, then fail on unreviewed drift across days.
    identity={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in protocol['locked_files']}
    lock=dict(protocol=protocol,files=identity)
    lockpath=state/'protocol_lock.json'
    if lockpath.exists() and json.loads(lockpath.read_text())!=lock:raise ValueError('Frozen prospective protocol/code drift')
    if not lockpath.exists():write(lockpath,lock)
    batch=state/'batches'/str(run['id']);attempt=0
    while batch.exists():
        attempt+=1;batch=state/'batches'/f"{run['id']}-r{attempt}"
    batch.mkdir(parents=True,exist_ok=False)
    write(state/'latest_batch.json',dict(path=str(batch.relative_to(state))))
    write(batch/'upstream_run.json',run)
    events=[];error=None;day=None
    try:
        slate,ref=prepare(upstream,batch/'inputs')
        day=str(slate.date_match.iloc[0]);destination=state/'captures'/day
        if destination.exists():events.append(dict(date=day,status='ALREADY_CAPTURED_FIRST_PAIR_RETAINED'))
        elif (utc(slate.start_time_utc)<=pd.Timestamp.now(tz='UTC')).any():events.append(dict(date=day,status='MISSED_PREGAME_WINDOW_NO_BACKFILL'))
        else:
            past=pd.read_csv(batch/'inputs/history.csv',low_memory=False)
            dates=pd.to_datetime(past.date_match);previous=dates[dates<pd.Timestamp(day)]
            if previous.nunique()<160 or (pd.Timestamp(day)-previous.max()).days>7:raise ValueError('Insufficient/stale training history')
            provenance=json.loads((batch/'inputs/provenance.json').read_text());provenance['upstream_run_id']=run['id'];provenance['head_sha']=run['head_sha']
            staging=batch/'candidate'
            capture(past,slate,ref,staging,provenance)
            # Daily fit must consume exactly the approved V1 algorithm.
            if not ref.config_sha.eq(protocol['v1_config_sha']).all():raise ValueError('V1 algorithm changed')
            destination.parent.mkdir(exist_ok=True);shutil.move(str(staging),str(destination))
            events.append(dict(date=day,status='CAPTURED',n=len(ref)))
    except Exception as exc:
        error=exc;events.append(dict(date=day,status='CAPTURE_FAILED',error=str(exc)))
    write(batch/'capture_status.json',events)
    # Preserve exact training source once per content hash, compressed; avoid intraday duplication.
    history_file=batch/'inputs/history.csv'
    if history_file.exists():
        import gzip
        digest=hashlib.sha256(history_file.read_bytes()).hexdigest()
        sources=state/'training_sources';sources.mkdir(exist_ok=True)
        target=sources/(digest+'.csv.gz')
        if not target.exists():
            with history_file.open('rb') as inp,gzip.open(target,'wb',compresslevel=1) as out:shutil.copyfileobj(inp,out)
        write(batch/'inputs/history_reference.json',dict(sha256=digest,path=str(target.relative_to(state))))
        history_file.unlink()  # newly generated local copy only; source artifact is never changed
    capture_dirs=sorted((state/'captures').glob('*')) if (state/'captures').exists() else []
    if capture_dirs:
        allrows=pd.concat([pd.read_csv(c/'predictions.csv') for c in capture_dirs],ignore_index=True)
        outcomes,errors=collect_outcomes(allrows,state,session)
        write(batch/'collection_errors.json',errors)
        outcomes.to_csv(batch/'outcomes.csv',index=False)
        settlements=[]
        for c in capture_dirs:
            out=batch/'reports'/c.name
            report(c,outcomes,out);settlements.append(pd.read_csv(out/'settlement.csv'))
        combined=pd.concat(settlements,ignore_index=True)
        aggregate=batch/'cumulative';aggregate.mkdir()
        summary=summarize(combined);write(aggregate/'report.json',summary);write_daily(combined,summary,aggregate)
        combined.to_csv(aggregate/'settlement.csv',index=False)
        accepted_dates=sorted(combined.date_match.unique())
        age=(pd.Timestamp.now(tz='UTC').normalize().tz_localize(None)-pd.Timestamp(accepted_dates[0])).days+1
        end=pd.Timestamp(accepted_dates[0])+pd.Timedelta(days=protocol['final_calendar_day']-1)
        primary=combined[pd.to_datetime(combined.date_match)<=end]
        primary_summary=summarize(primary);write(aggregate/'primary_fixed_window.json',primary_summary)
        dated_events=[e for p in (state/'batches').glob('*/capture_status.json') for e in json.loads(p.read_text())]
        expected={e['date'] for e in dated_events if e.get('date') and pd.Timestamp(e['date'])<=end}
        coverage=len(set(primary.date_match))/len(expected) if expected else None
        unknown=sum(not e.get('date') for e in dated_events)
        positive=int(primary.loc[primary.settlement_status.isin(['WIN','LOSS']),'y'].sum())
        sample_ok=(primary_summary['settled']>=protocol['min_rows'] and primary.date_match.nunique()>=protocol['min_dates']
                   and positive>=protocol['min_positive_outcomes'] and primary_summary['coverage']>=protocol['min_settlement_coverage'])
        write(aggregate/'checkpoint.json',dict(calendar_day=age,first_review_due=age>=30,
            final_review_due=age>=protocol['final_calendar_day'],primary_end_date=str(end.date()),promotion_authorized=False,
            sample_floors_met=sample_ok,observed_day_capture_coverage=coverage,unclassified_batches=unknown,
            coverage_requires_external_schedule_audit=True,
            note='One final assessment on the fixed 180-day cohort; counts alone never authorize promotion'))
    seal(state)
    if error:raise error
    return events
