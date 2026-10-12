"""Fail-closed prospective lock and immutable pregame prediction batches (research)."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research_structural import protocol, digest


def lock_candidate(path, candidate, artifacts, now=None):
    p=protocol();now=pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None:raise ValueError('UTC lock time required')
    if now>=pd.Timestamp(p['prospective']['lock_deadline_exclusive']):raise ValueError('Prospective lock deadline passed')
    if candidate.get('variant') not in p['variants'] or candidate.get('target') not in ['POINT','GOAL']:raise ValueError('Unknown candidate')
    if candidate.get('features')!=p['variants'][candidate['variant']] or candidate.get('parameters')!=p['parameters']:raise ValueError('Candidate differs from preregistration')
    if not {'model','calibrator','baseline_model','baseline_calibrator','code','development_report'}.issubset(artifacts):raise ValueError('Missing immutable artifacts')
    report=json.loads(Path(artifacts['development_report']).read_text())
    if report['candidate']!=candidate['variant'] or report['target']!=candidate['target']:raise ValueError('Candidate not selected by development criteria')
    payload=dict(candidate=candidate,locked_at_utc=now.isoformat(),prospective=p['prospective'],artifacts={name:dict(path=str(file),sha256=digest(file)) for name,file in artifacts.items()},production_approved=False)
    with Path(path).open('x') as f:json.dump(payload,f,indent=2)
    return payload


def read_lock(path):
    lock=json.loads(Path(path).read_text())
    for item in lock['artifacts'].values():
        if digest(item['path'])!=item['sha256']:raise ValueError('Frozen artifact changed')
    return lock


def archive_predictions(lock_path, rows, destination, now=None):
    lock=read_lock(lock_path);rows=rows.copy()
    required=['id_match','id_joueur','date_match','event_start_utc','collected_at_utc','probability','baseline_probability','player_smoothed_probability']
    if set(required)-set(rows) or rows.empty:raise ValueError('Incomplete prediction snapshot')
    if rows.duplicated(['id_match','id_joueur']).any():raise ValueError('Duplicate predictions')
    for key in ['id_match','id_joueur']:
        ids=pd.to_numeric(rows[key],errors='raise')
        if not np.isfinite(ids).all() or not ids.gt(0).all() or not ids.mod(1).eq(0).all():raise ValueError('Invalid NHL identity')
    for field in ['probability','baseline_probability','player_smoothed_probability']:
        values=pd.to_numeric(rows[field],errors='raise')
        if not np.isfinite(values).all() or not values.between(0,1).all():raise ValueError('Invalid probabilities')
    event=pd.to_datetime(rows.event_start_utc,utc=True,errors='raise');collected=pd.to_datetime(rows.collected_at_utc,utc=True,errors='raise')
    if event.isna().any() or collected.isna().any() or not collected.lt(event).all() or not collected.ge(pd.Timestamp(lock['locked_at_utc'])).all():raise ValueError('Prediction not strictly pregame after lock')
    dates=pd.to_datetime(rows.date_match,errors='raise')
    if not (dates.ge(lock['prospective']['start']) & dates.lt(lock['prospective']['end_exclusive'])).all():raise ValueError('Outside prospective window')
    # One immutable daily file, all candidates for that NHL date; exact reruns only.
    if dates.nunique()!=1:raise ValueError('One NHL date per batch')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    path=destination/f"{lock['candidate']['target']}_{dates.iloc[0].strftime('%Y-%m-%d')}.json"
    payload=dict(lock_sha256=digest(lock_path),rows=json.loads(rows[required].sort_values(['id_match','id_joueur']).to_json(orient='records')))
    if path.exists():
        if json.loads(path.read_text())!=payload:raise ValueError('Cannot replace archived predictions')
    else:
        observed=pd.Timestamp(now or datetime.now(timezone.utc))
        if observed.tzinfo is None or not collected.le(observed).all() or not event.gt(observed).all():raise ValueError('Archive must be written before actual puck drop')
        with path.open('x') as f:json.dump(payload,f,indent=2)
    return path


def open_final_once(lock_path, output, observations, now=None):
    lock=read_lock(lock_path);now=pd.Timestamp(now or datetime.now(timezone.utc))
    if now.tzinfo is None or now<pd.Timestamp(lock['prospective']['end_exclusive'],tz='UTC'):raise ValueError('Prospective period not closed')
    dates=pd.to_datetime(observations.date_match,errors='raise')
    if not (dates.ge(lock['prospective']['start']) & dates.lt(lock['prospective']['end_exclusive'])).all():raise ValueError('Final observations outside prospective window')
    keys=['id_match','id_joueur']
    if observations.duplicated(keys).any():raise ValueError('Duplicate final observations')
    minimum=lock['prospective']
    if 'position' not in observations or observations.position.eq('D').sum()<minimum['minimum_defender_games']:raise ValueError('Insufficient defender coverage; inconclusive')
    if len(observations)<minimum['minimum_player_games'] or dates.nunique()<minimum['minimum_dates']:raise ValueError('Insufficient prospective coverage; inconclusive')
    # Consume before any metric access; interruption cannot enable reselection.
    with Path(output).open('x') as f:json.dump(dict(lock_sha256=digest(lock_path),opened_at_utc=now.isoformat(),status='consumed'),f)
