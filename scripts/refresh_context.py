"""Refresh annotations from archived predictions; never fit, bet, or change odds.

--dry-run always uses a SQLite backup at a distinct isolated output path.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.context import ContextStore,clock,upcoming_baselines,validate_observation
from henachel.context_sources import collect_context
from henachel.matching import utc


def run(archive,output,dry_run=False,enable_pp=False,window_hours=18,min_interval_minutes=60,session=None):
    archive=Path(archive).resolve();output=Path(output).resolve()
    if not archive.exists():return dict(status='no_archived_predictions',http_requests=0)
    if dry_run:
        isolated=output/'context_dryrun.sqlite3'
        if isolated.resolve()==archive or isolated.exists():raise ValueError('Dry-run needs a new isolated destination')
        output.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(f'file:{archive}?mode=ro',uri=True) as source:
            with sqlite3.connect(isolated) as dest:source.backup(dest)
        archive=isolated
    store=ContextStore(archive);started=clock()
    try:
        store.seed_baselines(started)
        baselines=upcoming_baselines(store,started,window_hours)
        if not baselines:return dict(status='no_games_in_window',http_requests=0,executed_at_utc=started)
        game_ids={b['model_baseline']['game_id'] for b in baselines}
        prior=store.rows('context_observations',game_ids);latest={}
        for r in prior:latest[r['game_id']]=max(latest.get(r['game_id'],utc(r['collected_at_utc'])),utc(r['collected_at_utc']))
        baselines=[b for b in baselines if b['model_baseline']['game_id'] not in latest or
                   (utc(started)-latest[b['model_baseline']['game_id']]).total_seconds()>=min_interval_minutes*60]
        if not baselines:return dict(status='cadence_not_due',http_requests=0,executed_at_utc=started)
        http=session or requests.Session();calls=[]
        # Count actual HTTP requests, including retries, not cache hits.
        original=http.get
        def counted(url,*args,**kwargs):calls.append(url);return original(url,*args,**kwargs)
        http.get=counted
        try:rows,errors=collect_context(baselines,http,output/'cache',enable_pp)
        finally:http.get=original
        finished=clock();accepted=[]
        for r in rows:
            try:validate_observation(r,finished);accepted.append(r)
            except (ValueError,TypeError,KeyError) as exc:errors.append(dict(source=r.get('source'),error=str(exc)))
        count=store.append_observations(accepted,finished)
        blocked={e['game_id'] for e in errors if e.get('detail')=='Target event changed or started'}
        due_games={b['model_baseline']['game_id'] for b in baselines}-blocked
        changed=store.update_views(finished,game_ids=due_games)
        latest_views={v['baseline_id']:v for v in store.rows('context_views',game_ids)}
        current=[v for v in latest_views.values() if utc(finished)<utc(v['event_start_utc'])]
        current.sort(key=lambda v:(v['market'],v['model_rank'],v['game_id'],v['player_id']))
        report=dict(schema_version=1,status='collected',executed_at_utc=started,completed_at_utc=finished,
                    dry_run=dry_run,http_requests=len(calls),observations_added=count,views_added=len(changed),
                    games=len({b['model_baseline']['game_id'] for b in baselines}),
                    coverage=dict(Counter((r['kind']+':'+r['evidence_status']) for r in [validate_observation(x,finished) for x in accepted])),
                    unknown_stat_fields=dict(Counter(k for r in accepted if isinstance(r.get('value'),dict) for k,v in r['value'].items() if v is None)),
                    errors=errors,model_probabilities_modified=False)
        # Immutable per-execution output; the canonical history remains SQLite.
        dest=output/finished.replace(':','-');dest.mkdir(parents=True,exist_ok=False)
        for name,value in [('report',report),('contextual_picks',current)]:
            (dest/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False))
        return report
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('outputs/history/prospective.sqlite3'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/context'))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--enable-pp',action='store_true')
    p.add_argument('--window-hours',type=float,default=18);p.add_argument('--min-interval-minutes',type=float,default=60)
    a=p.parse_args()
    if not 0<a.window_hours<=48 or not 15<=a.min_interval_minutes<=1440:p.error('Invalid collection cadence/window')
    try:print(json.dumps(run(a.archive,a.output_dir,a.dry_run,a.enable_pp,a.window_hours,a.min_interval_minutes)))
    except Exception as exc:
        # Workflow marks this isolated step continue-on-error, never changes models.
        print(json.dumps(dict(status='context_failed',error=type(exc).__name__,detail=str(exc)[:300])),file=sys.stderr)
        raise SystemExit(1)
