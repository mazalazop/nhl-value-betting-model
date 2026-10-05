"""Run existing entrypoints in an isolated /private/tmp copy, never publish or place bets."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import requests

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_ENTRYPOINTS = {
    '00_refresh_sources','00a_refresh_pp_stats','00b_build_base_match_fusionnee',
    '00c_refresh_team_standings','01_build_base_features','05_predict_upcoming_games',
    '06_match_model_to_unibet_odds','07_build_daily_bets','09_settle_previous_bets',
}


def location():
    root=Path(json.loads((ROOT/'outputs/operational_validation_location.json').read_text())['root']).resolve()
    if root.parent!=Path('/private/tmp') or not root.name.startswith('henachel-real-validation-'):
        raise ValueError('Validation must use the isolated temporary checkout')
    return root


def entrypoint(root,name,args=()):
    if name not in ALLOWED_ENTRYPOINTS:
        raise ValueError('Only isolated, non-publishing pipeline stages are allowed')
    log=root/'evidence'/f'{name}.log'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONWARNINGS='error')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('a') as out:
        result=subprocess.run([sys.executable,str(root/'model'/f'{name}.py'),*args],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
    record=dict(entrypoint=name,args=list(args),started_at=started,ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode,log=str(log))
    with (root/'evidence/executions.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
    if result.returncode:raise SystemExit(result.returncode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['init','probe','sources','pp','stats','standings','features','predict','history','dry-run','persistence'])
    parser.add_argument('--date',required=True);args=parser.parse_args()
    datetime.date.fromisoformat(args.date)
    if args.phase=='init':
        import tempfile,shutil
        root=Path(tempfile.mkdtemp(prefix='henachel-real-validation-',dir='/private/tmp'))
        for directory in ['model','scripts']:
            for source in (ROOT/directory).rglob('*.py'):
                target=root/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
        for directory in ['data/raw','data/final','outputs','evidence']:(root/directory).mkdir(parents=True,exist_ok=True)
        (ROOT/'outputs').mkdir(exist_ok=True)
        (ROOT/'outputs/operational_validation_location.json').write_text(json.dumps(dict(root=str(root))))
        print(root);return
    root=location()
    if args.phase=='probe':
        endpoints=dict(seasons='https://api-web.nhle.com/v1/season',schedule='https://api-web.nhle.com/v1/club-schedule-season/TOR/20262027',roster='https://api-web.nhle.com/v1/roster/TOR/current',standings='https://api-web.nhle.com/v1/standings/2026-10-01')
        for name,url in endpoints.items():
            response=requests.get(url,timeout=30);record=dict(url=url,status=response.status_code,server_date=response.headers.get('Date'))
            if response.ok:
                payload=response.json();(root/'evidence'/f'probe_{name}.json').write_text(json.dumps(payload));record['keys']=list(payload)[:12] if isinstance(payload,dict) else f'list {len(payload)}'
            print(json.dumps(record),flush=True)
    elif args.phase=='history':
        import importlib.util
        import pandas as pd
        import shutil
        sys.path.insert(0,str(root/'model'))
        spec=importlib.util.spec_from_file_location('real_stats',root/'model/00b_build_base_match_fusionnee.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from henachel.data import final_mask
        from henachel.nhl_cache import BoxscoreCache
        matches=pd.read_csv(root/'data/raw/matchs.csv');final=matches[final_mask(matches)]
        stats=pd.read_csv(root/'evidence/remote_artifacts/model/data/raw/stats.csv',low_memory=False)
        stats=stats[stats.id_match.isin(final.id_match)].copy()
        missing=set(final.id_match)-set(stats.id_match)
        sample=set(final.sort_values('date_match').groupby('saison').head(1).id_match)|set(final.sort_values('date_match').groupby('saison').tail(1).id_match)
        session=module.build_session();cache=BoxscoreCache(root/'data/raw/boxscores');checks=[]
        for game in sorted(missing|sample):
            match=final[final.id_match.eq(game)].iloc[0]
            payload=cache.get(game,match.date_match,lambda gid:module.fetch_game_payload(session,gid))
            rows=module.parse_game_to_stats_rows(payload,int(game),str(match.date_match),str(match.saison))
            live=pd.DataFrame(rows)
            previous=stats[stats.id_match.eq(game)]
            comparison=previous[['id_joueur','points']].merge(live[['id_joueur','points']],on='id_joueur',how='outer',suffixes=('_artifact','_live'))
            checks.append(dict(id_match=int(game),live_rows=len(live),artifact_rows=len(previous),point_differences=int(comparison.points_artifact.ne(comparison.points_live).sum())))
            stats=pd.concat([stats[~stats.id_match.eq(game)],live],ignore_index=True)
        # Dates come from the freshly retrieved NHL schedule, never from fabricated labels.
        stats['date_match']=stats.id_match.map(matches.set_index('id_match').date_match)
        base=module.build_base_match_fusionnee(stats,matches)
        stats.to_csv(root/'data/raw/stats.csv',index=False);base.to_csv(root/'data/raw/base_match_fusionnee.csv',index=False)
        standings=root/'evidence/remote_artifacts/model/data/raw/team_standings_daily.csv'
        shutil.copyfile(standings,root/'data/raw/team_standings_daily.csv')
        report=dict(rows=len(stats),final_games=len(final),stats_games=int(stats.id_match.nunique()),fresh_boxscores=cache.fetches,missing_games_filled=len(missing),checks=checks,source_artifact_id=11236911254)
        (root/'evidence/history_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    elif args.phase=='sources':entrypoint(root,'00_refresh_sources',['--min-season','20242025','--max-season','20262027','--skip-boxscore-supplement','--sleep-seconds','0.05'])
    elif args.phase=='pp':entrypoint(root,'00a_refresh_pp_stats',['--start-date','2024-09-01','--end-date',args.date,'--sleep-seconds','0.05'])
    elif args.phase=='stats':entrypoint(root,'00b_build_base_match_fusionnee',['--sleep-seconds','0.05'])
    elif args.phase=='standings':entrypoint(root,'00c_refresh_team_standings',['--sleep-seconds','0.05'])
    elif args.phase=='features':entrypoint(root,'01_build_base_features')
    elif args.phase=='predict':entrypoint(root,'05_predict_upcoming_games',['--target-date',args.date])
    elif args.phase=='dry-run':
        entrypoint(root,'06_match_model_to_unibet_odds',['--odds-json',str(root/'evidence/remote_artifacts/market/normalized_points_odds.json'),'--run-date',args.date])
        entrypoint(root,'07_build_daily_bets',['--run-date',args.date])
        simulate_publication(root)
    elif args.phase=='persistence':
        validate_persistence(root,args.date)


def simulate_publication(root):
    import importlib.util
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pandas as pd
    sys.path.insert(0,str(root/'model'))
    spec=importlib.util.spec_from_file_location('sandbox_publisher',root/'model/08_publish_to_google_sheet.py')
    publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
    daily=publisher.load_daily_bets(root/'outputs/07_daily_bets.csv')
    history=pd.read_csv(root/'outputs/history/master_daily_bets_history.csv')
    connection=Mock()
    for sheet_id,frame in enumerate([publisher.build_daily_display_df(daily),publisher.build_history_display_df(history)]):
        publisher.write_replace(SimpleNamespace(id=sheet_id,row_count=2000,col_count=60,spreadsheet=connection),frame)
    assert connection.batch_update.call_count==2
    report=dict(mode='simulated_only',daily_rows=len(daily),history_rows=len(history),mock_batches=2,network_writes=0)
    (root/'evidence/publication_simulation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


def validate_persistence(root,run_date):
    """Synthetic bet terms on real final NHL results, across independent 09 processes."""
    import shutil
    import pandas as pd
    sys.path.insert(0,str(root/'model'))
    from henachel.history import append_ledger,read_ledger
    from henachel.matching import COLUMNS
    sandbox=root/'evidence/persistence'
    sandbox.mkdir(exist_ok=True)
    stats=pd.read_csv(root/'data/raw/stats.csv',low_memory=False)
    chosen=pd.concat([stats[stats.points.eq(0)].tail(1),stats[stats.points.ge(1)].tail(1)])
    bets=[]
    for _,player in chosen.iterrows():
        bet=dict.fromkeys(COLUMNS)
        bet.update(bet_id=f'TEST_ONLY:{int(player.id_match)}:{int(player.id_joueur)}',id_match=int(player.id_match),id_joueur=int(player.id_joueur),bookmaker='TEST_ONLY',market='player_points',stat='points',threshold=1,outcome_key='1_plus',bet_status='pending',result='pending',run_date=run_date)
        bets.append(bet)
    pending=pd.DataFrame(bets)
    first=sandbox/'run1/history.csv';second=sandbox/'run2/history.csv'
    append_ledger(first,pending)
    def execute(path):
        entrypoint(root,'09_settle_previous_bets',['--history-csv',str(path),'--output-dir',str(path.parent),'--run-date',run_date])
    execute(first)
    second.parent.mkdir(exist_ok=True)
    shutil.copyfile(first,second)
    shutil.copyfile(first.with_name('settlement_revisions.csv'),second.with_name('settlement_revisions.csv'))
    before=read_ledger(second)
    append_ledger(second,pending)
    execute(second);execute(second)
    after=read_ledger(second)
    pd.testing.assert_frame_equal(before,after)
    revisions=pd.read_csv(second.with_name('settlement_revisions.csv'))
    assert len(after)==2 and not revisions.revision_id.duplicated().any()
    assert set(after.result)=={'win','loss'}
    # A fresh, disposable fixture is interrupted after its revision journal is saved.
    import tempfile
    interrupted=Path(tempfile.mkdtemp(prefix='interrupted-',dir=sandbox))/'history.csv'
    append_ledger(interrupted,pending)
    driver='''import importlib.util,sys
from pathlib import Path
script=Path(sys.argv[1]);sys.path.insert(0,str(script.parent))
spec=importlib.util.spec_from_file_location('interrupted_settlement',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.argv=sys.argv[1:]
original=module.atomic_csv
def crash(frame,path):
    if Path(path).name=='history.csv':raise OSError('EXPECTED_TEST_INTERRUPTION')
    return original(frame,path)
module.atomic_csv=crash
module.main()
'''
    with (sandbox/'interruption.log').open('w') as log:
        failed=subprocess.run([sys.executable,'-c',driver,str(root/'model/09_settle_previous_bets.py'),'--history-csv',str(interrupted),'--output-dir',str(interrupted.parent)],cwd=root,stdout=log,stderr=log)
    assert failed.returncode!=0
    assert read_ledger(interrupted).bet_status.eq('pending').all()
    assert interrupted.with_name('settlement_revisions.csv').exists()
    execute(interrupted);execute(interrupted)
    recovered=read_ledger(interrupted)
    journal=pd.read_csv(interrupted.with_name('settlement_revisions.csv'))
    assert len(journal)==2 and not journal.revision_id.duplicated().any()
    assert set(recovered.result)=={'win','loss'}
    assert recovered.set_index('bet_id').settled_at.equals(journal.set_index('bet_id').settled_at)
    report=dict(mode='synthetic_bets_real_NHL_results',independent_settlement_processes=6,expected_interrupted_processes=1,rows=len(after),revisions=len(revisions),results=after.result.tolist(),restoration_preserved=True,rerun_preserved=True,partial_interruption_recovered=True)
    (root/'evidence/persistence_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
