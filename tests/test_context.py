import concurrent.futures
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest
from henachel.context import ContextStore, validate_observation, upcoming_baselines
from henachel.context_sources import aggregate, collect_context
from test_prospective_pp import player, NOW, START


def observation(**changes):
    row=dict(player(),kind='availability',source='fixture_official',source_url='https://example.test/update',
             collected_at_utc=NOW,source_timestamp='2026-10-13T11:50:00Z',value='OUT',evidence_status='CONFIRMED')
    row.update(changes);return row


def seeded(path):
    store=ContextStore(path)
    store.archive.append('prediction','pregame',[dict(player(),target='GOAL',probability=.25,collected_at_utc=NOW)],now=NOW)
    store.seed_baselines(NOW);return store


@pytest.mark.parametrize('change', [dict(collected_at_utc=START),dict(collected_at_utc='2026-10-14T01:00:00Z')])
def test_no_post_puck_context(change):
    with pytest.raises(ValueError,match='pregame'):validate_observation(observation(**change),change['collected_at_utc'])


@pytest.mark.parametrize('stamp',[None,'2026-10-13','2026-10-14T01:00:00Z','2026-10-12T00:00:00Z'])
def test_unproven_source_never_confirmed(stamp):
    assert validate_observation(observation(source_timestamp=stamp),NOW)['evidence_status']=='UNKNOWN'


def test_missing_pp_never_pp2():
    r=validate_observation(observation(kind='pp_unit',value=None),NOW)
    assert r['value'] is None and r['evidence_status']=='UNKNOWN'


def test_append_only_updates_conflicts_and_probability_invariance(tmp_path):
    store=seeded(tmp_path/'archive.db');base=store.rows('context_baselines')[0]
    first=store.update_views(NOW)[0]
    assert first['active'] and first['model_baseline']['probability']==.25
    assert store.append_observations([observation()],NOW)==1
    assert store.append_observations([observation()],NOW)==0
    excluded=store.update_views(NOW)[0]
    assert not excluded['active'] and excluded['previous_view_id']==first['view_id']
    assert store.update_views(NOW)==[]
    store.append_observations([observation(source='another',value='AVAILABLE')],NOW)
    conflict=store.update_views(NOW)[0]
    assert conflict['active'] and conflict['operational_status']=='ACTIVE_REVIEW_CONFLICT'
    assert len(store.rows('context_observations'))==2
    assert len(store.rows('context_views'))==3
    for view in store.rows('context_views'):
        assert view['model_baseline']==base['model_baseline']
        assert view['model_rank']==base['model_rank']
    with pytest.raises(Exception):store.db.execute('DELETE FROM context_views')
    store.db.rollback();store.close()


@pytest.mark.parametrize('level',['PROBABLE','UNKNOWN'])
def test_uncertain_absence_not_excluded(tmp_path,level):
    store=seeded(tmp_path/'a.db');store.append_observations([observation(evidence_status=level)],NOW)
    assert store.update_views(NOW)[0]['active'];store.close()


def test_stale_absence_loses_confirmed_exclusion(tmp_path):
    store=seeded(tmp_path/'a.db');store.append_observations([observation()],NOW)
    assert not store.update_views(NOW)[0]['active']
    assert store.update_views('2026-10-13T15:00:00Z')[0]['active']
    assert store.update_views(START)==[]
    assert upcoming_baselines(store,START)==[];store.close()


def test_concurrent_identical_observations_and_views_no_duplicates(tmp_path):
    path=tmp_path/'a.db';store=seeded(path);store.close()
    def writer(_):
        s=ContextStore(path)
        try:s.append_observations([observation()],NOW);s.update_views(NOW)
        finally:s.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(writer,range(3)))
    s=ContextStore(path)
    assert len(s.rows('context_observations'))==1 and len(s.rows('context_views'))==1;s.close()


def box():
    return dict(observed_at='2026-10-13T11:00:00Z',document=dict(id=2026020009,gameDate='2026-10-12',startTimeUTC='2026-10-12T23:00:00Z',gameState='FINAL',
        homeTeam=dict(abbrev='TOR',score=2,sog=30),awayTeam=dict(abbrev='MTL',score=1,sog=20),
        playerByGameStats={'homeTeam':{'forwards':[dict(playerId=8470001,toi='18:30',sog=3)]}}))


@pytest.mark.parametrize('change',[dict(id=2026020010),dict(gameState='LIVE'),dict(startTimeUTC='2026-10-14T00:00:00Z')])
def test_aggregate_refuses_target_live_future(change):
    b=box();b['document'].update(change)
    with pytest.raises(ValueError):aggregate([b],2026020010,'TOR',8470001,NOW)


def test_final_observed_after_cutoff_refused_and_missing_not_zero():
    b=box();b['observed_at']='2026-10-13T13:00:00Z'
    with pytest.raises(ValueError):aggregate([b],2026020010,'TOR',8470001,NOW)
    result=aggregate([box()],2026020010,'TOR',8470001,NOW)
    assert result['player']['pp_toi_minutes'] is None
    assert result['player']['toi_minutes']==18.5


def runner():
    spec=importlib.util.spec_from_file_location('refresh_context',Path(__file__).parents[1]/'scripts/refresh_context.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_empty_or_started_slate_no_http(tmp_path,monkeypatch):
    m=runner();path=tmp_path/'archive.db';s=seeded(path);s.close();http=Mock()
    monkeypatch.setattr(m,'clock',lambda:START)
    assert m.run(path,tmp_path/'out',session=http)['status']=='no_games_in_window'
    http.get.assert_not_called()
    assert collect_context([],http,tmp_path)==([],[]);http.get.assert_not_called()


def test_dryrun_never_writes_source_and_failure_preserves_predictions(tmp_path,monkeypatch):
    m=runner();path=tmp_path/'production.db';s=seeded(path);s.close();before=path.read_bytes()
    monkeypatch.setattr(m,'clock',lambda:NOW)
    monkeypatch.setattr(m,'collect_context',lambda *args:([observation(source_timestamp=None)], [{'source':'failed','error':'offline'}]))
    report=m.run(path,tmp_path/'sandbox',dry_run=True)
    assert path.read_bytes()==before and report['errors']
    s=ContextStore(tmp_path/'sandbox/context_dryrun.sqlite3')
    assert s.rows('context_views')[0]['model_baseline']['probability']==.25
    assert s.rows('context_views')[0]['active'];s.close()
    with pytest.raises(ValueError):m.run(path,tmp_path/'sandbox',dry_run=True)


def test_source_failure_unknown_other_players_continue(tmp_path):
    import requests
    http=Mock();http.get.side_effect=requests.ConnectionError('offline')
    rows,errors=collect_context([dict(model_baseline=player())],http,tmp_path,now_fn=lambda:NOW)
    assert errors
    assert all(r['evidence_status']=='UNKNOWN' for r in rows)


def test_same_time_conflicting_source_retained(tmp_path):
    s=seeded(tmp_path/'a.db')
    s.append_observations([observation(),observation(value='AVAILABLE')],NOW)
    assert s.update_views(NOW)[0]['operational_status']=='ACTIVE_REVIEW_CONFLICT';s.close()


def test_intraday_workflow_has_no_models_and_is_serialized():
    import yaml
    path=Path(__file__).parents[1]/'.github/workflows/Workflow global Henachel — POINTS.yml'
    w=yaml.safe_load(path.read_text());job=w['jobs']['context_update']
    assert w['concurrency']['group']=='henachel-points-history'
    assert w['concurrency']['cancel-in-progress'] is False
    commands='\n'.join(s.get('run','') for s in job['steps'])
    assert 'predict_goal' not in commands and '05_predict' not in commands and 'collect_unibet' not in commands
    assert 'context_only' in job['if']
    assert 'context_only' in w['jobs']['model_points']['if']
    step=next(s for s in job['steps'] if 'refresh_context.py' in s.get('run',''))
    assert step['continue-on-error'] is True


def test_failed_refresh_does_not_retract_fresh_confirmed_absence(tmp_path):
    s=seeded(tmp_path/'a.db');s.append_observations([observation()],NOW)
    later='2026-10-13T12:20:00Z'
    s.append_observations([observation(collected_at_utc=later,source_timestamp=None,value=None)],later)
    view=s.update_views(later)[0]
    assert not view['active'] and len(view['context_review'])==2
    assert s.update_views('2026-10-13T15:00:00Z')[0]['active'];s.close()


def test_normalized_observation_retry_is_idempotent(tmp_path):
    s=seeded(tmp_path/'a.db');raw=observation()
    assert s.append_observations([raw],NOW)==1
    assert s.append_observations([validate_observation(raw,NOW)],NOW)==0;s.close()


def test_nhl_pp_clamped_pages_are_complete_and_real_zero_preserved(tmp_path,monkeypatch):
    from urllib.parse import urlparse,parse_qs
    import henachel.context_sources as sources
    calls=[]
    def get(session,url,cache,ttl):
        calls.append(url)
        if '/club-schedule' in url:
            doc={'games':[dict(id=2026020010,gameState='FUT',startTimeUTC=START),
                          dict(id=2026020009,gameState='FINAL',startTimeUTC='2026-10-12T23:00:00Z',gameType=2)]}
        elif '/boxscore' in url:return dict(box(),url=url)
        elif '/skater/powerplay' in url:
            start=int(parse_qs(urlparse(url).query)['start'][0])
            rows=[dict(gameId=2026020009,playerId=8470001+i,gameDate='2026-10-12',teamAbbrev='TOR',ppTimeOnIce=0) for i in range(101)]
            doc=dict(total=101,data=rows[start:start+100])
        else:doc=dict(total=0,data=[])
        return dict(url=url,observed_at=NOW,document=doc)
    monkeypatch.setattr(sources,'get_json',get);monkeypatch.setattr(sources,'clock',lambda:NOW)
    rows,errors=sources.collect_context([dict(model_baseline=player())],Mock(),tmp_path,now_fn=lambda:NOW)
    assert not errors
    usage=next(r for r in rows if r['kind']=='player_usage')
    assert usage['value']['pp_toi_minutes']==0
    assert usage['value']['pp_share'] is None
    assert len([url for url in calls if '/skater/powerplay' in url])==2


def test_archive_backup_includes_context_tables(tmp_path):
    from test_runtime import module
    path=tmp_path/'archive.db';s=seeded(path)
    s.append_observations([observation()],NOW);s.update_views(NOW);s.close()
    result=module('seal_prospective_archive').seal(path)
    assert result['context_counts']=={'context_baselines':1,'context_observations':1,'context_views':1}


def test_archived_model_rank_preserved(tmp_path):
    store=ContextStore(tmp_path/'rank.db')
    row=dict(player(),target='GOAL',probability=.25,model_rank=42,collected_at_utc=NOW)
    store.archive.append('prediction','pregame',[row],now=NOW)
    store.seed_baselines(NOW)
    view=store.update_views(NOW)[0]
    assert view['model_rank']==42
    assert view['model_baseline']==row
    store.close()
