import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from henachel.but_v2_daily import outcome_rows,seal
from henachel.but_v2_prospective import summarize
from henachel.but_v2_inputs import prepare

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shadow_stage',ROOT/'scripts/stage_but_v2_shadow.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)

def test_only_successful_main_global_run_can_supply_inputs():
    run=dict(head_branch='main',status='completed',conclusion='success',path='.github/workflows/Workflow global Henachel — POINTS.yml',event='schedule')
    stage.validate_run(run)
    for changes in [dict(head_branch='research'),dict(conclusion='failure'),dict(event='pull_request'),dict(path='evil.yml')]:
        with pytest.raises(ValueError):stage.validate_run(dict(run,**changes))

def test_no_replacement_of_missing_top10_or_fake_single_day_ci():
    rows=pd.DataFrame([dict(id_match=1,id_joueur=i,date_match='2026-01-01',probability=.5,v1_probability=.4,
        rank=i+1,v1_rank=i+1,settlement_status='UNRESOLVED' if i==0 else 'WIN',y=np.nan if i==0 else 1,odds_decimal=np.nan) for i in range(12)])
    result=summarize(rows)
    assert result['top10']['v2']['selected']==10
    assert result['top10']['v2']['settled']==9  # rank 11 does not replace rank 1
    assert result['paired']['ll']['ci95'] is None and result['roi'] is None
    rows['settlement_status']='UNRESOLVED';rows['y']=np.nan
    assert summarize(rows)['settled']==0

def test_official_outcome_identity_participation_and_unfinished_game():
    box=dict(id=1,gameState='OFF',homeTeam={'abbrev':'A'},awayTeam={'abbrev':'B'},playerByGameStats={
        'homeTeam':{'forwards':[dict(playerId=2,toi='00:00',goals=0),dict(playerId=3,toi='12:30',goals=1)]}})
    rows=outcome_rows(box,1,'2026-01-02T00:00Z','https://api-web.nhle.com/v1/gamecenter/1/boxscore')
    assert rows[0]['toi_min']==0 and rows[1]['toi_min']==12.5
    with pytest.raises(ValueError):outcome_rows(box,2,'','')
    assert outcome_rows(dict(box,gameState='LIVE'),1,'','')==[]

def test_adapter_exact_schedule_and_odds_contract(tmp_path):
    root=tmp_path/'upstream'
    for path in ['data/raw','data/final','outputs/structured_bookmaker']:(root/path).mkdir(parents=True,exist_ok=True)
    ref=pd.DataFrame([dict(id_match=1,id_joueur=2,nom='A',date_match='2026-01-01',training_cutoff='2026-01-01',team_player_match='AAA',
        adversaire_match='BBB',start_time_utc='2026-01-01T23:00Z',model_name='GOAL17_OPERATIONAL_REFIT',probability_goal=.2,generated_at_utc='2026-01-01T12:00Z')])
    ref.to_csv(root/'outputs/predictions_upcoming_goal_research.csv',index=False)
    match=pd.DataFrame([dict(id_match=1,saison=20252026,id_equipe_domicile='AAA',id_equipe_exterieur='BBB',status='FUT',schedule_state='OK',game_type=2,
                            date_match='2026-01-01',start_time_utc='2026-01-01T23:00Z')])
    match.to_csv(root/'data/raw/matchs.csv',index=False);(root/'data/final/base_features_context_v2.csv').write_text('fixture\n')
    slate,r=prepare(root,tmp_path/'inputs')
    assert slate.id_joueur.tolist()==r.id_joueur.tolist()==[2]
    assert slate.buts.isna().all() and slate.is_home_player.tolist()==[1]
    match['id_equipe_domicile']='WRONG';match.to_csv(root/'data/raw/matchs.csv',index=False)
    with pytest.raises(ValueError,match='team mismatch'):prepare(root,tmp_path/'bad')

def test_workflow_is_disabled_without_explicit_repository_variable():
    text=(ROOT/'.github/workflows/but-v2-shadow.yml').read_text()
    assert "vars.HENACHEL_BUT_V2_SHADOW_ENABLED == 'true'" in text
    assert 'contents: read' in text and 'actions: read' in text
    assert 'contents: write' not in text.split('  backup:')[0]
    assert 'contents: write' in text.split('  backup:')[1]
    assert 'group: but-v2-shadow-state' in text and 'cancel-in-progress: false' in text
    assert 'schedule:' not in text

def test_daily_pipeline_freezes_first_capture_and_reports_unresolved(tmp_path,monkeypatch):
    import json
    from henachel import but_v2_daily as module
    from test_but_v2 import sample
    day=pd.Timestamp.now(tz='UTC').normalize().tz_localize(None)+pd.Timedelta(days=1)
    h=sample(200);h['date_match']=day-pd.Timedelta(days=200)+pd.to_timedelta(h.id_match,unit='D')
    s=sample(1);s['date_match']=day.strftime('%Y-%m-%d');s['id_match']=999
    s[['buts','tirs','temps_de_glace','temps_pp']]=np.nan
    s['start_time_utc']=(day.tz_localize('UTC')+pd.Timedelta(hours=23)).isoformat();s['status']='FUT';s['schedule_state']='OK'
    p=json.loads((ROOT/'configs/but_v2_prospective.json').read_text())
    ref=s[['id_match','id_joueur','start_time_utc']].copy();ref['probability_goal']=.25
    ref['config_sha']=p['v1_config_sha'];ref['generated_at_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    def adapter(upstream,out):
        out.mkdir(parents=True);h.to_csv(out/'history.csv',index=False);(out/'provenance.json').write_text('{}')
        return s,ref
    monkeypatch.setattr(module,'prepare',adapter)
    monkeypatch.setattr(module,'collect_outcomes',lambda *args:(pd.DataFrame(columns=module.OUTCOME_COLUMNS),[]))
    state=tmp_path/'state';config=ROOT/'configs/but_v2_prospective.json'
    first=module.daily(tmp_path,state,dict(id=1,head_sha='fixture'),None,config)
    assert first[0]['status']=='CAPTURED'
    capture=state/'captures'/s.date_match.iloc[0]/'predictions.csv';before=capture.read_bytes()
    second=module.daily(tmp_path,state,dict(id=2,head_sha='fixture'),None,config)
    assert second[0]['status']=='ALREADY_CAPTURED_FIRST_PAIR_RETAINED' and capture.read_bytes()==before
    summary=json.loads((state/'batches/2/cumulative/report.json').read_text())
    assert summary['unresolved']==4 and summary['top10']['v2']['settled']==0
    assert (state/'archive_manifest.json').exists()
    assert len(list((state/'training_sources').glob('*.gz')))==1

def test_final_assessment_cannot_promote_or_run_early():
    import json
    from henachel.but_v2_prospective import final_assessment
    protocol=json.loads((ROOT/'configs/but_v2_prospective.json').read_text())
    rows=pd.DataFrame([dict(date_match='2026-01-01',id_match=1,id_joueur=i,rank=i+1,v1_rank=i+1,
        probability=.2,v1_probability=.2,y=i%2,settlement_status='WIN' if i%2 else 'LOSS',odds_decimal=np.nan) for i in range(12)])
    with pytest.raises(ValueError,match='window'):final_assessment(rows,protocol,'2026-01-31')
    result=final_assessment(rows,protocol,'2026-08-01')
    assert not result['promotion_authorized'] and not result['sample_floors_met'] and not result['score_gate']


def test_pre_t0_registration_matches_frozen_files_and_features():
    import hashlib,json
    from henachel.but_v2 import FEATURES
    registration=json.loads((ROOT/'docs/but_v2/PRE_T0_REGISTRATION.json').read_text())
    assert registration['features']==FEATURES
    assert not registration['promotion_authorized']
    for name,digest in registration['file_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
