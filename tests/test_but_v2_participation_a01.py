import copy
import hashlib
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'model'))
from henachel import but_v2_participation_a01 as m

ASOF='2026-10-23T09:00:00Z'
ROW=dict(id_match=123,id_joueur=999,team_player_match='AAA',start_time_utc='2026-10-22T23:00:00Z')


def sources():
    box=dict(id=123,gameState='OFF',homeTeam=dict(id=1,abbrev='AAA'),awayTeam=dict(id=2,abbrev='BBB'),playerByGameStats={})
    play={k:copy.deepcopy(v) for k,v in box.items() if k!='playerByGameStats'}
    play['rosterSpots']=[]
    play['plays']=[dict(typeDescKey='game-end')]
    for side,tid in [('homeTeam',1),('awayTeam',2)]:
        players=[dict(playerId=tid*100+i,toi='10:00',goals=0) for i in range(18)]
        keeper=dict(playerId=tid*100+90)
        box['playerByGameStats'][side]=dict(forwards=players[:12],defense=players[12:],goalies=[keeper])
        play['rosterSpots'] += [dict(playerId=p['playerId'],teamId=tid,positionCode='C') for p in players]
        play['rosterSpots'].append(dict(playerId=keeper['playerId'],teamId=tid,positionCode='G'))
    return {e:dict(source=f'https://api-web.nhle.com/v1/gamecenter/123/{e}',observed_at_utc='2026-10-23T08:00:00Z',document=d)
            for e,d in [('boxscore',box),('play-by-play',play)]}


def player(e):
    return e['boxscore']['document']['playerByGameStats']['homeTeam']['forwards'][0]


def classify(e,**kwargs):
    return m.classify(dict(ROW,**kwargs),e,ASOF)


def test_complete_absence_and_empty_sources():
    assert classify(sources())['category']==m.ABSENT
    e=sources();e['play-by-play']['document']['rosterSpots']=[]
    assert classify(e)['category']==m.UNKNOWN
    e=sources();e['boxscore']['document']['playerByGameStats']={}
    assert classify(e)['category']==m.UNKNOWN
    assert classify({})['category']==m.PENDING  # not mature at 10h, never absent
    late=m.classify(ROW,{},'2026-10-23T12:00:00Z')
    assert late['category']==m.UNKNOWN and late['evaluable']


def test_truncated_both_sources_never_prove_absence():
    e=sources();e['boxscore']['document']['playerByGameStats']['homeTeam']['forwards'].pop()
    e['play-by-play']['document']['rosterSpots']=[p for p in e['play-by-play']['document']['rosterSpots'] if p['playerId']!=111]
    assert classify(e)['category']==m.UNKNOWN


@pytest.mark.parametrize('value,expected',[(0,m.KNOWN),(None,m.MISSING),(-1,m.MISSING),(1.5,m.MISSING)])
def test_participant_result(value,expected):
    e=sources();player(e)['goals']=value
    r=classify(e,id_joueur=100)
    assert r['category']==expected
    assert r['goals']==(0 if expected==m.KNOWN else None)


@pytest.mark.parametrize('value',['00:00',None,'broken','10:99'])
def test_zero_unknown_toi_not_absence(value):
    e=sources();player(e)['toi']=value
    assert classify(e,id_joueur=100)['category']==m.UNKNOWN


def test_postponement_and_closure_denominator():
    e=sources()
    for v in e.values():v['document'].update(gameState='FUT',gameScheduleState='PPD')
    r=m.classify(ROW,e,'2026-10-23T12:00:00Z')
    assert r['category']==m.PENDING and r['evaluable']
    stats=m.summarize([r],0,1)
    assert stats['unresolved_information_share']==1 and not stats['information_gate_98']
    assert m.classify(ROW,{},ASOF,closure=True)['evaluable']


def test_contradictions_duplicates_and_provenance():
    e=sources();e['play-by-play']['document']['rosterSpots'].append(copy.deepcopy(e['play-by-play']['document']['rosterSpots'][0]))
    assert classify(e)['category']==m.UNKNOWN
    e=sources();e['play-by-play']['document']['gameState']='LIVE'
    assert classify(e)['reason']=='CONTRADICTORY_GAME_STATES'
    e=sources();e['boxscore']['source']='https://untrusted.test'
    assert classify(e)['category']==m.UNKNOWN
    e=sources();e['boxscore']['observed_at_utc']='2026-10-24T00:00:00Z'
    assert classify(e)['category']==m.UNKNOWN
    with pytest.raises(ValueError,match='Duplicate'):
        m.audit([ROW,ROW],{'123':sources()},ASOF)


def test_official_correction_retains_both_versions(tmp_path):
    before=sources();after=copy.deepcopy(before);player(after)['goals']=1
    original=copy.deepcopy(before)
    row=dict(ROW,id_joueur=100)
    r=m.audit([row],{'123':after},ASOF,previous={'123':before})[0]
    assert r['category']==m.UNKNOWN and r['reason']=='OFFICIAL_REVISION_REQUIRES_REVIEW'
    assert r['previous_classification']['goals']==0 and r['proposed_classification']['goals']==1
    assert before==original
    a=m.archive_evidence(tmp_path,before);b=m.archive_evidence(tmp_path,after)
    assert a!=b and len(list(tmp_path.iterdir()))==2
    assert m.archive_evidence(tmp_path,before)==a
    assert m.read_evidence(tmp_path/(a+'.json'))==before
    (tmp_path/(a+'.json')).write_text('{}')
    with pytest.raises(ValueError,match='checksum'):m.read_evidence(tmp_path/(a+'.json'))


def test_diagnostic_does_not_replace_original_denominator():
    r=[dict(id_match=1,id_joueur=i,category=c,evaluable=True) for i,c in enumerate([m.KNOWN,m.ABSENT,m.MISSING,m.UNKNOWN,m.PENDING])]
    s=m.summarize(r,1,5)
    assert s['original_coverage']==.2 and s['information_coverage']==.4
    assert s['participant_result_availability']==.5 and s['nonparticipant_share']==.2
    assert s['genuinely_unknown_share']==.4 and s['pending_evaluable_share']==.2
    assert not s['promotion_authorized']
    assert m.summarize([],0,0)['information_coverage'] is None


def test_configuration_not_active_and_original_registration_unchanged():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/but_v2_a01_proposed.json').read_text())
    assert not cfg['enabled'] and cfg['t0_utc'] is None
    assert cfg['min_information_coverage']==cfg['min_participant_result_availability']==.98
    # Existing registration is independently tested by test_but_v2_daily.
    files=json.loads((root/'docs/but_v2/PRE_T0_REGISTRATION.json').read_text())
    assert files
    original=json.loads((root/'configs/but_v2_prospective.json').read_text())
    assert original['min_settlement_coverage']==.98
    assert not any('participation_a01' in (root/p).read_text() for p in original['locked_files'] if p.endswith('.py'))


def test_evidence_time_rollback_is_not_silent():
    previous=sources();current=copy.deepcopy(previous)
    current['boxscore']['observed_at_utc']='2026-10-23T07:59:00Z'
    r=m.audit([ROW],{'123':current},ASOF,previous={'123':previous})[0]
    assert r['category']==m.UNKNOWN and r['reason']=='EVIDENCE_TIME_ROLLBACK'


def test_activation_and_threshold_drift_rejected():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/but_v2_a01_proposed.json').read_text())
    m.validate_preparation_config(cfg)
    for key,value in [('enabled',True),('t0_utc','2026-10-24T00:00:00Z'),('min_information_coverage',.97)]:
        with pytest.raises(ValueError,match='activation'):
            m.validate_preparation_config(dict(cfg,**{key:value}))


def test_recovery_from_nonfinal_is_not_official_correction():
    old=sources()
    for value in old.values():value['document']['gameState']='LIVE'
    r=m.audit([dict(ROW,id_joueur=100)],{'123':sources()},ASOF,previous={'123':old})[0]
    assert r['category']==m.KNOWN


def test_cli_preserves_inputs_and_never_overwrites_output(tmp_path):
    import subprocess
    import pandas as pd
    root=Path(__file__).resolve().parents[1]
    capture=tmp_path/'capture';capture.mkdir()
    predictions=capture/'predictions.csv';pd.DataFrame([ROW]).to_csv(predictions,index=False)
    manifest=capture/'manifest.json';manifest.write_text(json.dumps(dict(predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())))
    report=tmp_path/'original.json';report.write_text(json.dumps(dict(n=1,settled=0,coverage=0.0)))
    evidence=tmp_path/'evidence.json';evidence.write_text(json.dumps({'123':sources()}))
    inputs=[predictions,manifest,report,evidence];before={p:p.read_bytes() for p in inputs}
    output=tmp_path/'audit'
    command=[sys.executable,str(root/'scripts/audit_but_v2_participation_a01.py'),
             '--predictions',str(predictions),'--original-report',str(report),
             '--evidence',str(evidence),'--as-of',ASOF,'--output',str(output)]
    subprocess.run(command,check=True,capture_output=True)
    assert all(p.read_bytes()==payload for p,payload in before.items())
    stats=json.loads((output/'metrics.json').read_text())
    assert stats['original_coverage']==0 and stats['nonparticipant_share']==1
    assert not stats['confirmatory_assessment_authorized']
    archived={p:p.read_bytes() for p in output.rglob('*') if p.is_file()}
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert all(p.read_bytes()==payload for p,payload in archived.items())
    assert len(list((output/'evidence').glob('*.json')))==2


def test_cli_rejects_original_registry_as_output(tmp_path):
    import subprocess
    import pandas as pd
    root=Path(__file__).resolve().parents[1]
    predictions=tmp_path/'predictions.csv';pd.DataFrame([ROW]).to_csv(predictions,index=False)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())))
    (tmp_path/'protocol_lock.json').write_text('{}')
    command=[sys.executable,str(root/'scripts/audit_but_v2_participation_a01.py'),
             '--predictions',str(predictions),'--original-report',str(tmp_path/'report.json'),
             '--output',str(tmp_path/'new-audit')]
    run=subprocess.run(command,capture_output=True,text=True)
    assert run.returncode!=0 and 'outside the original registry' in run.stderr
    assert not (tmp_path/'new-audit').exists()


def test_play_activity_and_missing_end_prevent_false_absence():
    e=sources();e['play-by-play']['document']['plays'].append(dict(details=dict(shootingPlayerId=999)))
    r=classify(e)
    assert r['category']==m.UNKNOWN and r['reason']=='PLAY_ACTIVITY_CONTRADICTS_MISSING_ROSTER'
    e=sources();e['play-by-play']['document']['plays']=[]
    assert classify(e)['category']==m.UNKNOWN
