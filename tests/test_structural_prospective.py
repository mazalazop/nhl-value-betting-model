import json
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import structural_prospective as s
from research_structural import protocol


def fixture(tmp_path):
    p=protocol();artifacts={}
    for name in ['model','calibrator','baseline_model','baseline_calibrator','code','development_report']:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(dict(candidate='V1',target='POINT')));artifacts[name]=path
    candidate=dict(variant='V1',target='POINT',features=p['variants']['V1'],parameters=p['parameters'])
    return candidate,artifacts


def locked(tmp_path):
    c,a=fixture(tmp_path);path=tmp_path/'lock.json';s.lock_candidate(path,c,a,now='2026-10-12T12:00:00Z');return path


def rows():
    return pd.DataFrame([dict(id_match=1,id_joueur=2,date_match='2026-11-02',event_start_utc='2026-11-03T00:00:00Z',collected_at_utc='2026-11-02T12:00:00Z',probability=.5,baseline_probability=.48,player_smoothed_probability=.4)])


def test_lock_immutable_and_deadline_enforced(tmp_path):
    c,a=fixture(tmp_path);path=tmp_path/'lock.json'
    s.lock_candidate(path,c,a,now='2026-10-12T12:00:00Z')
    with pytest.raises(FileExistsError):s.lock_candidate(path,c,a,now='2026-10-12T12:00:00Z')
    with pytest.raises(ValueError,match='deadline'):s.lock_candidate(tmp_path/'late.json',c,a,now='2026-11-01T00:00:00Z')


def test_unselected_candidate_refused(tmp_path):
    c,a=fixture(tmp_path);a['development_report'].write_text(json.dumps(dict(candidate=None,target='POINT')))
    with pytest.raises(ValueError,match='not selected'):s.lock_candidate(tmp_path/'lock.json',c,a,now='2026-10-12T12:00:00Z')


def test_artifact_modification_detected(tmp_path):
    path=locked(tmp_path);(tmp_path/'model.json').write_text('changed')
    with pytest.raises(ValueError,match='changed'):s.read_lock(path)


def test_snapshot_rerun_idempotent_and_no_overwrite(tmp_path):
    path=locked(tmp_path);data=rows();dest=tmp_path/'snapshots';out=s.archive_predictions(path,data,dest,now='2026-11-02T13:00:00Z')
    assert s.archive_predictions(path,data,dest,now='2026-11-02T13:00:00Z')==out and len(list(dest.iterdir()))==1
    data.loc[0,'probability']=.9
    with pytest.raises(ValueError,match='replace'):s.archive_predictions(path,data,dest,now='2026-11-02T13:00:00Z')


@pytest.mark.parametrize('field,value',[('collected_at_utc','2026-11-03T00:00:00Z'),('collected_at_utc',None),('probability',float('nan')),('date_match','2024-02-01')])
def test_invalid_or_late_predictions_refused(tmp_path,field,value):
    path=locked(tmp_path);data=rows();data.loc[0,field]=value
    with pytest.raises(ValueError):s.archive_predictions(path,data,tmp_path/'snapshots')


def test_prospective_not_opened_early_or_reopened(tmp_path):
    path=locked(tmp_path);obs=pd.DataFrame(dict(position=['D']*4000+['F']*16000,id_match=range(20000),id_joueur=range(20000),date_match=[pd.Timestamp('2026-11-01')+pd.Timedelta(days=i%100) for i in range(20000)]))
    with pytest.raises(ValueError,match='not closed'):s.open_final_once(path,tmp_path/'opened.json',obs,now='2027-04-19T00:00:00Z')
    s.open_final_once(path,tmp_path/'opened.json',obs,now='2027-04-20T00:00:00Z')
    with pytest.raises(FileExistsError):s.open_final_once(path,tmp_path/'opened.json',obs,now='2027-04-20T00:00:00Z')


def test_cannot_backdate_archive_after_game(tmp_path):
    path=locked(tmp_path)
    with pytest.raises(ValueError,match='actual puck drop'):
        s.archive_predictions(path,rows(),tmp_path/'snapshots',now='2026-11-04T00:00:00Z')


def test_prospective_requires_preregistered_defender_coverage(tmp_path):
    path=locked(tmp_path)
    obs=pd.DataFrame(dict(id_match=range(20000),id_joueur=range(20000),position=['F']*20000,
                         date_match=[pd.Timestamp('2026-11-01')+pd.Timedelta(days=i%100) for i in range(20000)]))
    with pytest.raises(ValueError,match='defender coverage'):
        s.open_final_once(path,tmp_path/'opened.json',obs,now='2027-04-20T00:00:00Z')
    assert not (tmp_path/'opened.json').exists()
