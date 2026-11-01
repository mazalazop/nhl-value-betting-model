"""Network-free operational regressions. No prediction/model/protocol changes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import requests
from test_but_v2_a01_registry import original, m
from test_but_v2_participation_a01 import ROW, ASOF, sources, player

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_but_v2_a01 as runtime
import backup_but_v2_state as backup
from test_but_v2_backup import MemoryStore

LATER='2026-10-23T10:00:00Z'


def incomplete(kind):
    e=sources()
    if kind=='outage':return {}
    if kind=='empty':e['boxscore']['document']={};e['play-by-play']['document']={}
    if kind=='one_endpoint':del e['play-by-play']
    if kind=='partial':e['play-by-play']['document']['rosterSpots']=[]
    if kind=='missing_result':player(e)['goals']=None
    if kind=='zero_toi':player(e)['toi']='00:00'
    if kind=='unknown_toi':player(e)['toi']=None
    if kind=='desynchronized':e['boxscore']['observed_at_utc']='2026-10-23T08:30:00Z'
    if kind=='nonfinal':
        for envelope in e.values():envelope['document']['gameState']='LIVE'
    if kind=='malformed':e['play-by-play']['document']=[]
    return e


@pytest.mark.parametrize('kind',['outage','empty','one_endpoint','partial','missing_result','zero_toi','unknown_toi','desynchronized','nonfinal','malformed'])
def test_unreliable_fetch_never_opens_revision_or_replaces_baseline(tmp_path,kind):
    src=original(tmp_path);state=tmp_path/'state';good={'123':sources()}
    first=m.cycle(src,state,good,ASOF,'first')
    baseline=(state/'baselines.json').read_bytes()
    report=m.cycle(src,state,{'123':incomplete(kind)},LATER,'outage')
    again=m.cycle(src,state,{'123':incomplete(kind)},'2026-10-23T11:00:00Z','outage2')
    assert report['comparison']==again['comparison']==first['comparison']
    assert json.loads((state/'baselines.json').read_text())['123-100']==json.loads(baseline)['123-100']
    assert not list((state/'cases').glob('*'))
    classes=json.loads((state/'batches/outage/classifications.json').read_text())
    participant=next(c for c in classes if c['id_joueur']==100)
    assert participant['retained_previous'] and 'last_collection' in participant
    recovered=m.cycle(src,state,good,LATER,'recovered')
    assert recovered['comparison']==first['comparison']
    m.verify(state)


def test_same_evidence_different_evaluation_clock_and_recollection(tmp_path):
    src=original(tmp_path);state=tmp_path/'state';good={'123':sources()}
    m.cycle(src,state,good,ASOF,'first')
    changed=copy.deepcopy(good);player(changed['123'])['goals']=1
    m.cycle(src,state,changed,ASOF,'correction')
    proofs=list((state/'cases').glob('*/*.json'));assert len(proofs)==1
    contents={p:p.read_bytes() for p in proofs}
    m.cycle(src,state,changed,LATER,'correction-repeat')
    assert list((state/'cases').glob('*/*.json'))==proofs
    assert all(p.read_bytes()==b for p,b in contents.items())
    for e in changed['123'].values():e['observed_at_utc']='2026-10-23T09:30:00Z'
    m.cycle(src,state,changed,LATER,'recollected')
    assert len(list((state/'cases').glob('*/*.json')))==2
    assert len(list((state/'cases').iterdir()))==1
    classes=json.loads((state/'batches/recollected/classifications.json').read_text())
    assert next(c for c in classes if c['id_joueur']==100)['reason']=='OFFICIAL_REVISION_REQUIRES_REVIEW'
    assert all(p.read_bytes()==b for p,b in contents.items())


def test_partial_contradiction_is_not_a_certified_revision():
    current=sources();current['play-by-play']['document']['plays']=[];player(current)['goals']=2
    result=m.part.audit([dict(ROW,id_joueur=100)],{'123':current},ASOF,{'123':sources()})[0]
    assert result['goals']==0 and result['retained_previous']
    assert result['last_collection']['goals']==2


def test_uncertainty_without_previous_certificate_stays_unknown():
    for kind in ['empty','one_endpoint','partial','missing_result']:
        e=incomplete(kind)
        c=m.part.audit([ROW],{'123':e},LATER)[0]
        assert c['category'] not in {m.part.KNOWN,m.part.ABSENT}
    c=m.part.audit([ROW],{'123':incomplete('nonfinal')},LATER)[0]
    assert c['category']==m.part.PENDING


class Response:
    def __init__(self,status=200,document=None):self.status_code=status;self.document=document;self.headers={}
    def raise_for_status(self):
        if self.status_code>=400:raise requests.HTTPError(str(self.status_code))
    def json(self):
        if isinstance(self.document,Exception):raise self.document
        return self.document


class Session:
    def __init__(self,values):self.values=iter(values);self.calls=0
    def get(self,*args,**kwargs):
        self.calls+=1;value=next(self.values)
        if isinstance(value,Exception):raise value
        return value


@pytest.mark.parametrize('bad',[requests.Timeout('timeout'),requests.ConnectionError('offline'),*[Response(n) for n in [429,500,502,503,504]],Response(document=ValueError('invalid JSON')),Response(document={}),Response(document=[])])
def test_bounded_retry_and_recovery(bad):
    session=Session([bad,bad,Response(document=sources()['boxscore']['document'])]);waits=[]
    envelope,errors=runtime.fetch_official(session,'fixture',123,sleep=waits.append)
    assert len(errors)==2 and session.calls==3 and waits==[1,2]
    assert envelope['document']['id']==123
    session=Session([bad,bad,bad]);waits=[]
    envelope,errors=runtime.fetch_official(session,'fixture',123,sleep=waits.append)
    assert envelope is None and len(errors)==3 and waits==[1,2]


def test_permanent_http_error_not_retried():
    session=Session([Response(404)])
    envelope,errors=runtime.fetch_official(session,'fixture',123,sleep=lambda _:pytest.fail('unexpected retry'))
    assert envelope is None and len(errors)==1


def test_retry_after_is_bounded():
    bad=Response(429);bad.headers={'Retry-After':'600'};waits=[]
    runtime.fetch_official(Session([bad,bad,bad]),'fixture',123,sleep=waits.append)
    assert waits==[30,30]


def test_record_attempt_and_restore_after_intermediate_failure(tmp_path,monkeypatch):
    src=original(tmp_path);state=tmp_path/'state';e={'123':sources()}
    m.cycle(src,state,e,ASOF,'first');pred=(state/'captures/2026-10-22/predictions.csv').read_bytes()
    runtime.record_attempt(state,{'123':{}},[{'error':'fixture timeout'}],'broken',LATER)
    real_write=m.write
    def fail(path,value):
        if str(path).endswith('batches/broken/report.json'):raise RuntimeError('injected failure')
        real_write(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(m,'write',fail)
        with pytest.raises(RuntimeError,match='injected'):m.cycle(src,state,{'123':{}},LATER,'broken')
    m.verify(state)
    store=MemoryStore();receipt=backup.backup(store,state,'fixture','failed-run')
    restored=tmp_path/'restored';backup.restore(store,store.latest(),restored)
    assert receipt['remote_restore_verified'] and backup.inventory(restored)==backup.inventory(state)
    with pytest.raises(ValueError,match='Incomplete'):m.cycle(src,restored,e,LATER,'broken')
    m.cycle(src,restored,e,LATER,'recovery')
    assert (restored/'captures/2026-10-22/predictions.csv').read_bytes()==pred
    assert (restored/'collection-attempts/broken.json').exists()
    assert not (restored/'batches/broken/complete.json').exists()
    with pytest.raises(ValueError,match='reused'):m.cycle(src,restored,{},LATER,'recovery')


def test_capture_gate_filters_redundant_events_but_not_changed_capture():
    name='captures/2026-10-26/predictions.csv';item={'sha256':'one'}
    prior={'files':{name:item}}
    source={'files':{name:item,'batches/settlement/report.json':{'sha256':'new'}}}
    start=dict(first_nhl_date='2026-10-26',excluded_through_nhl_date='2026-10-25')
    assert not runtime.new_capture_available(source,prior,'PROSPECTIVE',start)
    assert not runtime.new_capture_available(source,prior,'PROSPECTIVE',start) # close repeated triggers
    source['files']['captures/2026-10-25/predictions.csv']=item
    assert not runtime.new_capture_available(source,prior,'PROSPECTIVE',start)
    source['files']['captures/2026-10-27/predictions.csv']=item
    assert runtime.new_capture_available(source,prior,'PROSPECTIVE',start)
    source={'files':{name:{'sha256':'changed'}}}
    assert runtime.new_capture_available(source,prior,'PROSPECTIVE',start) # must reach integrity rejection
    script=Path(runtime.__file__).read_text()
    assert "os.environ.get('GITHUB_EVENT_NAME')!='workflow_run'" in script
    workflow=Path('.github/workflows/but-v2-a01.yml').read_text()
    assert "45 7,11,13,17 * * *" in workflow and 'cancel-in-progress: false' in workflow


def test_patch_cannot_change_scientific_files(tmp_path,monkeypatch):
    release=json.loads(Path('configs/but_v2_a01_release.json').read_text())
    patch=json.loads(m.PATCH_PATH.read_text());patch['files']['configs/but_v2.json']={'before_sha256':'x','after_sha256':'y'}
    path=tmp_path/'patch.json';path.write_text(json.dumps(patch));monkeypatch.setattr(m,'PATCH_PATH',path)
    with pytest.raises(ValueError,match='scope'):m.validate_operational_files(release)
    patch['files'].pop('configs/but_v2.json');patch['files']['scripts/run_but_v2_a01.py']['after_sha256']='0'*64
    path.write_text(json.dumps(patch))
    with pytest.raises(ValueError,match='drift'):m.validate_operational_files(release)


def test_real_pending_correction_survives_outage_and_can_be_reviewed(tmp_path):
    src=original(tmp_path);state=tmp_path/'state';e={'123':sources()}
    m.cycle(src,state,e,ASOF,'first')
    corrected=copy.deepcopy(e);player(corrected['123'])['goals']=1
    m.cycle(src,state,corrected,ASOF,'correction')
    old_proofs={p:p.read_bytes() for p in (state/'cases').glob('*/*.json')}
    report=m.cycle(src,state,{'123':{}},LATER,'outage')
    assert report['coverage']['counts'][m.part.UNKNOWN]==1
    classes=json.loads((state/'batches/outage/classifications.json').read_text())
    pending=next(c for c in classes if 'case_id' in c)
    assert pending['reason']=='OFFICIAL_REVISION_REQUIRES_REVIEW'
    assert all(p.read_bytes()==value for p,value in old_proofs.items())
    assert len(list((state/'cases').glob('*/*.json')))==len(old_proofs)
    decision=dict(case_id=pending['case_id'],choice='ACCEPT_CURRENT',reviewer='fixture',rationale='Verified correction',
                  reviewed_at_utc=LATER,review_reference='fixture://authorized')
    accepted=m.cycle(src,state,{'123':{}},LATER,'review',decisions=[decision])
    assert accepted['comparison']['v2']['hit_rate']==1
    again=m.cycle(src,state,{'123':{}},LATER,'after-review')
    assert again['comparison']['v2']['hit_rate']==1


def test_collection_budget_prevents_unbounded_outage_calls():
    session=Session([])
    envelope,errors=runtime.fetch_official(session,'fixture',123,deadline=0)
    assert envelope is None and session.calls==0
    assert errors[0]['error']=='COLLECTION_TIME_BUDGET_EXHAUSTED'


def test_case_immutability_is_not_disabled(tmp_path):
    path=tmp_path/'proof.json';m.write(path,{'goals':0});before=path.read_bytes()
    with pytest.raises(ValueError,match='Immutable collision'):m.write(path,{'goals':1})
    assert path.read_bytes()==before


def test_latest_failed_batch_is_retried_without_new_capture():
    src={'files':{'captures/2026-10-26/predictions.csv':{'sha256':'one'}}}
    prior=copy.deepcopy(src);prior['workflow_run_id']='42'
    prior['files']['batches/42-1/request.json']={'sha256':'request'}
    assert runtime.new_capture_available(src,prior,'EXPLORATORY',{})
    prior['files']['batches/42-1/complete.json']={'sha256':'complete'}
    assert not runtime.new_capture_available(src,prior,'EXPLORATORY',{})
    prior['files']['batches/old-failure/request.json']={'sha256':'old'}
    assert not runtime.new_capture_available(src,prior,'EXPLORATORY',{})
