import copy
import json
import hashlib
from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel import but_v2_a01_registry as m
from test_but_v2_participation_a01 import sources, ROW, ASOF, player


def original(tmp_path):
    root=tmp_path/'original';cap=root/'captures/2026-10-22';cap.mkdir(parents=True)
    rows=pd.DataFrame([dict(ROW,id_joueur=100),ROW])
    rows['date_match']='2026-10-22';rows['generated_at_utc']='2026-10-22T11:00:00Z'
    rows['archive_group']='CANONICAL';rows['scope']='BUT_V2_SHADOW'
    rows['probability']=[.2,.3];rows['v1_probability']=[.25,.35]
    rows['rank']=[2,1];rows['v1_rank']=[2,1];rows['odds_decimal']=[3.,None]
    rows.to_csv(cap/'predictions.csv',index=False)
    (cap/'manifest.json').write_text(json.dumps(dict(predictions_sha256=m.sha(cap/'predictions.csv'))))
    batch=root/'batches/original/cumulative';batch.mkdir(parents=True)
    rows['settlement_status']=['LOSS','UNRESOLVED'];rows.to_csv(batch/'settlement.csv',index=False)
    (root/'latest_batch.json').write_text(json.dumps(dict(path='batches/original')))
    m.seal(root)
    return root


def test_cycle_resume_reviews_and_restore(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('a01_backup',Path(__file__).resolve().parents[1]/'scripts/backup_but_v2_state.py')
    backup=importlib.util.module_from_spec(spec);spec.loader.exec_module(backup)
    src=original(tmp_path);state=tmp_path/'a01';before=m.sha(src/'archive_manifest.json');e={'123':sources()}
    result=m.cycle(src,state,e,ASOF,'test1')
    assert result['coverage']['original_coverage']==.5
    assert result['coverage']['information_coverage']==1
    assert result['comparison']['top10']['v2']['selected']==2
    assert result['comparison']['top10']['v2']['settled']==1
    count=len(list(state.rglob('*')))
    assert m.cycle(src,state,e,ASOF,'test1')==result and len(list(state.rglob('*')))==count
    after=copy.deepcopy(e);player(after['123'])['goals']=1
    corrected=m.cycle(src,state,after,ASOF,'test2')
    assert corrected['coverage']['counts']['PARTICIPATION_UNKNOWN']==1
    # A later identical poll cannot silently accept the unreviewed correction.
    repeat=m.cycle(src,state,after,ASOF,'test3')
    assert repeat['coverage']['counts']['PARTICIPATION_UNKNOWN']==1
    classes=json.loads((state/'batches/test3/classifications.json').read_text())
    case=next(c['case_id'] for c in classes if 'case_id' in c)
    decision=dict(case_id=case,choice='ACCEPT_CURRENT',reviewer='fixture-reviewer',rationale='Official correction fixture',reviewed_at_utc=ASOF,review_reference='fixture://review')
    accepted=m.cycle(src,state,after,ASOF,'test4',decisions=[decision])
    assert accepted['comparison']['v2']['hit_rate']==1
    assert (state/'batches/test1/report.json').exists() and (state/'batches/test2/report.json').exists()
    assert m.sha(src/'archive_manifest.json')==before
    backup.validate_state(state)
    # Exercise the actual durable serialization and restore machinery with an in-memory transport.
    from test_but_v2_backup import MemoryStore
    store=MemoryStore()
    receipt=backup.backup(store,state,'fixture','fixture-run')
    restored=tmp_path/'restored'
    backup.restore(store,store.latest(),restored)
    assert backup.inventory(state)==backup.inventory(restored)
    assert receipt['remote_restore_verified']


def test_start_guards_and_history_exclusion(tmp_path):
    release=dict(files={})
    start=dict(enabled=False,mode='PROSPECTIVE')
    with pytest.raises(ValueError,match='disabled'):m.validate_start(start,release,ASOF)
    start=dict(enabled=True,mode='PROSPECTIVE',study_id='fixture',registered_at_utc='2026-10-23T01:00:00Z',t0_utc='2026-10-23T02:00:00Z',
               first_nhl_date='2026-10-24',excluded_through_nhl_date='2026-10-23',approved_by='fixture',approval_reference='fixture',release_sha256=m.part.digest(release))
    m.validate_start(start,release,ASOF)
    src=original(tmp_path)
    result=m.cycle(src,tmp_path/'a01',{'123':sources()},ASOF,'test',mode='PROSPECTIVE',start=start,release=release)
    assert result['status']=='NO_ADMISSIBLE_CAPTURE'
    with pytest.raises(ValueError,match='preregistered'):m.validate_start(dict(start,registered_at_utc=start['t0_utc']),release,ASOF)
    rows=pd.DataFrame([dict(date_match='2026-10-24',generated_at_utc='2026-10-22T12:00:00Z')])
    assert not m.eligible(rows,start)


def test_mode_switch_tamper_and_duplicate_rejected(tmp_path):
    src=original(tmp_path);state=tmp_path/'a01';m.cycle(src,state,{'123':sources()},ASOF,'one')
    with pytest.raises(ValueError,match='Separate'):m.cycle(src,src/'child',{},ASOF,'bad')
    with pytest.raises(ValueError,match='reused'):m.cycle(src,state,{},ASOF,'one')
    (state/'captures/2026-10-22/predictions.csv').write_text('tampered')
    with pytest.raises(ValueError,match='integrity'):m.cycle(src,state,{},ASOF,'two')


def test_workflow_gate_and_preregistration_contract():
    root=Path(__file__).resolve().parents[1]
    start=json.loads((root/'configs/but_v2_a01_start.json').read_text())
    if start['enabled']:
        release=json.loads((root/'configs/but_v2_a01_release.json').read_text())
        m.validate_start(start,release,start['t0_utc'])
        assert start['approval_reference'] and start['approved_by']
        assert start['first_nhl_date']>start['excluded_through_nhl_date']
    else:
        assert start['t0_utc'] is None
    workflow=(root/'.github/workflows/but-v2-a01.yml').read_text()
    assert "vars.HENACHEL_BUT_V2_A01_ENABLED == 'true'" in workflow
    assert 'cancel-in-progress: false' in workflow
    assert 'group: but-v2-a01-registry' in workflow


def test_keep_previous_and_invalid_decision(tmp_path):
    src=original(tmp_path);state=tmp_path/'a01';e={'123':sources()}
    m.cycle(src,state,e,ASOF,'one')
    changed=copy.deepcopy(e);player(changed['123'])['goals']=1
    m.cycle(src,state,changed,ASOF,'two')
    classes=json.loads((state/'batches/two/classifications.json').read_text())
    case=next(c['case_id'] for c in classes if 'case_id' in c)
    d=dict(case_id=case,choice='KEEP_PREVIOUS',reviewer='fixture',rationale='Contradictory revision rejected',reviewed_at_utc=ASOF,review_reference='fixture://review')
    kept=m.cycle(src,state,changed,ASOF,'three',decisions=[d])
    assert kept['comparison']['v2']['hit_rate']==0
    polled=m.cycle(src,state,changed,ASOF,'four')
    assert polled['comparison']['v2']['hit_rate']==0
    with pytest.raises(ValueError,match='overwritten'):
        m.cycle(src,state,changed,ASOF,'five',decisions=[dict(d,choice='ACCEPT_CURRENT')])


def test_confirmatory_window_not_claimed_early():
    assert m.final_report(pd.DataFrame(),{}, {},ASOF,'2026-10-23')['status']=='WINDOW_NOT_COMPLETE'


def test_release_hashes_and_frozen_files_match():
    root=Path(__file__).resolve().parents[1]
    release=json.loads((root/'configs/but_v2_a01_release.json').read_text())
    original=json.loads((root/'docs/but_v2/PRE_T0_REGISTRATION.json').read_text())['file_sha256']
    for name,digest in release['files'].items():assert m.sha(root/name)==digest,name
    for name,digest in original.items():assert release['files'][name]==digest
    script=(root/'scripts/run_but_v2_a01.py').read_text()
    assert "Registration commit is not before T0" in script
    assert 'never reset' in script


def test_first_confirmatory_assessment_is_immutable(tmp_path):
    src=original(tmp_path);state=tmp_path/'a01'
    release=dict(files={});start=dict(enabled=True,mode='PROSPECTIVE',study_id='fixture',
        registered_at_utc='2026-10-21T01:00:00Z',t0_utc='2026-10-22T01:00:00Z',first_nhl_date='2026-10-22',
        excluded_through_nhl_date='2026-10-21',approved_by='fixture',approval_reference='fixture',release_sha256=m.part.digest(release))
    m.cycle(src,state,{'123':sources()},'2027-05-01T00:00:00Z','closed1','PROSPECTIVE',start,release)
    fixed=(state/'confirmatory/assessment.json').read_bytes()
    changed={'123':sources()};player(changed['123'])['goals']=1
    result=m.cycle(src,state,changed,'2027-05-02T00:00:00Z','closed2','PROSPECTIVE',start,release)
    assert (state/'confirmatory/assessment.json').read_bytes()==fixed
    assert not result['confirmatory']['promotion_authorized']
    assert 'post_closure_revision_note' in result


def test_incomplete_attempt_is_retained_not_silently_reused(tmp_path):
    src=original(tmp_path);state=tmp_path/'a01'
    m.cycle(src,state,{'123':sources()},ASOF,'one')
    (state/'batches/one/complete.json').unlink();m.seal(state)
    with pytest.raises(ValueError,match='Incomplete'):
        m.cycle(src,state,{'123':sources()},ASOF,'one')
    result=m.cycle(src,state,{'123':sources()},ASOF,'recovery')
    assert result['comparison']['n']==2
    assert (state/'batches/one/report.json').exists()


def test_review_applies_to_same_documents_recollected_after_review(tmp_path):
    src=original(tmp_path);state=tmp_path/'a01';e={'123':sources()}
    m.cycle(src,state,e,ASOF,'one')
    changed=copy.deepcopy(e);player(changed['123'])['goals']=1
    m.cycle(src,state,changed,ASOF,'two')
    classes=json.loads((state/'batches/two/classifications.json').read_text())
    case=next(c['case_id'] for c in classes if 'case_id' in c)
    for envelope in changed['123'].values():envelope['observed_at_utc']='2026-10-23T08:30:00Z'
    d=dict(case_id=case,choice='ACCEPT_CURRENT',reviewer='fixture',rationale='Reviewed 08:00 version',
           reviewed_at_utc='2026-10-23T08:15:00Z',review_reference='fixture://review')
    result=m.cycle(src,state,changed,ASOF,'three',decisions=[d])
    assert result['comparison']['v2']['hit_rate']==1


def test_official_cache_bounds_requests_and_refresh_is_explicit(tmp_path):
    import importlib.util
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('a01_runtime_cache',scripts/'run_but_v2_a01.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    src=original(tmp_path);state=tmp_path/'a01';m.cycle(src,state,{'123':sources()},ASOF,'one')
    cache=state/'official-cache';cache.mkdir();(cache/'123.json').write_text(json.dumps(sources()));m.seal(state)
    class Session:
        calls=[]
        def get(self,url,**kw):
            self.calls.append(url)
            endpoint=url.rsplit('/',1)[-1]
            class Response:
                def raise_for_status(self):pass
                def json(self):return sources()[endpoint]['document']
            return Response()
    session=Session()
    data,errors=runtime.collect(src,state,session,'EXPLORATORY',{},pd.Timestamp('2026-11-02T00:00:00Z'))
    assert not session.calls and data['123']==sources() and not errors
    runtime.collect(src,state,session,'EXPLORATORY',{},pd.Timestamp('2026-11-02T00:00:00Z'),refresh=True)
    assert len(session.calls)==2
    session.calls=[]
    data,_=runtime.collect(src,state,session,'EXPLORATORY',{},pd.Timestamp('2026-10-22T00:00:00Z'))
    assert not session.calls and data['123']=={}
