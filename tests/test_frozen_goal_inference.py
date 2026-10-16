import sys
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from predict_frozen_goal import predict


def inputs():
    model=Mock(); model.predict_proba.return_value=np.array([[.8,.2]])
    cal=Mock(); cal.predict.return_value=np.array([.21])
    bundle=dict(features=['tirs_moy_5'], model=model, calibrator=cal, cutoff='2024-02-01')
    frame=pd.DataFrame([dict(id_match=1,id_joueur=2,nom='Player Test',team_player_match='TOR',date_match='2026-10-13',start_time_utc='2026-10-14T00:00:00Z',tirs_moy_5=2.)])
    return bundle,frame


def test_frozen_inference_only_no_fit_no_promotion():
    b,f=inputs(); out=predict(b,f,'frozenhash',now='2026-10-13T12:00:00Z')
    assert out.proba_goal_1p_calibree.iloc[0]==.21
    b['model'].fit.assert_not_called(); b['calibrator'].fit.assert_not_called()
    assert out.model_status.iloc[0]=='research_only_not_production_approved'
    assert out.model_cutoff.iloc[0]=='2024-02-01'


@pytest.mark.parametrize('failure',['late','labels','cutoff','feature'])
def test_refuse_temporal_or_target_violation(failure):
    b,f=inputs(); now='2026-10-13T12:00:00Z'
    if failure=='late': now='2026-10-14T00:00:00Z'
    if failure=='labels': f['buts']=0
    if failure=='cutoff': b['cutoff']='2026-10-13'
    if failure=='feature': b['features']=['points']
    with pytest.raises(ValueError): predict(b,f,'hash',now=now)
    b['model'].predict_proba.assert_not_called()


def test_production_feature_export_to_frozen_goal_parity(tmp_path):
    import hashlib
    import json
    import joblib
    from fixtures import games_fixture
    from test_feature_parity import build, future
    from henachel.goal import augment_goal_features
    from predict_frozen_goal import prepare
    root=Path(__file__).parents[1]
    artifact=root/'model/research_artifacts/goal17_frozen_20240201.joblib'
    manifest=json.loads(artifact.with_suffix('.json').read_text())
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==manifest['sha256']
    bundle=joblib.load(artifact)
    source=games_fixture(); cut=sorted(source.date_match.unique())[36]
    history=build(source[source.date_match<cut])
    matches=source[source.date_match==cut].drop_duplicates('id_match').copy()
    matches['status']='FUT'; matches[['buts_domicile','buts_exterieur']]=np.nan
    pool=source[source.date_match==cut][['id_joueur','team_player_match','nom','position']]
    upcoming=future.build_upcoming_universe(matches,history,source.drop_duplicates('id_match'),pool,{})
    upcoming['start_time_utc']=(pd.Timestamp(cut).tz_localize('UTC')+pd.Timedelta(hours=20)).isoformat()
    path=tmp_path/'features.json';upcoming.to_json(path,orient='table',date_format='iso',index=False)
    prepared=prepare(pd.read_json(path,orient='table'),history,bundle['features'])
    actual=augment_goal_features(build(source))
    actual=actual[actual.date_match==cut].sort_values('id_joueur')
    np.testing.assert_allclose(prepared.sort_values('id_joueur')[bundle['features']],actual[bundle['features']],equal_nan=True)
    out=predict(bundle,prepared,manifest['sha256'],now=pd.Timestamp(cut).tz_localize('UTC'))
    assert len(out)==2 and out.proba_goal_1p_calibree.between(0,1).all()
    assert out.model_status.eq('research_only_not_production_approved').all()
