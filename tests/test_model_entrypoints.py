"""Exercise real entrypoints on synthetic data and temporary artifacts, without APIs."""
from pathlib import Path
import json
import sys
import pandas as pd
import pytest
from conftest import load_script
from fixtures import games_fixture
from test_feature_parity import build


def relocate_outputs(module,path,monkeypatch):
    old=module.OUTPUTS_DIR
    for key,value in list(vars(module).items()):
        if isinstance(value,Path) and (value==old or old in value.parents):
            monkeypatch.setattr(module,key,path/value.relative_to(old))


def test_train_calibrate_and_predict_entrypoints(tmp_path,monkeypatch):
    raw=games_fixture();features=build(raw);path=tmp_path/'features.csv';features.to_csv(path,index=False)
    train=load_script('02_train_point_model');relocate_outputs(train,tmp_path,monkeypatch);monkeypatch.setattr(train,'FEATURES_PATH',path)
    train.main()
    cal=load_script('03_calibrate_point_model');relocate_outputs(cal,tmp_path,monkeypatch);cal.main()
    assert (tmp_path/'02_point_enriched.joblib').exists();assert (tmp_path/'03_point_calibrator.joblib').exists()
    future=load_script('05_predict_upcoming_games');relocate_outputs(future,tmp_path,monkeypatch)
    date=pd.Timestamp(raw.date_match.max())+pd.Timedelta(days=2)
    matches=pd.DataFrame([dict(id_match=99999,date_match=date,saison=20242025,id_equipe_domicile='TOR',id_equipe_exterieur='NYR',status='FUT',buts_domicile=None,buts_exterieur=None,start_time_utc=date.strftime('%Y-%m-%dT23:00:00Z'))])
    match_path=tmp_path/'matches.csv';matches.to_csv(match_path,index=False)
    players=raw.groupby('id_joueur').tail(1)[['id_joueur','nom','position','team_player_match']].rename(columns={'team_player_match':'id_equipe'})
    player_path=tmp_path/'players.csv';players.to_csv(player_path,index=False)
    for name,value in [('FEATURES_HISTORY_PATH',path),('MATCHS_PATH',match_path),('JOUEURS_PATH',player_path),('RAW_DIR',tmp_path),('TEAM_STANDINGS_PATH',tmp_path/'absent_standings.csv')]:monkeypatch.setattr(future,name,value)
    monkeypatch.setattr(sys,'argv',['05','--target-date',date.date().isoformat()])
    with pytest.warns(UserWarning,match='Roster unavailable'):future.main()
    result=pd.read_csv(future.PRED_UPCOMING_CAL_PATH)
    assert len(result)==2;assert result.id_match.eq(99999).all();assert result.start_time_utc.notna().all()
    summary=json.loads(future.SUMMARY_PATH.read_text());assert summary['manifest']['commit']
    import joblib
    bundle=joblib.load(tmp_path/'05_point_bundle.joblib')
    assert bundle['calibrator'].method==result.calibration_method.iloc[0]
