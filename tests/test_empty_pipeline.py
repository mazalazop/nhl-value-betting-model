import json
import pandas as pd
from conftest import load_script


def test_empty_day_from_predictions_through_publication_views(tmp_path,monkeypatch):
    future=load_script('05_predict_upcoming_games');matcher=load_script('06_match_model_to_unibet_odds');picks=load_script('07_build_daily_bets');publisher=load_script('08_publish_to_google_sheet')
    for attr,name in [('PRED_UPCOMING_RAW_PATH','raw.csv'),('PRED_UPCOMING_CAL_PATH','cal.csv'),('SUMMARY_PATH','summary.json')]:monkeypatch.setattr(future,attr,tmp_path/name)
    future.write_empty_predictions('no_games',pd.Timestamp('2026-10-02'))
    model=matcher.load_model_predictions(tmp_path/'cal.csv')
    path=tmp_path/'odds.json';path.write_text(json.dumps(dict(bookmaker='unibet',market='player_points',stat='points',threshold=1,outcome_label='1+',rows=[])))
    _,odds=matcher.load_odds_json(path)
    matched,_,_,_=matcher.match_rows(model,odds,'2026-10-02')
    matched.to_csv(tmp_path/'matched.csv',index=False)
    loaded=picks.load_candidates(tmp_path/'matched.csv')
    selected,_=picks.build_daily_bets(loaded,'2026-10-02',10,1.4,.9,.02,True,True)
    picks.append_history(tmp_path/'history.csv',selected)
    assert publisher.build_daily_display_df(selected).empty
    assert publisher.build_history_display_df(pd.read_csv(tmp_path/'history.csv')).empty
    assert json.loads((tmp_path/'summary.json').read_text())['status']=='no_games'


def test_manifest_hashes_only_explicit_inputs(tmp_path):
    from henachel.manifest import manifest
    import pytest
    path=tmp_path/'data.csv';path.write_text('x\n1\n')
    before=manifest([path]);path.write_text('x\n2\n');after=manifest([path])
    assert before['sources'][0]['sha256']!=after['sources'][0]['sha256']
    assert before['dependencies']['pandas']=='2.2.3'
    with pytest.raises(ValueError):manifest([tmp_path/'credentials.json'])
