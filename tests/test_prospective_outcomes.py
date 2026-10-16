import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from archive_prospective_outcomes import record
from henachel.prospective_pp import Archive


def test_outcomes_rerun_corrections_and_missing_player(tmp_path):
    a=Archive(tmp_path/'a.sqlite3')
    rows=[dict(game_id=1,player_id=i,game_date='2026-10-13',event_start_utc='2026-10-14T00:00:00Z',team='TOR',player_name=f'Test {i}',collected_at_utc='2026-10-13T12:00:00Z',pp_unit='UNKNOWN') for i in [1,2]]
    a.append('pre','pregame',rows,now='2026-10-13T12:00:00Z')
    stats=pd.DataFrame([dict(id_match=1,id_joueur=1,points=1,buts=0,passes=1)])
    matches=pd.DataFrame([dict(id_match=1,status='LIVE')])
    assert record(a,stats,matches,now='2026-10-14T05:00:00Z')['changes']==0
    matches['status']='OFF'
    first=record(a,stats,matches,now='2026-10-14T05:00:00Z')
    assert first['changes']==1 and first['unresolved'][0]['reason']=='missing_player_not_zero'
    assert record(a,stats,matches,now='2026-10-14T06:00:00Z')['changes']==0
    stats.loc[0,['buts','passes']]=[1,0]
    assert record(a,stats,matches,now='2026-10-15T05:00:00Z')['changes']==1
    stats.loc[0,['buts','passes']]=[0,1]
    assert record(a,stats,matches,now='2026-10-16T05:00:00Z')['changes']==1
    assert a.db.execute('SELECT count(*) FROM batches').fetchone()[0]==4
    a.close()
