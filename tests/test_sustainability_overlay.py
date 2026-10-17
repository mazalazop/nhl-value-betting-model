import pandas as pd
from scripts.audit_player_sustainability import build, judge

def test_hot_streak_diagnostic_is_not_a_prediction_adjustment():
    a=dict(games=50,ppg=.6,shots_pg=2.,toi=20.,pp=3.,hit=.5)
    c=dict(games=8,ppg=1.1,shots_pg=3.,toi=22.,pp=4.,hit=.5)
    recent=dict(games=5,ppg=1.8,shots_pg=3.,toi=22.,pp=4.,hit=.8)
    recent3=dict(games=3,ppg=2.3,shots_pg=3.,toi=22.,pp=4.,hit=1.)
    label,_=judge(a,c,recent,recent3,3,0)
    assert label == 'SURCHAUFFE FORTE'

def test_rebound_requires_volume_and_toi():
    a=dict(games=70,ppg=1.,shots_pg=2.,toi=19.,pp=3.,hit=.6)
    c=dict(games=8,ppg=.5,shots_pg=3.,toi=20.,pp=4.,hit=.3)
    recent=dict(games=5,ppg=.4,shots_pg=3.2,toi=20.,pp=4.,hit=.2)
    label,_=judge(a,c,recent,recent,0,3)
    assert label == 'REBOND A EXAMINER'
    recent['shots_pg']=1.
    assert judge(a,c,recent,recent,0,3)[0] != 'REBOND A EXAMINER'

def test_future_and_non_regular_games_are_not_used():
    before=[dict(id_joueur=1,id_match=2025020001+i,date_match='2025-10-01',
                 season_source=20252026,points=1,tirs=2,temps_de_glace='20:00')
            for i in range(40)]
    before.extend(dict(id_joueur=1,id_match=2026020001+i,date_match='2026-10-16',
                 season_source=20262027,points=0,tirs=4,temps_de_glace='21:00') for i in range(5))
    before.append(dict(id_joueur=1,id_match=2026020198,date_match='2026-10-17',
                 season_source=20262027,points=7,tirs=15,temps_de_glace='30:00'))
    before.append(dict(id_joueur=1,id_match=2026030198,date_match='2026-10-15',
                 season_source=20262027,points=7,tirs=15,temps_de_glace='30:00'))
    pred=pd.DataFrame([dict(date_match='2026-10-17',id_joueur=1,nom='Test Player',
        team_player_match='ABC',adversaire_match='DEF',proba_point_1p_calibree=.6,
        rank_proba_sur_date=3,current_point_streak_pre=0,current_no_point_streak_pre=5)])
    out=build(pred,pd.DataFrame(before),'2026-10-17').iloc[0]
    assert out.ppg_saison == 0
    assert out.p_henachel == .6
    assert out.diagnostic == 'REBOND A EXAMINER'
