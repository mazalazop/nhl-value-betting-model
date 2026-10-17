import pandas as pd
import pytest
from scripts.audit_goal_sustainability import build, diagnose

def historical():
    rows=[]
    for i in range(40):
        rows.append(dict(id_joueur=1,id_match=2025020001+i,date_match="2025-10-01",
                         season_source=20252026,buts=1 if i%3==0 else 0,
                         tirs=3,temps_de_glace="20:00"))
    for i in range(5):
        rows.append(dict(id_joueur=1,id_match=2026020001+i,
                         date_match=f"2026-10-{10+i:02d}",season_source=20262027,
                         buts=0,tirs=5,temps_de_glace="21:00"))
    rows.extend([
        dict(id_joueur=1,id_match=2026020099,date_match="2026-10-17",season_source=20262027,
             buts=12,tirs=20,temps_de_glace="40:00"),
        dict(id_joueur=1,id_match=2026030098,date_match="2026-10-14",season_source=20262027,
             buts=12,tirs=20,temps_de_glace="40:00"),
        dict(id_joueur=1,id_match=2026010097,date_match="2026-10-13",season_source=20262027,
             buts=12,tirs=20,temps_de_glace="40:00"),
    ])
    return pd.DataFrame(rows)

def future():
    return pd.DataFrame([dict(date_match="2026-10-17",id_match=2026020122,
        id_joueur=1,nom="Test Player",team_player_match="AAA",adversaire_match="BBB",
        probability_goal=.357654,proba_goal_1p_calibree=.357654,
        goal_probability_rank=11,model_status="research_only_operational")])

def test_goal_sustainability_never_reads_future_or_playoffs():
    result=build(future(),historical(),"2026-10-17")
    row=result.iloc[0]
    assert row.matchs_saison==5 and row.buts_pg_saison==0
    assert row.diagnostic=="REBOND A EXAMINER" and row.serie_sans_but==5
    assert row.p_henachel_goal==.357654 and row.rang_henachel_goal==11
    assert row.impact_sur_proba=="AUCUN" and row.pp1_statut=="NON_CONFIRME"

def test_rebound_needs_volume():
    h=historical()
    h.loc[h.season_source==20262027,"tirs"]=1
    assert build(future(),h,"2026-10-17").iloc[0].diagnostic!="REBOND A EXAMINER"

def test_surchauffe_conversion_and_hot_streak():
    a=dict(games=70,gpg=.3,shots_pg=3.,toi=20.,pp=3.,conversion=.10)
    b=dict(games=8,gpg=.8,shots_pg=3.,toi=20.,pp=3.,conversion=.26)
    recent=dict(games=5,gpg=1.2,shots_pg=3.2,toi=20.,pp=3.,conversion=.375)
    assert diagnose(a,b,recent,recent,3,0)[0]=="SURCHAUFFE CONVERSION"
    recent["conversion"]=.13
    assert diagnose(a,b,recent,recent,3,0)[0]=="SERIE DE BUTS A EXAMINER"

def test_only_target_slate_and_research_status():
    g=future();g.loc[0,"date_match"]="2026-10-16"
    with pytest.raises(ValueError,match="target date"):
        build(g,historical(),"2026-10-17")
    g=future();g.loc[0,"model_status"]="production"
    with pytest.raises(ValueError,match="research_only_operational"):
        build(g,historical(),"2026-10-17")

def test_pp_minutes_measured_do_not_imply_pp1():
    pp=pd.DataFrame([dict(id_joueur=1,id_match=2026020001,ppTimeOnIce=120)])
    row=build(future(),historical(),"2026-10-17",pp).iloc[0]
    assert row.pp1_statut=="NON_CONFIRME" and row.pp_5==2
    with pytest.raises(ValueError,match="Duplicate PP"):
        build(future(),historical(),"2026-10-17",pd.concat([pp,pp]))

def test_no_games_does_not_break_pipeline():
    out=build(future().iloc[:0],historical(),"2026-10-17")
    assert out.empty and "diagnostic" in out.columns
