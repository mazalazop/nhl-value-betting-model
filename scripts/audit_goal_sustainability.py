#!/usr/bin/env python3
"""Research-only pregame GOAL structural audit. Never changes model outputs, odds or picks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STATUS = "research_only_operational"
GOAL_COLS = ["date_match", "id_match", "id_joueur", "nom", "team_player_match",
             "adversaire_match", "probability_goal", "goal_probability_rank", "model_status"]
HIST_COLS = ["id_joueur", "id_match", "date_match", "season_source", "buts", "tirs", "temps_de_glace"]
PP_COLS = ["id_joueur", "id_match", "ppTimeOnIce"]

def minutes(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)): return float(value)
    try:
        parts = [float(x) for x in str(value).split(":")]
        if len(parts) == 2: return parts[0] + parts[1]/60
        if len(parts) == 3: return parts[0]*60 + parts[1] + parts[2]/60
        if len(parts) == 1: return parts[0]
    except (ValueError, TypeError, OverflowError): pass
    return np.nan

def metrics(rows):
    keys = ("gpg", "shots_pg", "toi", "pp", "pp_share", "conversion", "goals_60", "shots_60")
    if rows.empty: return {k:np.nan for k in keys} | {"games":0}
    n=len(rows)
    if rows[["buts","tirs","toi_minutes"]].isna().any().any():
        return {k:np.nan for k in keys} | {"games":n}
    goals=rows.buts.sum(); shots=rows.tirs.sum(); toi=rows.toi_minutes.sum()
    pp=rows.pp_minutes.sum(min_count=1)
    def divide(a,b): return float(a/b) if pd.notna(a) and pd.notna(b) and b>0 else np.nan
    return dict(games=n, gpg=divide(goals,n), shots_pg=divide(shots,n),
                toi=divide(toi,n), pp=float(rows.pp_minutes.mean()) if rows.pp_minutes.notna().any() else np.nan,
                pp_share=divide(pp,toi), conversion=divide(goals,shots),
                goals_60=divide(goals*60,toi), shots_60=divide(shots*60,toi))

def series(goals):
    hot=dry=0
    for count in goals:
        if pd.isna(count): return (0,0)
        if count>0: hot+=1;dry=0
        else: dry+=1;hot=0
    return hot,dry

def diagnose(prior,season,recent,last3,hot,dry):
    if prior["games"]<30 or season["games"]<4 or not np.isfinite(prior["gpg"]) or prior["gpg"]<=0:
        return "DONNEES INSUFFISANTES", "Historique buteur insuffisant"
    if not (np.isfinite(prior["shots_pg"]) and prior["shots_pg"]>0 and
            np.isfinite(recent["shots_pg"]) and np.isfinite(prior["toi"]) and np.isfinite(recent["toi"])):
        return "DONNEES INSUFFISANTES", "Tirs ou temps de glace incomplets"
    pp_good=(np.isfinite(prior["pp"]) and np.isfinite(recent["pp"]) and
             recent["pp"]>=.8*prior["pp"])
    pp_missing=not np.isfinite(prior["pp"]) or not np.isfinite(recent["pp"])
    if (dry>=3 and season["gpg"]<=.8*prior["gpg"] and
            recent["shots_pg"]>=1.15*prior["shots_pg"] and recent["toi"]>=.9*prior["toi"] and
            (pp_good or pp_missing)):
        note="PP non mesure" if pp_missing else "PP non affaibli"
        return "REBOND A EXAMINER", "Disette, tirs/TOI maintenus ; "+note
    if (np.isfinite(prior["conversion"]) and prior["conversion"]>0 and
            np.isfinite(recent["conversion"]) and
            recent["conversion"]>=1.75*prior["conversion"] and
            recent["gpg"]>=1.6*prior["gpg"] and recent["shots_pg"]<1.35*prior["shots_pg"]):
        return "SURCHAUFFE CONVERSION", "Conversion en hausse sans augmentation proportionnelle des tirs"
    if hot>=3 and recent["gpg"]>=1.3*prior["gpg"]:
        return "SERIE DE BUTS A EXAMINER", "Serie chaude ; verifier volume, role et PP"
    if recent["shots_pg"]>=1.3*prior["shots_pg"] and recent["toi"]>=1.05*prior["toi"] and recent["gpg"]<=1.1*prior["gpg"]:
        return "VOLUME EN HAUSSE", "Tirs et temps de glace augmentent davantage que les buts"
    return "PAS DE SIGNAL FORT", "Verifier gardien, PP1 et opposition"

def build(goal,stats,target_date,pp=None):
    target=pd.Timestamp(target_date)
    if target.tzinfo is not None or target!=target.normalize(): raise ValueError("Expected NHL date only")
    missing=set(GOAL_COLS)-set(goal.columns)
    if missing: raise ValueError("Missing GOAL columns: "+str(sorted(missing)))
    missing=set(HIST_COLS)-set(stats.columns)
    if missing: raise ValueError("Missing history columns: "+str(sorted(missing)))
    g=goal.copy()
    if not pd.to_datetime(g.date_match,errors="raise").eq(target).all():
        raise ValueError("GOAL output not entirely from target date")
    if not g.model_status.eq(STATUS).all(): raise ValueError("Expected research_only_operational GOAL")
    g["id_joueur"]=pd.to_numeric(g.id_joueur,errors="raise").astype("int64")
    g["id_match"]=pd.to_numeric(g.id_match,errors="raise").astype("int64")
    if g.duplicated(["id_joueur","id_match"]).any(): raise ValueError("Duplicate GOAL predictions")
    probabilities=pd.to_numeric(g.probability_goal,errors="raise")
    if not probabilities.between(0,1).all(): raise ValueError("Invalid GOAL probability")
    if "proba_goal_1p_calibree" in g and not np.allclose(
            pd.to_numeric(g.proba_goal_1p_calibree,errors="raise"), probabilities, atol=1e-12,rtol=0):
        raise ValueError("GOAL probability aliases disagree")
    h=stats.copy()
    h["id_match"]=pd.to_numeric(h.id_match,errors="coerce").astype("Int64")
    h["id_joueur"]=pd.to_numeric(h.id_joueur,errors="coerce").astype("Int64")
    historical_date=pd.to_datetime(h.date_match,errors="coerce")
    regular=h.id_match.astype("string").str.fullmatch(r"\d{4}02\d{4}").fillna(False)
    h=h.loc[regular & historical_date.lt(target)].copy()
    if h.duplicated(["id_joueur","id_match"]).any():raise ValueError("Duplicate history player/game identity")
    for c in ["buts","tirs"]: h[c]=pd.to_numeric(h[c],errors="coerce")
    if (h["buts"].dropna()<0).any() or (h["tirs"].dropna()<0).any(): raise ValueError("Negative goals/shots")
    h["toi_minutes"]=h.temps_de_glace.map(minutes)
    h["season_source"]=pd.to_numeric(h.season_source,errors="coerce").astype("Int64")
    h["pp_minutes"]=np.nan
    if pp is not None:
        missing=set(PP_COLS)-set(pp.columns)
        if missing:raise ValueError("Missing PP columns: "+str(sorted(missing)))
        p=pp[PP_COLS].copy()
        for c in ["id_joueur","id_match"]:p[c]=pd.to_numeric(p[c],errors="coerce").astype("Int64")
        if p.duplicated(["id_joueur","id_match"]).any():raise ValueError("Duplicate PP player/game identity")
        p["pp_minutes"]=pd.to_numeric(p.ppTimeOnIce,errors="coerce")/60
        if (p.pp_minutes.dropna()<0).any():raise ValueError("Negative PP time")
        h=h.drop(columns=["pp_minutes"]).merge(p[["id_joueur","id_match","pp_minutes"]],
                on=["id_joueur","id_match"],how="left",validate="one_to_one")
    h=h.sort_values(["date_match","id_match"])
    groups={int(k):v for k,v in h.groupby("id_joueur") if pd.notna(k)}
    y=target.year if target.month>=7 else target.year-1
    previous=(y-1)*10000+y; current=y*10000+y+1
    result=[]
    for row in g.itertuples(index=False):
        history=groups.get(int(row.id_joueur),h.iloc[0:0])
        prior=metrics(history.loc[history.season_source.eq(previous).fillna(False)])
        season_rows=history.loc[history.season_source.eq(current).fillna(False)]
        season=metrics(season_rows);recent=metrics(season_rows.tail(5));last3=metrics(season_rows.tail(3))
        hot,dry=series(season_rows.buts.to_list())
        diagnostic,why=diagnose(prior,season,recent,last3,hot,dry)
        result.append(dict(date_match=target_date,id_match=int(row.id_match),id_joueur=int(row.id_joueur),
            joueur=row.nom,equipe=row.team_player_match,adversaire=row.adversaire_match,
            p_henachel_goal=float(row.probability_goal),rang_henachel_goal=int(row.goal_probability_rank),
            matchs_reference=prior["games"],matchs_saison=season["games"],
            buts_pg_reference=prior["gpg"],buts_pg_saison=season["gpg"],
            buts_pg_5=recent["gpg"],buts_pg_3=last3["gpg"],
            tirs_pg_reference=prior["shots_pg"],tirs_pg_5=recent["shots_pg"],
            buts_60_reference=prior["goals_60"],buts_60_5=recent["goals_60"],
            tirs_60_reference=prior["shots_60"],tirs_60_5=recent["shots_60"],
            conversion_reference=prior["conversion"],conversion_5=recent["conversion"],
            toi_reference=prior["toi"],toi_5=recent["toi"],pp_reference=prior["pp"],pp_5=recent["pp"],
            part_pp_reference=prior["pp_share"],part_pp_5=recent["pp_share"],
            serie_buts=hot,serie_sans_but=dry,diagnostic=diagnostic,justification=why,
            tirs_dangereux_statut="NON_DISPONIBLE",pp1_statut="NON_CONFIRME",
            statut_modele=STATUS,impact_sur_proba="AUCUN",type_analyse="context_only"))
    return (pd.DataFrame(result).sort_values("rang_henachel_goal") if result else
            pd.DataFrame(columns=["date_match","id_joueur","p_henachel_goal","rang_henachel_goal",
                                  "diagnostic","statut_modele"]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date",required=True)
    parser.add_argument("--predictions",type=Path,default=Path("outputs/predictions_upcoming_goal_research.csv"))
    parser.add_argument("--stats",type=Path,default=Path("data/raw/stats.csv"))
    parser.add_argument("--pp-stats",type=Path,default=Path("data/raw/pp_stats_game.csv"))
    parser.add_argument("--output",type=Path,default=Path("outputs/player_sustainability_goal_research.csv"))
    a=parser.parse_args()
    for f in [a.predictions,a.stats]:
        if not f.is_file(): raise FileNotFoundError(f)
    goal=pd.read_csv(a.predictions,low_memory=False)
    history=pd.read_csv(a.stats,usecols=HIST_COLS,low_memory=False)
    pp=pd.read_csv(a.pp_stats,usecols=PP_COLS,low_memory=False) if a.pp_stats.is_file() else None
    result=build(goal,history,a.target_date,pp)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    print("GOAL research sustainability diagnostic:",len(result),"players")
    if len(result):print(result.diagnostic.value_counts().to_string())

if __name__=="__main__":main()
