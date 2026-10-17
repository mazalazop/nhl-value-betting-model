"""POINT-only correction of long-absence proxy lifetimes.

The shared henachel.features.py is intentionally UNCHANGED: it is SHA-locked
by the GOAL17 reproducibility protocol. This overlay is strictly pregame,
uses only earlier player games + the date of the upcoming match, and changes
only POINT return-related fields. Never interpret a schedule gap as a medically
confirmed injury.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def repair_point_returns(frame: pd.DataFrame, recent_games=3, stabilized_until_games=8,
                         min_toi=15.0, min_toi_ratio=0.75,
                         current_weight_default=0.60, current_weight_return=0.80):
    if not 1 <= recent_games < stabilized_until_games <= 20:
        raise ValueError("Invalid finite return windows")
    if frame.empty:
        return frame.copy()
    required=["id_joueur","id_match","date_match","season_source","absence_longue_flag",
              "toi_moy_10","pp_moy_5","temps_de_glace","temps_pp",
              "point_hit_rate_season_pre","point_hit_rate_prev_season",
              "points_per_game_season_pre","points_per_game_prev_season",
              "point_hit_rate_last_10","point_hit_rate_last_20"]
    missing=set(required)-set(frame)
    if missing: raise ValueError("Missing POINT return fields: "+str(sorted(missing)))
    x=frame.copy()
    x["_repair_original_order"]=np.arange(len(x))
    if x.duplicated(["id_joueur","id_match"]).any():
        raise ValueError("Duplicate player/match in return overlay")
    x["date_match"]=pd.to_datetime(x.date_match,errors="raise")
    if x["date_match"].isna().any():raise ValueError("Missing game dates")
    x=x.sort_values(["id_joueur","date_match","id_match"])
    keys=["id_joueur","season_source"]
    x["retour_episode"]=x.groupby(keys,sort=False)["absence_longue_flag"].cumsum()
    episode_keys=keys+["retour_episode"]
    after=x.groupby(episode_keys,sort=False).cumcount()
    x["matchs_depuis_retour_avant_match"]=after
    active=x.retour_episode.gt(0)
    x["toi_pre_absence_ref"]=np.where(x.absence_longue_flag.eq(1),x.toi_moy_10,np.nan)
    x["pp_pre_absence_ref"]=np.where(x.absence_longue_flag.eq(1),x.pp_moy_5,np.nan)
    for name in ["toi_pre_absence_ref","pp_pre_absence_ref"]:
        x[name]=x.groupby(episode_keys,sort=False)[name].ffill()
    for name,raw in [("toi_moy_retour_3_avant_match","temps_de_glace"),
                     ("pp_moy_retour_3_avant_match","temps_pp")]:
        x[name]=x.groupby(episode_keys,sort=False)[raw].transform(
            lambda s: s.shift(1).rolling(3,min_periods=1).mean())
    for out,num,den in [
            ("ratio_toi_retour_vs_pre_absence","toi_moy_retour_3_avant_match","toi_pre_absence_ref"),
            ("ratio_pp_retour_vs_pre_absence","pp_moy_retour_3_avant_match","pp_pre_absence_ref")]:
        x[out]=x[num]/x[den].where(x[den]>0)
    x["return_from_absence_flag"]=(active & after.lt(recent_games)).astype(int)
    x["return_stabilized_flag"]=(
        active & after.ge(recent_games) & after.le(stabilized_until_games)
        & x.toi_moy_retour_3_avant_match.ge(min_toi)
        & x.ratio_toi_retour_vs_pre_absence.ge(min_toi_ratio)
    ).astype(int)
    x["historical_current_weight"]=np.where(
        x.return_stabilized_flag.eq(1),current_weight_return,current_weight_default)
    x["historical_prev_weight"]=1.-x.historical_current_weight
    for out,cur,prev in [
       ("point_hit_rate_weighted_pre","point_hit_rate_season_pre","point_hit_rate_prev_season"),
       ("points_per_game_weighted_pre","points_per_game_season_pre","points_per_game_prev_season")]:
        x[out]=np.where(x[cur].notna() & x[prev].notna(),
                        x.historical_current_weight*x[cur]+x.historical_prev_weight*x[prev],
                        x[cur].fillna(x[prev]))
    combo=(.6*x.point_hit_rate_last_10.fillna(x.point_hit_rate_last_20)
           +.4*x.point_hit_rate_last_20.fillna(x.point_hit_rate_last_10))
    x["recent_vs_expected_gap"]=x.point_hit_rate_weighted_pre.fillna(.45)-combo.fillna(.45)
    return x.sort_values("_repair_original_order").drop(
        columns="_repair_original_order").reset_index(drop=True)
