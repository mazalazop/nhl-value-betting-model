#!/usr/bin/env python3
"""Audit structurel pré-match POINT : ne modifie jamais les probabilités HENACHEL.

Les seuils de diagnostic sont exploratoires, non validés pour reclasser automatiquement.
Exclut matchs de préparation, playoffs, et toute donnée du match à venir.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['id_joueur','id_match','date_match','season_source','points','tirs','temps_de_glace']

def minutes(v):
    try:
        mm, ss = str(v).split(':')
        return float(mm) + float(ss)/60
    except (ValueError, AttributeError):
        return np.nan

def summarize(rows):
    if rows.empty:
        return dict(games=0,ppg=np.nan,shots_pg=np.nan,toi=np.nan,pp=np.nan,hit=np.nan)
    return dict(games=len(rows),ppg=rows.points.mean(),shots_pg=rows.tirs.mean(),
                toi=rows.toi_minutes.mean(),pp=rows.pp_minutes.mean(),hit=(rows.points>0).mean())

def judge(previous,current,recent,last3,streak,blank):
    # Une serie ne predit pas une fin de serie : ce sont des alertes a controler.
    if previous['games'] < 30 or current['games'] < 4 or previous['ppg'] <= 0:
        return 'DONNEES INSUFFISANTES','Historique insuffisant'
    seasonal_ratio = current['ppg']/previous['ppg']
    recent_ratio = recent['ppg']/previous['ppg']
    if (blank>=3 and seasonal_ratio<=.80 and recent['shots_pg']>=1.2*previous['shots_pg']
            and recent['toi']>=previous['toi']):
        return 'REBOND A EXAMINER','Production basse; tirs et minutes maintenus ou augmentes'
    if (seasonal_ratio>=1.4 and recent_ratio>=2 and last3['ppg']>=2.5*previous['ppg']
            and recent['hit']>=.8):
        return 'SURCHAUFFE FORTE','Production tres elevee; comparer aux opportunites reellement creees'
    if ((streak>=5 and recent_ratio>=1.2) or
            (streak>=3 and seasonal_ratio>=1.4 and recent_ratio>=1.3)):
        return 'VIGILANCE SERIE','Serie et production elevees; confirmer le role et le volume'
    return 'PAS DE SIGNAL FORT','Toujours verifier PP1, adversaire et gardien'

def build(predictions,stats,target_date,pp=None):
    date = pd.Timestamp(target_date)
    year = date.year if date.month>=7 else date.year-1
    previous_season = (year-1)*10000+year
    current_season = year*10000+year+1
    predictions = predictions[predictions['date_match'].astype(str)==date.strftime('%Y-%m-%d')].copy()
    predictions['id_joueur'] = pd.to_numeric(predictions['id_joueur'],errors='coerce')
    stats = stats.copy()
    stats['id_match'] = stats['id_match'].astype(str)
    stats = stats[(stats['id_match'].str[4:6]=='02') &
                  (pd.to_datetime(stats['date_match'],errors='coerce')<date)].copy()
    for col in ['points','tirs']:
        stats[col] = pd.to_numeric(stats[col],errors='coerce')
    stats['id_joueur'] = pd.to_numeric(stats['id_joueur'],errors='coerce')
    stats['toi_minutes'] = stats['temps_de_glace'].map(minutes)
    if pp is not None:
        pp = pp[['id_joueur','id_match','ppTimeOnIce']].copy()
        pp['id_match'] = pp['id_match'].astype(str)
        pp['pp_minutes'] = pd.to_numeric(pp['ppTimeOnIce'],errors='coerce')/60
        if pp.duplicated(['id_joueur','id_match']).any():
            raise ValueError('Duplicate PP records')
        stats = stats.merge(pp[['id_joueur','id_match','pp_minutes']],
                            on=['id_joueur','id_match'],how='left',validate='many_to_one')
    else:
        stats['pp_minutes'] = np.nan
    stats = stats.sort_values(['date_match','id_match'])
    groups = dict(iter(stats.groupby('id_joueur')))
    rows=[]
    for p in predictions.to_dict('records'):
        hist = groups.get(p['id_joueur'],stats.iloc[0:0])
        prev = summarize(hist[hist.season_source==previous_season])
        cur = hist[hist.season_source==current_season]
        curr= summarize(cur)
        recent = summarize(cur.tail(5))
        last3 = summarize(cur.tail(3))
        streak = int(pd.to_numeric(p.get('current_point_streak_pre'),errors='coerce')
                     if pd.notna(p.get('current_point_streak_pre')) else 0)
        blank = int(pd.to_numeric(p.get('current_no_point_streak_pre'),errors='coerce')
                    if pd.notna(p.get('current_no_point_streak_pre')) else 0)
        verdict,reason = judge(prev,curr,recent,last3,streak,blank)
        rows.append(dict(date_match=target_date,id_joueur=p['id_joueur'],joueur=p['nom'],
                         equipe=p['team_player_match'],adversaire=p['adversaire_match'],
                         p_henachel=p['proba_point_1p_calibree'],
                         rang_henachel=p['rank_proba_sur_date'],
                         matchs_precedente=prev['games'],ppg_precedente=prev['ppg'],
                         ppg_saison=curr['ppg'],ppg_5=recent['ppg'],ppg_3=last3['ppg'],
                         tirs_precedente=prev['shots_pg'],tirs_5=recent['shots_pg'],
                         toi_precedente=prev['toi'],toi_5=recent['toi'],
                         pp_precedente=prev['pp'],pp_5=recent['pp'],
                         serie_points=streak,serie_sans_point=blank,
                         diagnostic=verdict,justification=reason,
                         statut='research_only_context'))
    return (pd.DataFrame(rows).sort_values('rang_henachel') if rows else pd.DataFrame(
        columns=['date_match','id_joueur','joueur','p_henachel','rang_henachel','diagnostic','statut']))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target-date',required=True)
    ap.add_argument('--predictions',type=Path,
        default=Path('outputs/predictions_upcoming_point_enrichi_calibre_v2.csv'))
    ap.add_argument('--stats',type=Path,default=Path('data/raw/stats.csv'))
    ap.add_argument('--pp-stats',type=Path,default=Path('data/raw/pp_stats_game.csv'))
    ap.add_argument('--output',type=Path,default=Path('outputs/player_sustainability_point.csv'))
    args=ap.parse_args()
    for path in [args.stats,args.predictions]:
        if not path.is_file(): raise FileNotFoundError(path)
    pred=pd.read_csv(args.predictions,low_memory=False)
    stats=pd.read_csv(args.stats,low_memory=False,usecols=COLS)
    pp=(pd.read_csv(args.pp_stats,usecols=['id_joueur','id_match','ppTimeOnIce'])
        if args.pp_stats.is_file() else None)
    result=build(pred,stats,args.target_date,pp)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print('Research-only structural audit:',len(result),'players')
    if len(result): print(result['diagnostic'].value_counts().to_string())

if __name__=='__main__': main()
