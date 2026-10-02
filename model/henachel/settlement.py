"""POINT 1+ outcomes. DNP/cancellations require bookmaker evidence; never guess a void."""
import math
import pandas as pd
from henachel.data import game_state
from henachel.history import validate_ledger


def settle(history,stats,matches,settled_at):
    validate_ledger(history)
    if stats.duplicated(['id_match','id_joueur']).any(): raise ValueError('Duplicate player results')
    if matches.id_match.duplicated().any(): raise ValueError('Duplicate match status')
    results=stats.set_index(['id_match','id_joueur']);games=matches.set_index('id_match')
    out=history.copy();changes=[];unresolved=[]
    for i,bet in out.iterrows():
        reason=None; key=(bet.id_match,bet.id_joueur)
        state=game_state(games.loc[bet.id_match,'status'],games.loc[bet.id_match].get('schedule_state')) if bet.id_match in games.index else 'unknown'
        if state!='final': reason=f'match_{state}'
        elif bet.get('stat')!='points' or bet.get('market')!='player_points' or float(bet.get('threshold',0))!=1: reason='unsupported_market'
        elif bet.get('outcome_key') not in {'1_plus','points_1_plus','1+','over_0.5'}: reason='unsupported_outcome'
        elif key not in results.index: reason='missing_player_or_dnp_requires_bookmaker_rule'
        else:
            row=results.loc[key]
            try:
                points,goals,assists=[float(row.get(c,float('nan'))) for c in ['points','buts','passes']]
                if not all(math.isfinite(x) and x>=0 and x.is_integer() for x in [points,goals,assists]) or points!=goals+assists:
                    reason='missing_or_inconsistent_stats'
            except (TypeError,ValueError): reason='missing_or_inconsistent_stats'
            if not reason:
                result='win' if points>=1 else 'loss'
                if bet.get('result')!=result or bet.get('actual_stat_value')!=points or bet.get('bet_status')!='settled':
                    changes.append(dict(bet_id=bet.bet_id,previous_result=bet.get('result'),previous_value=bet.get('actual_stat_value'),result=result,actual_stat_value=points,settled_at=settled_at))
                    out.at[i,'result']=result;out.at[i,'actual_stat_value']=points
                    out.at[i,'bet_status']='settled';out.at[i,'settled_at']=settled_at
        if reason: unresolved.append(dict(bet_id=bet.bet_id,reason=reason))
    return out,pd.DataFrame(changes,columns=['bet_id','previous_result','previous_value','result','actual_stat_value','settled_at']),pd.DataFrame(unresolved,columns=['bet_id','reason'])
