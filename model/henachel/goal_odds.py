"""Research-only GOAL odds join; probabilities and production POINT stay unchanged."""
from collections import Counter
import math
import pandas as pd
from henachel.matching import name,utc,value_metrics,MAX_ODDS_AGE


def match_goal_odds(predictions, rows, now=None):
    clock=utc(now or pd.Timestamp.now(tz='UTC'))
    if predictions.duplicated(['game_id','player_id']).any():raise ValueError('Duplicate predictions')
    accepted=[];rejected=[];valid=[]
    event_signatures={}
    for row in rows:
        event=str(row.get('event_id'))
        signature=tuple(str(row.get(k)) for k in ['nhl_game_id','home_team','away_team','event_start_utc','date_match'])
        event_signatures.setdefault(event,set()).add(signature)
    for index,row in enumerate(rows):
        try:
            if row.get('market')!='player_goals' or row.get('stat')!='goals' or row.get('threshold')!=1 or row.get('outcome_key')!='1_plus':raise ValueError('unsupported_goal_market')
            for key in ['bookmaker','event_id','team','player_name','source_market_id','source_outcome_id']:
                if pd.isna(row.get(key)) or not str(row.get(key,'')).strip():raise ValueError('missing_identity')
            if row.get('suspended') or row.get('hidden'):raise ValueError('suspended_market')
            if len(event_signatures[str(row['event_id'])])!=1:raise ValueError('ambiguous_event')
            start,captured=utc(row.get('event_start_utc')),utc(row.get('collected_at_utc'))
            if start<=clock or not pd.Timedelta(0)<=clock-captured<=MAX_ODDS_AGE:raise ValueError('stale_or_started')
            observed=utc(row.get('roster_observed_at'))
            if not pd.Timedelta(0)<=clock-observed<=pd.Timedelta(hours=24):raise ValueError('stale_roster')
            ids=[]
            for key in ['nhl_game_id','nhl_player_id']:
                val=float(row.get(key))
                if not math.isfinite(val) or val<=0 or val!=int(val):raise ValueError('invalid_nhl_id')
                ids.append(int(val))
            person=predictions[predictions.game_id.eq(ids[0])&predictions.player_id.eq(ids[1])]
            if len(person)!=1:raise ValueError('no_unique_model_identity')
            p=person.iloc[0]
            if name(p.player_name)!=name(row['player_name']) or p.team!=row['team']:raise ValueError('player_team_conflict')
            if {p.team,p.opponent}!={row.get('home_team'),row.get('away_team')} or p.team==p.opponent:raise ValueError('event_team_conflict')
            if utc(p.event_start_utc)!=start or pd.Timestamp(p.game_date).date()!=pd.Timestamp(row['date_match']).date():raise ValueError('event_date_conflict')
            metrics=value_metrics(p.probability_goal,row.get('odds_decimal'),row.get('implied_probability'))
            valid.append((index,dict(p),row,metrics))
        except (ValueError,TypeError,KeyError,OverflowError) as exc:
            rejected.append(dict(index=index,reason=str(exc)))
    counts=Counter((p['game_id'],p['player_id'],r['bookmaker']) for _,p,r,_ in valid)
    for index,p,row,metrics in valid:
        if counts[(p['game_id'],p['player_id'],row['bookmaker'])]!=1:
            rejected.append(dict(index=index,reason='duplicate_candidate'));continue
        accepted.append(dict(p,**{k:row[k] for k in ['bookmaker','event_id','collected_at_utc','source_market_id','source_outcome_id']},
                             odds_decimal=metrics['odds_decimal'],implied_probability=metrics['implied_probability'],
                             edge=metrics['edge_probability'],expected_value=metrics['ev_per_unit']))
    columns=list(predictions.columns)+['bookmaker','event_id','collected_at_utc','source_market_id','source_outcome_id','odds_decimal','implied_probability','edge','expected_value']
    output=pd.DataFrame(accepted,columns=columns)
    output=output.sort_values(['expected_value','goal_probability_rank'],ascending=[False,True]).reset_index(drop=True)
    output['goal_value_rank']=range(1,len(output)+1)
    # Operational experimental policy, not a claim of validated betting performance.
    output['experimental_value_eligible']=output.expected_value.gt(0)&output.edge.gt(0)
    return output,rejected


def annotate_probability_ranking(predictions, matched):
    """Keep every scored player and exact model probability, including no-odds days."""
    columns = ['game_id', 'player_id', 'bookmaker', 'odds_decimal',
               'implied_probability', 'edge', 'expected_value', 'goal_value_rank']
    # One bookmaker is currently supported; never pick an arbitrary duplicate.
    if matched.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Ambiguous ranking odds')
    return predictions.merge(matched[columns], on=['game_id', 'player_id'],
                             how='left', validate='one_to_one', sort=False)
