"""Validate evidence supplied by the bookmaker; never infer missing identities or times."""
import pandas as pd
from henachel.matching import utc, value_metrics, MAX_ODDS_AGE

REQUIRED = ('player_name','team','home_team','away_team','event_id','bookmaker',
            'date_match','event_start_utc','captured_at')


def validate_rows(rows, now):
    now=utc(now)
    errors={}
    events={}
    identities={}
    for index,row in enumerate(rows):
        problems=[]
        for field in REQUIRED:
            value=row.get(field)
            if value is None or not isinstance(value,str) or not value.strip() or value.lower() in {'nan','none','null'}:
                problems.append('missing_or_invalid:'+field)
        if row.get('market')!='player_points' or row.get('stat')!='points' or row.get('threshold')!=1 or row.get('outcome_key') not in {'1_plus','points_1_plus','1+','over_0.5'}:
            problems.append('unsupported_market')
        try:
            value_metrics(.5,row.get('odds_decimal'),row.get('implied_probability'))
        except (ValueError,TypeError,OverflowError):
            problems.append('invalid_odds')
        try:
            start,captured=utc(row.get('event_start_utc')),utc(row.get('captured_at'))
            if start<=now:problems.append('event_already_started')
            if not pd.Timedelta(0)<=now-captured<=MAX_ODDS_AGE:problems.append('stale_or_future_odds')
            if captured>=start:problems.append('capture_not_pre_match')
            date=pd.Timestamp(row.get('date_match'))
            if pd.isna(date) or str(row.get('date_match'))!=date.date().isoformat():problems.append('invalid_match_date')
        except (ValueError,TypeError):
            problems.append('invalid_date_or_timestamp')
        event=(str(row.get('bookmaker')),str(row.get('event_id')))
        signature=tuple(str(row.get(f)) for f in ['home_team','away_team','date_match','event_start_utc'])
        events.setdefault(event,{}).setdefault(signature,[]).append(index)
        identity=event+(str(row.get('player_name')),str(row.get('market')),str(row.get('threshold')))
        identities.setdefault(identity,[]).append(index)
        errors[index]=problems
    for signatures in events.values():
        if len(signatures)>1:
            for indices in signatures.values():
                for index in indices:errors[index].append('ambiguous_event')
    for indices in identities.values():
        if len(indices)>1:
            for index in indices:errors[index].append('duplicate_selection')
    rejected=[{'row_index':index,'reasons':reasons} for index,reasons in errors.items() if reasons]
    return dict(status='invalid_contract' if rejected else ('ok' if rows else 'no_markets'),
                rows=len(rows),valid_rows=len(rows)-len(rejected),rejected=rejected)
