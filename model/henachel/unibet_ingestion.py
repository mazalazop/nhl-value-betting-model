"""Unibet public SSR POINT 1+ ingestion with explicit NHL identity evidence."""
import json
import re
import pandas as pd
from henachel.matching import name, utc, value_metrics, MAX_ODDS_AGE

# Explicit bookmaker spellings, not fuzzy team guesses.
ALIASES={'BUF Sabres':'BUF','CHI Blackhawks':'CHI','ANA Ducks':'ANA','FLO Panthers':'FLA',
 'CLR Avalanche':'COL','STL Blues':'STL','CLB B.Jackets':'CBJ','CLB Bjackets':'CBJ',
 'UTA HockeyClub':'UTA','UTA Hockeyclub':'UTA','CAR Hurricanes':'CAR','WAS Capitals':'WSH',
 'BOS Bruins':'BOS','CAL Flames':'CGY','DAL Stars':'DAL','DET Red Wings':'DET',
 'EDM Oilers':'EDM','LA Kings':'LAK','MIN Wild':'MIN','MON Canadiens':'MTL',
 'NAS Predators':'NSH','NJ Devils':'NJD','NY Islanders':'NYI','NY Rangers':'NYR',
 'OTT Senators':'OTT','PHI Flyers':'PHI','PIT Penguins':'PIT','SEA Kraken':'SEA',
 'SJ Sharks':'SJS','TB Lightning':'TBL','TOR Maple Leafs':'TOR','TOR MapleLeafs':'TOR','VAN Canucks':'VAN',
 'VEG Golden Knights':'VGK','WIN Jets':'WPG'}


def public_events(html):
    found=re.findall(r'<script\b[^>]*\bid="serverApp-state"[^>]*>(.*?)</script>',html,re.S)
    if len(found)!=1:raise ValueError('Missing or ambiguous public SSR state')
    state=json.loads(found[0])
    events=state.get('EventsDetail',{}).get('events')
    if not isinstance(events,list):raise ValueError('Missing event details')
    return events


def normalize_event(event,url,captured_at,matches,roster,now):
    now,captured,start=utc(now),utc(captured_at),utc(event.get('parsedStart'))
    if not pd.Timedelta(0)<=now-captured<=MAX_ODDS_AGE or start<=now:
        raise ValueError('Stale capture or started event')
    if event.get('sportCode')!='ICEH' or event.get('path',{}).get('league',{}).get('label')!='NHL':
        raise ValueError('Not an NHL event')
    event_id=event.get('id')
    if not isinstance(event_id,int) or event_id<=0 or f'/nhl/{event_id}/' not in url:
        raise ValueError('Event URL identity mismatch')
    aliases={name(k):v for k,v in ALIASES.items()}
    teams={aliases.get(name(event.get(k,{}).get('label'))) for k in ['opponentA','opponentB']}
    if None in teams or len(teams)!=2:raise ValueError('Unknown or ambiguous event teams')
    games=matches[
        pd.to_datetime(matches.start_time_utc,utc=True).eq(start)
        & matches.id_equipe_domicile.isin(teams)&matches.id_equipe_exterieur.isin(teams)
        & matches.status.isin(['FUT','PRE'])]
    if len(games)!=1:raise ValueError('No unique scheduled NHL event')
    game=games.iloc[0]
    pool=roster[roster.id_equipe.isin(teams)].copy()
    observed=pd.to_datetime(pool.observed_at,utc=True,errors='coerce')
    pool=pool[(now-observed).between(pd.Timedelta(0),pd.Timedelta(hours=24))]
    rows=[];rejected=[]
    for group in event.get('groupedMarkets',[]):
        if group.get('id')!=4267 or group.get('description')!='Nombre de Points - Joueur - Match (Hors TAB)':continue
        for market in group.get('markets',[]):
            if market.get('suspended') or market.get('period')!='Match' or market.get('parent')!=f'e{event_id}':continue
            prefix='Nombre de Points - '
            if not str(market.get('description','')).startswith(prefix):continue
            player=market['description'][len(prefix):]
            choices=[o for o in market.get('outcomes',[]) if o.get('description')==player+' 1+' and not o.get('hidden') and not o.get('suspended')]
            if len(choices)!=1:
                rejected.append({'player':player,'reason':'missing_or_ambiguous_1_plus'});continue
            outcome=choices[0]
            if outcome.get('eventId')!=event_id or outcome.get('marketId')!=market.get('id') or outcome.get('groupId')!=4267:
                rejected.append({'player':player,'reason':'conflicting_outcome_identity'});continue
            people=pool[pool.nom.map(name).eq(name(player))]
            if len(people)!=1:
                rejected.append({'player':player,'reason':'no_unique_recent_roster_identity'});continue
            person=people.iloc[0]
            odds=float(str(outcome.get('price')).replace(',','.'));value_metrics(.5,odds)
            rows.append(dict(bookmaker='Unibet',market='player_points',stat='points',threshold=1,
                outcome_key='1_plus',outcome_label='1+',event_id=str(event_id),event_url=url,
                event_slug=url.rstrip('/').rsplit('/',1)[-1],
                home_team=game.id_equipe_domicile,away_team=game.id_equipe_exterieur,
                team=person.id_equipe,player_name=player,date_match=str(game.date_match),
                event_start_utc=start.isoformat(),captured_at=captured.isoformat(),collected_at_utc=captured.isoformat(),
                nhl_game_id=int(game.id_match),nhl_player_id=int(person.id_joueur),
                odds_decimal=odds,implied_probability=1/odds,
                identity_source='NHL schedule exact start and teams + unique exact recent roster name',
                roster_observed_at=person.observed_at,source_market_id=market['id'],source_outcome_id=outcome['id']))
    return rows,rejected


def normalized_payload(rows, events, contract):
    """The existing 06 wire schema; metadata describes only the explicitly parsed market."""
    return dict(bookmaker='Unibet',market='player_points',stat='points',threshold=1,
                outcome_label='1+',rows=rows,rows_count=len(rows),source='unibet_public_ssr',
                events=events,contract=contract)
