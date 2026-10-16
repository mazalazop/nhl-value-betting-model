"""Bounded official NHL context. No lineup/PP confirmation inferred from absence."""
from collections import defaultdict
from urllib.parse import urlencode
import math
import pandas as pd
from henachel.context import clock
from henachel.matching import utc, name
from henachel.prospective_http import get_json
from henachel.prospective_pp import collect as collect_pp

API='https://api-web.nhle.com/v1'


def mean_known(values):
    if not values or any(v is None for v in values): return None
    numbers=[float(v) for v in values]
    return sum(numbers)/len(numbers) if all(math.isfinite(v) and v>=0 for v in numbers) else None


def minutes(value):
    if value is None:return None
    parts=str(value).split(':')
    if len(parts)!=2:return None
    return int(parts[0])+int(parts[1])/60


def aggregate(boxes, target_id, team, player_id, cutoff):
    """Only FINAL observed before cutoff, target excluded; missing != zero."""
    games=[];gf=[];ga=[];shots=[];against=[];toi=[];pp=[];player_shots=[];played=[]
    for saved in boxes:
        b=saved['document'];observed=utc(saved['observed_at'])
        if b['id']==target_id:raise ValueError('Target game in context aggregate')
        if b.get('gameState') not in {'OFF','FINAL'} or not utc(b['startTimeUTC']) < observed <= utc(cutoff):
            raise ValueError('Game not proven final before cutoff')
        sides=[s for s in ['homeTeam','awayTeam'] if b[s]['abbrev']==team]
        if len(sides)!=1:raise ValueError('Boxscore team mismatch')
        side=sides[0];other='awayTeam' if side=='homeTeam' else 'homeTeam'
        games.append(dict(game_id=b['id'],event_start_utc=b['startTimeUTC'],final_observed_at_utc=saved['observed_at'],status=b['gameState'],game_date=b['gameDate']))
        gf.append(b[side].get('score'));ga.append(b[other].get('score'));shots.append(b[side].get('sog'));against.append(b[other].get('sog'))
        stats=b.get('playerByGameStats',{}).get(side,{})
        for goalie in stats.get('goalies',[]):
            if minutes(goalie.get('toi')) not in {None,0}:played.append(dict(player_id=goalie['playerId'],game_id=b['id'],game_date=b['gameDate'],toi=goalie.get('toi')))
        people=[p for group in ['forwards','defense'] for p in stats.get(group,[]) if p.get('playerId')==player_id]
        if len(people)>1:raise ValueError('Duplicate player in boxscore')
        if people:
            p=people[0];toi.append(minutes(p.get('toi')));pp.append(minutes(p.get('powerPlayToi')));player_shots.append(p.get('sog'))
    return dict(team=dict(goals_for_per_game=mean_known(gf),goals_against_per_game=mean_known(ga),shots_for_per_game=mean_known(shots),shots_against_per_game=mean_known(against)),
                player=dict(toi_minutes=mean_known(toi),pp_toi_minutes=mean_known(pp),shots=mean_known(player_shots),observed_appearances=len(toi)),
                games=games,goalies_played=played)


def observation(p,kind,value,source,url,observed=None,members=None,reason=None,player=True):
    r=dict(schema_version=1,game_id=int(p['game_id']),game_date=p['game_date'],team=p['team'],
           player_id=int(p['player_id']) if player else None,event_start_utc=p['event_start_utc'],
           collected_at_utc=clock(),source=source,source_url=url,source_timestamp=observed,
           source_timestamp_kind='retrieval_of_official_data' if observed else 'publication_time_unavailable',
           kind=kind,value=value,evidence_status='CONFIRMED' if observed and value is not None else 'UNKNOWN',
           max_age_seconds=7200,reason=reason)
    if members is not None:
        r.update(stat_window='last up to 5 completed team games; player present appearances only',games_used=members,
                 data_through_utc=max((g['final_observed_at_utc'] for g in members),default=None))
    return r


def collect_context(baselines, session, cache, enable_pp=False, now_fn=clock):
    if not baselines:return [],[]
    unique={ (b['model_baseline']['game_id'],b['model_baseline']['player_id']): b['model_baseline'] for b in baselines }
    rows=[];errors=[];by_team={};schedules={};pp_evidence=None
    # Recheck time immediately before each network operation, not only at job start.
    blocked=set()
    def eligible(p): return p['game_id'] not in blocked and utc(now_fn())<utc(p['event_start_utc'])
    def fetch(url,ttl,p=None):
        class Guarded:
            def get(self,*args,**kwargs):
                if not (eligible(p) if p else any(eligible(x) for x in unique.values())):
                    raise ValueError('Puck drop reached before request/retry')
                return session.get(*args,**kwargs)
        if not (eligible(p) if p else any(eligible(x) for x in unique.values())):
            raise ValueError('Puck drop reached')
        return get_json(Guarded(),url,cache,ttl)

    for p in unique.values():
        if not eligible(p):continue
        key=(p['game_id'],p['team'])
        if key in by_team:continue
        url=f"{API}/club-schedule-season/{p['team']}/now"
        try:
            saved=fetch(url,1800,p);schedule=saved['document']['games'];schedules[key]=saved
            target=[g for g in schedule if g['id']==p['game_id']]
            if len(target)!=1 or target[0]['gameState'] not in {'PRE','FUT'} or utc(target[0]['startTimeUTC'])!=utc(p['event_start_utc']):
                blocked.add(p['game_id'])
                raise ValueError('Target event changed or started')
            prior=[g for g in schedule if g['id']!=p['game_id'] and g.get('gameType') in {2,3}
                   and g['gameState'] in {'OFF','FINAL'} and utc(g['startTimeUTC'])<utc(now_fn())]
            prior=sorted(prior,key=lambda g:g['startTimeUTC'])[-5:]
            boxes=[]
            for g in prior:
                if not eligible(p):break
                box=fetch(f"{API}/gamecenter/{g['id']}/boxscore",3600,p)
                if box['document']['id']!=g['id']:raise ValueError('Wrong boxscore identity')
                boxes.append(box)
            by_team[key]=boxes
        except Exception as exc:
            errors.append(dict(source=url,error=type(exc).__name__,detail=str(exc)[:200],game_id=p['game_id']));by_team[key]=[]
    # Official game-level special teams reports, only IDs with a proven final boxscore.
    ids=sorted({b['document']['id'] for boxes in by_team.values() for b in boxes})
    special={}
    fields={'powerplay':['ppOpportunities','powerPlayGoalsFor','ppTimeOnIcePerGame'],
            'penaltykill':['timesShorthanded','ppGoalsAgainst'], 'penalties':['totalPenaltiesDrawn','penalties']}
    if ids and any(eligible(p) for p in unique.values()):
        for report in fields:
            if not any(eligible(p) for p in unique.values()):break
            url='https://api.nhle.com/stats/rest/en/team/'+report+'?'+urlencode(dict(isAggregate='false',isGame='true',start=0,limit=500,cayenneExp=' or '.join(f'gameId={i}' for i in ids)))
            try:
                saved=fetch(url,3600)
                if saved['document']['total']!=len(saved['document']['data']):raise ValueError('Truncated report')
                special[report]=saved
            except Exception as exc:errors.append(dict(source=url,error=type(exc).__name__,detail=str(exc)[:200]))
    player_pp=None
    if ids and any(eligible(p) for p in unique.values()):
        expression=' or '.join(f'gameId={i}' for i in ids)
        try:
            merged=[];urls=[];stamps=[];total=None
            # NHL clamps this report to 100 rows even when a larger limit is sent.
            # Deterministic paging, bounded to 2,000 rows; no silent truncation.
            for page in range(20):
                url='https://api.nhle.com/stats/rest/en/skater/powerplay?'+urlencode(dict(isAggregate='false',isGame='true',start=len(merged),limit=100,
                    sort='[{"property":"gameId","direction":"ASC"},{"property":"playerId","direction":"ASC"}]',cayenneExp=expression))
                chunk=fetch(url,3600);document=chunk['document']
                if total is not None and document['total']!=total:raise ValueError('PP pagination changed during collection')
                total=document['total'];merged.extend(document['data']);urls.append(url);stamps.append(chunk['observed_at'])
                if len(merged)==total:break
                if not document['data']:raise ValueError('Incomplete PP page')
            if len(merged)!=total or len({(r['gameId'],r['playerId']) for r in merged})!=len(merged):
                raise ValueError('Truncated or duplicate player PP data; remains UNKNOWN')
            player_pp=dict(document=dict(data=merged,total=total),url=';'.join(urls),observed_at=min(stamps))
        except Exception as exc:
            errors.append(dict(source='NHL skater powerplay',error=type(exc).__name__,detail=str(exc)[:200]));player_pp=None
    if enable_pp and any(eligible(p) for p in unique.values()):
        pp_evidence=collect_pp(session)
        if pp_evidence.get('error_type'):errors.append(dict(source='RotoWire',error=pp_evidence['error_type']))
    seen_teams=set()
    for p in unique.values():
        if not eligible(p):continue
        key=(p['game_id'],p['team']);boxes=by_team.get(key,[])
        try:
            data=aggregate(boxes,p['game_id'],p['team'],p['player_id'],now_fn())
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(dict(source='NHL aggregate',detail=str(exc)));data=dict(games=[],team=None,player=None,goalies_played=[])
        members=data['games'];stamp=max((b['observed_at'] for b in boxes),default=None)
        if key not in seen_teams:
            seen_teams.add(key)
            rows.append(observation(p,'team_stats',data['team'] if members else None,'NHL final boxscores',API,stamp,members,player=False))
            sched=schedules.get(key);dates=[g['game_date'] for g in members]
            calendar=dict(previous_game_date=max(dates),rest_calendar_days=(pd.Timestamp(p['game_date'])-pd.Timestamp(max(dates))).days-1,
                          back_to_back=(pd.Timestamp(p['game_date'])-pd.Timestamp(max(dates))).days==1,
                          goalies_played_recently=data['goalies_played']) if dates else None
            rows.append(observation(p,'calendar',calendar,'NHL schedule and final boxscores',API,sched['observed_at'] if sched else None,members,player=False))
            values={};sources=[]
            for report,names in fields.items():
                selected=[]
                for b in boxes:
                    doc=b['document'];other=next((doc[s]['abbrev'] for s in ['homeTeam','awayTeam'] if doc[s]['abbrev']!=p['team']),None)
                    matches=[r for r in special.get(report,{}).get('document',{}).get('data',[]) if r['gameId']==doc['id'] and r['opponentTeamAbbrev']==other]
                    if len(matches)!=1 or pd.Timestamp(matches[0]['gameDate']).date()!=pd.Timestamp(doc['gameDate']).date():selected=[];break
                    if utc(special[report]['observed_at']) < utc(b['observed_at']):selected=[];break
                    selected.append(matches[0])
                for field in names:values[field]=mean_known([r.get(field) for r in selected]) if len(selected)==len(boxes) else None
                if report in special:sources.append(special[report]['url'])
            values['pp_efficiency']=values['powerPlayGoalsFor']/values['ppOpportunities'] if values.get('ppOpportunities') and values.get('powerPlayGoalsFor') is not None else None
            values['pk_efficiency']=1-values['ppGoalsAgainst']/values['timesShorthanded'] if values.get('timesShorthanded') and values.get('ppGoalsAgainst') is not None else None
            rows.append(observation(p,'special_teams',values if any(v is not None for v in values.values()) else None,'NHL game reports',';'.join(sources),max((v['observed_at'] for v in special.values()),default=None),members,player=False))
            for kind in ['goalie','goalie_season','travel','dangerous_shots']:
                rows.append(observation(p,kind,None,'unavailable',API,reason='No verified pregame source/complete season coverage in this bounded collector',player=False))
        usage=data['player']
        if usage and player_pp:
            pp_values=[];team_seconds=[]
            for b in boxes:
                doc=b['document'];side=next(s for s in ['homeTeam','awayTeam'] if doc[s]['abbrev']==p['team'])
                present=any(x['playerId']==p['player_id'] for group in ['forwards','defense'] for x in doc.get('playerByGameStats',{}).get(side,{}).get(group,[]))
                if not present:continue
                candidates=[x for x in player_pp['document']['data'] if x['gameId']==doc['id'] and x['playerId']==p['player_id']]
                if len(candidates)!=1 or candidates[0].get('teamAbbrev')!=p['team'] or utc(player_pp['observed_at'])<utc(b['observed_at']) or pd.Timestamp(candidates[0]['gameDate']).date()!=pd.Timestamp(doc['gameDate']).date():pp_values.append(None)
                else:pp_values.append(candidates[0].get('ppTimeOnIce'))
                other=next(doc[s]['abbrev'] for s in ['homeTeam','awayTeam'] if doc[s]['abbrev']!=p['team'])
                team_rows=[x for x in special.get('powerplay',{}).get('document',{}).get('data',[]) if x['gameId']==doc['id'] and x['opponentTeamAbbrev']==other]
                team_seconds.append(team_rows[0].get('ppTimeOnIcePerGame') if len(team_rows)==1 else None)
            pp_seconds=mean_known(pp_values);denominator=mean_known(team_seconds)
            usage['pp_toi_minutes']=pp_seconds/60 if pp_seconds is not None else None
            usage['pp_share']=pp_seconds/denominator if pp_seconds is not None and denominator and pp_seconds<=denominator else None
        rows.append(observation(p,'player_usage',usage if usage and usage['observed_appearances'] else None,'NHL final boxscores and skater powerplay',
                                API+(';'+player_pp['url'] if player_pp else ''),stamp,members))
        for kind in ['availability','line']:
            rows.append(observation(p,kind,None,'unavailable',API,reason='Roster absence is not evidence of injury/scratch; no verified timestamped lineup source'))
        pp=[r for r in (pp_evidence or {}).get('observations',[]) if r.get('team')==p['team'] and name(r.get('player_name'))==name(p['player_name'])]
        rows.append(observation(p,'pp_unit',pp[0].get('observed_unit') if len(pp)==1 else None,'RotoWire' if pp_evidence else 'disabled',
                                (pp_evidence or {}).get('source_url',''),reason='No verified publication timestamp; explicit raw unit is not confirmation'))
    return rows,errors
