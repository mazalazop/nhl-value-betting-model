"""Research-only structural pregame features. Unknown exposures remain unknown."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from henachel.features import normalize_team_code as normalize_team

KEYS = ['id_match', 'id_joueur']
STATS = {'points': 'points', 'buts': 'buts', 'tirs': 'tirs', 'toi': 'temps_de_glace', 'pp': 'temps_pp'}


def structural_features(frame, context, prior_history=None):
    current = frame.copy(); current['_requested'] = True
    if prior_history is not None:
        past = prior_history.copy(); past['_requested'] = False
        if not past.empty and pd.to_datetime(past.date_match).max() >= pd.to_datetime(current.date_match).min():
            raise ValueError('History must be strictly prior')
        current[list(STATS.values())] = np.nan
        work = pd.concat([past, current], ignore_index=True)
    else:
        work = current.reset_index(drop=True)
    work.date_match = pd.to_datetime(work.date_match, errors='raise')
    if work.duplicated(KEYS).any() or work.duplicated(['id_joueur', 'date_match']).any():
        raise ValueError('Ambiguous player/date or identity')
    work = work.sort_values(['id_joueur', 'date_match']).reset_index(drop=True)
    season = 'season_source'
    for col in ['team_player_match', 'adversaire_match']:
        work[col] = work[col].map(normalize_team)
    ctx = context.copy(); ctx.date_match = pd.to_datetime(ctx.date_match, errors='raise')
    ctx.team = ctx.team.map(normalize_team); ctx.opponent = ctx.opponent.map(normalize_team)
    if ctx.duplicated(['id_match', 'team']).any() or ctx.duplicated(['date_match', 'team']).any():
        raise ValueError('Ambiguous team context')
    joined = work[['id_match', 'team_player_match', 'date_match', season, 'adversaire_match']].merge(
        ctx[['id_match', 'team', 'date_match', season, 'opponent', 'ppTimeOnIcePerGame']],
        left_on=['id_match', 'team_player_match'], right_on=['id_match', 'team'], how='left', validate='many_to_one', suffixes=('', '_ctx'))
    found = joined.date_match_ctx.notna()
    if ((joined.loc[found, 'date_match'] != joined.loc[found, 'date_match_ctx']).any()
        or (joined.loc[found, season] != joined.loc[found, season+'_ctx']).any()
        or (joined.loc[found, 'adversaire_match'] != joined.loc[found, 'opponent']).any()):
        raise ValueError('Context identity mismatch')
    raw = pd.DataFrame(index=work.index)
    for name, col in STATS.items():
        raw[name] = pd.to_numeric(work[col], errors='raise')
        if (raw[name].dropna() < 0).any() or np.isinf(raw[name]).any():
            raise ValueError('Invalid structural statistic')
    for name, col in [('point_hit','points'), ('goal_hit','buts')]:
        raw[name] = raw[col].ge(1).astype(float).where(raw[col].notna())
    raw['team_pp_minutes'] = joined.ppTimeOnIcePerGame.to_numpy() / 60
    group_keys = [work.id_joueur, work[season]]
    def previous_total(values):
        return values.groupby(group_keys).transform(lambda x: x.fillna(0).cumsum().shift(1, fill_value=0))
    def previous_count(values):
        return previous_total(values.notna().astype(float))
    def last_season(values, operation):
        totals = values.groupby(group_keys).agg(operation)
        index = pd.MultiIndex.from_arrays([work.id_joueur, work[season] - 10001])
        return pd.Series(totals.reindex(index).to_numpy(), index=work.index)
    work['s_games_pre'] = work.groupby(['id_joueur', season]).cumcount().astype(float)
    work['s_prev_games'] = last_season(pd.Series(1., index=work.index), 'sum')
    for name in raw:
        if name == 'team_pp_minutes':
            continue
        x = raw[name]; n = previous_count(x); total = previous_total(x)
        previous = last_season(x, 'mean')
        work[f's_{name}_prev_mean'] = previous
        work[f's_{name}_current_mean'] = total / n.where(n > 0)
        if name not in ['toi', 'pp']:
            for k in [10, 30]:
                work[f's_{name}_blend{k}'] = (total + k * previous) / (n+k)
        if name in ['points', 'buts', 'tirs']:
            valid = x.notna() & raw.toi.notna()
            num = x.where(valid); den = raw.toi.where(valid)
            den_pre = previous_total(den)
            work[f's_{name}_current60'] = 60 * previous_total(num) / den_pre.where(den_pre > 0)
            den_last = last_season(den, 'sum')
            work[f's_{name}_prev60'] = 60 * last_season(num, 'sum') / den_last.where(den_last > 0)
    valid = raw.pp.notna() & raw.team_pp_minutes.notna()
    pp = raw.pp.where(valid); team_pp = raw.team_pp_minutes.where(valid)
    den = previous_total(team_pp); den_last = last_season(team_pp, 'sum')
    work['s_share_current'] = previous_total(pp) / den.where(den > 0)
    work['s_share_prev'] = last_season(pp, 'sum') / den_last.where(den_last > 0)
    def rolling(x, n, op='mean', minimum=1):
        return x.groupby(work.id_joueur).transform(lambda s: getattr(s.shift(1).rolling(n, min_periods=minimum), op)())
    work['s_pp10'] = rolling(raw.pp, 10)
    work['s_pp_sd10'] = rolling(raw.pp, 10, 'std', 5)
    work['s_pp_delta'] = rolling(raw.pp, 5) - work.s_pp10
    work['s_toi_delta'] = rolling(raw.toi, 5) - rolling(raw.toi, 10)
    for name in ['points', 'buts']:
        mean = rolling(raw[name], 5); n = rolling(raw[name], 5, 'count', 0)
        prev = work[f's_{name}_prev_mean']
        work[f's_{name}_recent_shrunk'] = (n * mean + 10 * prev) / (n+10)
        work[f's_{name}_recent_gap'] = mean - prev
    # Team statistics become available on their date; lookup is strictly earlier.
    ctx = ctx.sort_values(['team', season, 'date_match']).reset_index(drop=True)
    groups = [ctx.team, ctx[season]]
    cols = ['ppOpportunities','totalPenaltiesDrawn','penalties','timesShorthanded','powerPlayGoalsFor','ppGoalsAgainst']
    for col in cols:
        ctx[col] = pd.to_numeric(ctx[col], errors='raise')
        if (ctx[col].dropna() < 0).any() or np.isinf(ctx[col]).any():
            raise ValueError('Invalid special teams statistic')
        ctx[col+'_rate'] = ctx[col].groupby(groups).transform(lambda x: x.expanding().mean())
        ctx[col+'_n'] = ctx[col].notna().astype(int).groupby(groups).cumsum()
    for name, num, den in [('pp_eff','powerPlayGoalsFor','ppOpportunities'),('pk_allowed','ppGoalsAgainst','timesShorthanded')]:
        valid = ctx[num].notna() & ctx[den].notna()
        numerator = ctx[num].where(valid).fillna(0).groupby(groups).cumsum()
        denominator = ctx[den].where(valid).fillna(0).groupby(groups).cumsum()
        ctx[name] = numerator / denominator.where(denominator > 0)
    def lookup(side, columns):
        left = work[['date_match', season, side]].rename(columns={side:'team'}).assign(_index=work.index)
        out = pd.merge_asof(left.sort_values('date_match'), ctx[['date_match',season,'team']+columns].sort_values('date_match'),
                            on='date_match', by=[season,'team'], direction='backward', allow_exact_matches=False)
        return out.set_index('_index').reindex(work.index)
    own = lookup('team_player_match', ['ppOpportunities_rate','ppOpportunities_n','totalPenaltiesDrawn_rate','pp_eff'])
    opp = lookup('adversaire_match', ['penalties_rate','timesShorthanded_rate','timesShorthanded_n','pk_allowed'])
    work['team_pp_opportunities_rate_pre'] = own.ppOpportunities_rate
    work['team_penalties_drawn_rate_pre'] = own.totalPenaltiesDrawn_rate
    work['opponent_penalties_taken_rate_pre'] = opp.penalties_rate
    work['opponent_times_shorthanded_rate_pre'] = opp.timesShorthanded_rate
    work['team_pp_efficiency_pre'] = own.pp_eff
    work['opponent_pk_efficiency_pre'] = 1 - opp.pk_allowed
    work['expected_pp_opportunities'] = ((own.ppOpportunities_rate + opp.timesShorthanded_rate) / 2).where(own.ppOpportunities_n.ge(5) & opp.timesShorthanded_n.ge(5))
    work['player_pp_exposure_interaction'] = work.s_share_current * work.expected_pp_opportunities
    return work[work._requested].drop(columns='_requested').sort_values(['date_match']+KEYS).reset_index(drop=True)
