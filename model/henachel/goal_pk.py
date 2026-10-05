"""Strictly earlier opponent PK; regular and playoff games retain source season."""
import numpy as np
import pandas as pd

FEATURE='goal_opponent_pk_10_pre'


def add_opponent_pk(frame, source):
    pk=source.copy();out=frame.copy()
    if pk.duplicated(['id_match','team']).any():raise ValueError('Duplicate PK identities')
    for col in ['pp_goals_against','times_shorthanded']:
        pk[col]=pd.to_numeric(pk[col],errors='raise')
        if pk[col].isna().any() or not np.isfinite(pk[col]).all() or (pk[col]<0).any() or (pk[col]%1!=0).any():
            raise ValueError('Invalid PK counts')
    if (pk.pp_goals_against>pk.times_shorthanded).any():raise ValueError('Invalid PK goals/opportunities')
    for data in [pk,out]:
        data['date_match']=pd.to_datetime(data.date_match,errors='raise')
        data['season_source']=pd.to_numeric(data.season_source,errors='raise').astype('int64')
    pk=pk.sort_values(['date_match','id_match','team'])
    sums=pk.groupby(['team','season_source'])[['pp_goals_against','times_shorthanded']].transform(lambda x:x.rolling(10,min_periods=1).sum())
    pk[FEATURE]=1-sums.pp_goals_against/sums.times_shorthanded.where(sums.times_shorthanded>0)
    snapshots=pk.rename(columns={'team':'adversaire_match','date_match':'pk_source_date'})[
        ['adversaire_match','season_source','pk_source_date',FEATURE]]
    if snapshots.duplicated(['adversaire_match','season_source','pk_source_date']).any():
        raise ValueError('Ambiguous same-day PK snapshots')
    out['_pk_order']=range(len(out))
    out=pd.merge_asof(out.drop(columns=[FEATURE,'pk_source_date'],errors='ignore').sort_values('date_match'),
                      snapshots.sort_values('pk_source_date'),left_on='date_match',right_on='pk_source_date',
                      by=['adversaire_match','season_source'],direction='backward',allow_exact_matches=False)
    stale=(out.date_match-out.pk_source_date)>pd.Timedelta(days=30)
    out.loc[stale,FEATURE]=np.nan
    return out.sort_values('_pk_order').drop(columns='_pk_order').reset_index(drop=True)
