"""Append-only research shadow captures; no Sheets, picks or production writes."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from henachel import but_v2 as v2
from henachel.but_v2_validation import metrics, paired, write


def utc(values):
    # Naive timestamps cannot establish pregame provenance.
    parsed=[pd.Timestamp(x) for x in values]
    if any(pd.isna(x) or x.tzinfo is None for x in parsed):raise ValueError('Aware UTC timestamps required')
    return pd.DatetimeIndex(parsed).tz_convert('UTC')


def capture(history, slate, reference, output, provenance, now=None):
    """now is an injectable test clock; CLI always uses the real clock."""
    started=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    v2.validate(slate,outcomes=False)
    if slate.empty or slate.date_match.nunique()!=1:raise ValueError('Exactly one nonempty NHL date required')
    date=pd.Timestamp(slate.date_match.iloc[0])
    starts=utc(slate.start_time_utc)
    if (starts<=started).any():raise ValueError('Puck drop passed; no backdated shadow capture')
    for col in ['buts','points','passes','tirs','temps_de_glace','temps_pp','a_marque_un_but']:
        if col in slate and slate[col].notna().any():raise ValueError('Future observations forbidden')
    if not slate.status.isin(['FUT','PRE']).all():raise ValueError('Upcoming games only')
    if not slate.schedule_state.eq('OK').all():raise ValueError('Schedule not confirmed')
    history=history[pd.to_datetime(history.date_match)<date].copy()
    v2.validate(history)
    if reference.duplicated(v2.KEYS).any():raise ValueError('Duplicate V1 identity')
    if set(map(tuple,reference[v2.KEYS].to_numpy()))!=set(map(tuple,slate[v2.KEYS].to_numpy())):
        raise ValueError('V1 and V2 universes must be identical')
    for col in ['team_player_match','date_match']:
        if col in reference:
            check=slate[v2.KEYS+[col]].merge(reference[v2.KEYS+[col]],on=v2.KEYS,validate='one_to_one')
            if not check[col+'_x'].astype(str).eq(check[col+'_y'].astype(str)).all():raise ValueError('V1 metadata mismatch')
    refs=utc(reference.generated_at_utc)
    refstarts=utc(reference.start_time_utc)
    if (refs>=refstarts).any() or (refs>started).any() or ((started-refs)>pd.Timedelta(hours=2)).any():raise ValueError('V1 is not pregame')
    p=pd.to_numeric(reference.probability_goal,errors='raise')
    if not p.between(0,1).all():raise ValueError('Invalid V1 probability')
    pricecols=['odds_decimal','odds_captured_at_utc','odds_source','odds_market']
    if set(pricecols)<=set(reference):
        priced=reference.odds_decimal.notna()
        odds=pd.to_numeric(reference.loc[priced,'odds_decimal'],errors='raise')
        if not np.isfinite(odds).all() or not odds.gt(1).all():raise ValueError('Invalid odds')
        if not reference.loc[priced,'odds_market'].eq('GOAL_1_PLUS').all():raise ValueError('Wrong odds market')
        quotes=utc(reference.loc[priced,'odds_captured_at_utc'])
        if (quotes>started).any() or (quotes>=refstarts[priced]).any() or ((started-quotes)>pd.Timedelta(hours=2)).any():raise ValueError('Postgame odds forbidden')
        if reference.loc[priced,'odds_source'].isna().any():raise ValueError('Odds source required')
    else:
        if 'odds_decimal' in reference and reference.odds_decimal.notna().any():raise ValueError('Incomplete odds provenance')
        reference=reference.copy()
        for col in pricecols:reference[col]=np.nan
    keep=v2.KEYS+['probability_goal','generated_at_utc','start_time_utc']+pricecols
    keep += [c for c in ['model_name','model_sha256','config_sha','algorithm_id','algorithm_manifest_sha256'] if c in reference]
    reference=reference[keep].rename(columns={'probability_goal':'v1_probability','generated_at_utc':'v1_generated_at_utc',
                                            'start_time_utc':'v1_start_time_utc',**{c:'v1_'+c for c in ['model_name','model_sha256','config_sha','algorithm_id','algorithm_manifest_sha256'] if c in reference}})
    # Strictly lagged features from observations actually present at generation.
    features=v2.build_features(history)
    bundle=v2.fit(features,date,v2.config()['shadow_candidate'])
    future=slate.copy()
    for col in ['buts','tirs','temps_de_glace','temps_pp']:future[col]=np.nan
    combined=v2.build_features(pd.concat([history,future],ignore_index=True))
    derived=combined[combined.date_match.eq(date)].copy()
    cols=v2.KEYS+['date_match','team_player_match','start_time_utc']+[c for c in ['nom','adversaire_match'] if c in derived]
    out=derived[cols].copy();out['probability']=v2.predict(bundle,derived)
    out=out.merge(reference,on=v2.KEYS,how='left',validate='one_to_one')
    if out.v1_probability.isna().any():raise ValueError('V1 universe does not cover shadow players')
    if not (utc(out.start_time_utc)==utc(out.v1_start_time_utc)).all():raise ValueError('V1 kickoff mismatch')
    generated=pd.Timestamp.now(tz='UTC') if now is None else started
    if (utc(out.start_time_utc)<=generated).any():raise ValueError('Puck drop passed during fit')
    out['generated_at_utc']=generated.isoformat();out['archive_group']='CANONICAL'
    out['scope']='BUT_V2_SHADOW';out['model_version']=v2.config()['version']
    out['status']='UNRESOLVED'
    out['context_source']='NOT_AVAILABLE';out['context_age_hours']=np.nan
    out['information_captured_at_utc']=started.isoformat()
    out['history_last_game_date']=str(pd.to_datetime(history.date_match).max().date())
    out['history_age_days']=(date-pd.to_datetime(history.date_match).max()).days
    out['source_manifest']=json.dumps(provenance,sort_keys=True)
    out['prediction_id']=[hashlib.sha256(f"{generated.isoformat()}|{g}|{p}|{v2.config()['version']}".encode()).hexdigest()
                          for g,p in out[v2.KEYS].itertuples(index=False,name=None)]
    out=out.sort_values(['probability']+v2.KEYS,ascending=[False,True,True])
    out['rank']=np.arange(1,len(out)+1)
    out['v1_rank']=out.sort_values(['v1_probability']+v2.KEYS,ascending=[False,True,True]).assign(_rank=np.arange(1,len(out)+1))['_rank']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    joblib.dump(bundle,output/'model.joblib')
    model_sha=hashlib.sha256((output/'model.joblib').read_bytes()).hexdigest()
    out['model_sha256']=model_sha
    out.to_csv(output/'predictions.csv',index=False)
    # Keep exact feature matrices to reproduce the frozen estimator's predictions.
    derived.to_csv(output/'features.csv.gz',index=False)
    write(output/'manifest.json',dict(version=v2.config(),generated_at_utc=generated.isoformat(),
          input_provenance=provenance,model_sha256=model_sha,training=bundle['metadata'],
          n=len(out),point_or_v1_modified=False,odds_status='ARCHIVED_IF_PROVIDED_ELSE_UNKNOWN',
          code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(v2.__file__),Path(__file__)]},
          predictions_sha256=hashlib.sha256((output/'predictions.csv').read_bytes()).hexdigest()))
    return out


def report(capture_dir,outcomes,output):
    capture_dir=Path(capture_dir);manifest=json.loads((capture_dir/'manifest.json').read_text())
    pred=capture_dir/'predictions.csv'
    if hashlib.sha256(pred.read_bytes()).hexdigest()!=manifest['predictions_sha256']:raise ValueError('Capture changed')
    rows=pd.read_csv(pred)
    if not rows.archive_group.eq('CANONICAL').all() or not rows.scope.eq('BUT_V2_SHADOW').all():raise ValueError('Wrong cohort')
    if (utc(rows.generated_at_utc)>=utc(rows.start_time_utc)).any():raise ValueError('Not pregame')
    if outcomes.duplicated(v2.KEYS).any():raise ValueError('Duplicate official result')
    required=v2.KEYS+['status','buts','source','observed_at_utc','toi_min','team_player_match']
    if set(required)-set(outcomes):raise ValueError('Official outcome provenance/participation required')
    official=outcomes[required].rename(columns={'team_player_match':'official_team'})
    merged=rows.drop(columns='status').merge(official,on=v2.KEYS,how='left',validate='one_to_one')
    merged['settlement_status']='UNRESOLVED';merged['y']=np.nan
    final=merged.status.isin(['OFF','FINAL']) & merged.buts.notna() & pd.to_numeric(merged.toi_min,errors='raise').gt(0)
    if final.any():
        if not merged.loc[final,'official_team'].eq(merged.loc[final,'team_player_match']).all():raise ValueError('Official team mismatch')
        for _,row in merged.loc[final].iterrows():
            expected=f"https://api-web.nhle.com/v1/gamecenter/{int(row.id_match)}/boxscore"
            if row.source!=expected:raise ValueError('Official NHL game source required')
        observed=utc(merged.loc[final,'observed_at_utc'])
        if (observed<utc(merged.loc[final,'start_time_utc'])).any() or (observed>pd.Timestamp.now(tz='UTC')).any():raise ValueError('Invalid outcome observation time')
        merged.loc[final,'y']=v2.goal_labels(merged.loc[final]).to_numpy()
        merged.loc[final,'settlement_status']=np.where(merged.loc[final,'y'].eq(1),'WIN','LOSS')
    from henachel.but_v2_prospective import summarize
    result=summarize(merged)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    merged.to_csv(output/'settlement.csv',index=False);write(output/'report.json',result)
    from henachel.but_v2_prospective import write_daily
    write_daily(merged,result,output)
    return result
