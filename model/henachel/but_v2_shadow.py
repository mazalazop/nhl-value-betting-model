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
    refs=utc(reference.generated_at_utc)
    refstarts=utc(reference.start_time_utc)
    if (refs>=refstarts).any() or (refs>started).any():raise ValueError('V1 is not pregame')
    p=pd.to_numeric(reference.probability_goal,errors='raise')
    if not p.between(0,1).all():raise ValueError('Invalid V1 probability')
    pricecols=['odds_decimal','odds_captured_at_utc','odds_source','odds_market']
    if set(pricecols)<=set(reference):
        priced=reference.odds_decimal.notna()
        odds=pd.to_numeric(reference.loc[priced,'odds_decimal'],errors='raise')
        if not np.isfinite(odds).all() or not odds.gt(1).all():raise ValueError('Invalid odds')
        if not reference.loc[priced,'odds_market'].eq('GOAL_1_PLUS').all():raise ValueError('Wrong odds market')
        quotes=utc(reference.loc[priced,'odds_captured_at_utc'])
        if (quotes>started).any() or (quotes>=refstarts[priced]).any():raise ValueError('Postgame odds forbidden')
        if reference.loc[priced,'odds_source'].isna().any():raise ValueError('Odds source required')
    else:
        if 'odds_decimal' in reference and reference.odds_decimal.notna().any():raise ValueError('Incomplete odds provenance')
        reference=reference.copy()
        for col in pricecols:reference[col]=np.nan
    keep=v2.KEYS+['probability_goal','generated_at_utc','start_time_utc']+pricecols
    reference=reference[keep].rename(columns={'probability_goal':'v1_probability','generated_at_utc':'v1_generated_at_utc',
                                            'start_time_utc':'v1_start_time_utc'})
    # Strictly lagged features from observations actually present at generation.
    features=v2.build_features(history)
    bundle=v2.fit(features,date,v2.config()['shadow_candidate'])
    future=slate.copy()
    for col in ['buts','tirs','temps_de_glace','temps_pp']:future[col]=np.nan
    combined=v2.build_features(pd.concat([history,future],ignore_index=True))
    derived=combined[combined.date_match.eq(date)].copy()
    cols=v2.KEYS+['date_match','team_player_match','start_time_utc']
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
    out['source_manifest']=json.dumps(provenance,sort_keys=True)
    out['prediction_id']=[hashlib.sha256(f"{generated.isoformat()}|{g}|{p}|{v2.config()['version']}".encode()).hexdigest()
                          for g,p in out[v2.KEYS].itertuples(index=False,name=None)]
    out=out.sort_values(['probability']+v2.KEYS,ascending=[False,True,True])
    out['rank']=np.arange(1,len(out)+1)
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
    required=v2.KEYS+['status','buts','source','observed_at_utc']
    if set(required)-set(outcomes):raise ValueError('Official outcome provenance required')
    merged=rows.drop(columns='status').merge(outcomes[required],on=v2.KEYS,how='left',validate='one_to_one')
    merged['settlement_status']='UNRESOLVED'
    final=merged.status.isin(['OFF','FINAL']) & merged.buts.notna()
    if final.any():
        if merged.loc[final,'source'].isna().any():raise ValueError('Missing official source')
        sources=merged.loc[final,'source'].astype(str)
        if not sources.str.startswith('https://api-web.nhle.com/').all():raise ValueError('Official NHL source required')
        observed=utc(merged.loc[final,'observed_at_utc'])
        if (observed<utc(merged.loc[final,'start_time_utc'])).any():raise ValueError('Outcome predates game')
        merged.loc[final,'y']=v2.goal_labels(merged.loc[final]).to_numpy()
        merged.loc[final,'settlement_status']=np.where(merged.loc[final,'y'].eq(1),'WIN','LOSS')
    valid=merged.loc[final].copy()
    result=dict(n=len(rows),settled=len(valid),unresolved=int((~final).sum()),roi=None,
                note='Shadow CANONICAL only; no real bets, no postgame reconstructed predictions')
    if len(valid):
        reference=valid.copy();reference['probability']=reference.v1_probability
        result.update(v1=metrics(reference),v2=metrics(valid),paired=paired(reference,valid))
    priced=valid[valid.odds_decimal.notna()].copy()
    if len(priced):
        priced['implied_probability']=1/priced.odds_decimal
        priced['v2_edge']=priced.probability-priced.implied_probability
        priced['unit_return']=priced.y*priced.odds_decimal-1
        daily=priced.groupby('date_match').agg(total=('unit_return','sum'),n=('y','size'))
        rng=np.random.default_rng(v2.config()['seed'])
        n=len(daily);starts=rng.integers(0,n,size=(2000,(n+6)//7))
        sample=((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
        draws=daily.total.to_numpy()[sample].sum(axis=1)/daily.n.to_numpy()[sample].sum(axis=1)
        result['roi']=dict(n=len(priced),dates=n,simulation_only=True,mean=float(priced.unit_return.mean()),
                          ci95=np.quantile(draws,[.025,.975]).tolist(),
                          warning='All priced shadow rows, not placed bets; low date counts give unreliable uncertainty',
                          mean_v2_edge=float(priced.v2_edge.mean()),implied_is_not_devigged=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    merged.to_csv(output/'settlement.csv',index=False);write(output/'report.json',result)
    return result
