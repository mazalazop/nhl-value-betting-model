"""Prospective summaries: freeze ranks before observing missing outcomes."""
import numpy as np
import pandas as pd
from henachel.but_v2_validation import metrics,paired


def selection_stats(rows,rank):
    selected=rows[rows[rank]<=10]
    settled=selected[selected.settlement_status.isin(['WIN','LOSS'])]
    priced=settled[settled.odds_decimal.notna()].copy()
    return dict(selected=len(selected),settled=len(settled),unresolved=len(selected)-len(settled),
                wins=int(settled.y.sum()),hit_rate=float(settled.y.mean()) if len(settled) else None,
                priced_n=len(priced),roi_simulated=float((priced.y*priced.odds_decimal-1).mean()) if len(priced) else None)


def summarize(rows):
    if rows.duplicated(['id_match','id_joueur']).any():raise ValueError('Duplicate prospective player/game')
    valid=rows[rows.settlement_status.isin(['WIN','LOSS'])].copy()
    result=dict(n=len(rows),settled=len(valid),unresolved=len(rows)-len(valid),
                coverage=len(valid)/len(rows) if len(rows) else 0,
                note='CANONICAL shadow only. Top10 frozen before outcomes; no replacement of unresolved players.',roi=None)
    result['top10']={name:selection_stats(rows,rank) for name,rank in [('v1','v1_rank'),('v2','rank')]}
    if len(valid):
        ref=valid.copy();ref['probability']=ref.v1_probability
        result.update(v1=metrics(ref),v2=metrics(valid),paired=paired(ref,valid))
        # The shared retrospective helper ranks survivors: suppress those fields.
        for name in ['v1','v2']:
            for k in [5,10]:
                result[name].pop(f'top{k}_n',None);result[name].pop(f'top{k}_hit_rate',None)
        if valid.date_match.nunique()<14:
            for value in result['paired'].values():value['ci95']=value['family_adjusted_ci']=None
            result['uncertainty']='INSUFFICIENT_DATES_FOR_BLOCK_INTERVAL'
    priced=valid[valid.odds_decimal.notna()]
    if len(priced):
        result['roi']=dict(n=len(priced),simulation_only=True,mean=float((priced.y*priced.odds_decimal-1).mean()),
                          policy='all priced rows; identical universe for both models, not a selection comparison',ci95=None)
    return result


def write_daily(rows,result,output):
    top=[]
    for name,rank in [('V1','v1_rank'),('V2','rank')]:
        frame=rows[rows[rank]<=10].sort_values(['date_match',rank]).copy();frame['model']=name
        frame['selected_rank']=frame[rank];frame['selected_probability']=frame.v1_probability if name=='V1' else frame.probability
        top.append(frame)
    pd.concat(top).to_csv(output/'top10.csv',index=False)
    lines=['# BUT shadow — bilan prospectif','',f"Soldés : {result['settled']}/{result['n']}; non résolus : {result['unresolved']}.",'',
           '| Modèle | Top10 soldé/sélectionné | Réussite | Brier | Log-loss | ROI fictif Top10 (n coté) |',
           '|---|---:|---:|---:|---:|---:|']
    for name in ['v1','v2']:
        t=result['top10'][name];m=result.get(name,{})
        def fmt(x):return 'inconnu' if x is None else f'{x:.6f}'
        lines.append(f"| {name} | {t['settled']}/{t['selected']} | {fmt(t['hit_rate'])} | {fmt(m.get('brier'))} | {fmt(m.get('log_loss'))} | {fmt(t['roi_simulated'])} ({t['priced_n']}) |")
    lines+=['','Les joueurs non résolus du Top10 ne sont pas remplacés. Calibration par intervalles dans report.json.',
            'Aucun pari réel. Un bilan quotidien ne permet aucune promotion ni intervalle fiable à lui seul.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def final_assessment(rows, protocol, as_of):
    """Single fixed-window assessment; review gates never grant production approval."""
    first=pd.to_datetime(rows.date_match).min();end=first+pd.Timedelta(days=protocol['final_calendar_day']-1)
    if pd.Timestamp(as_of).normalize()<=end:raise ValueError('Fixed prospective window not complete')
    cohort=rows[pd.to_datetime(rows.date_match)<=end].copy()
    result=summarize(cohort);valid=cohort[cohort.settlement_status.isin(['WIN','LOSS'])].copy()
    if valid.empty:return dict(status='INSUFFICIENT_DATA',promotion_authorized=False)
    ref=valid.copy();ref['probability']=ref.v1_probability
    comparison=paired(ref,valid,replicates=10000,seed=20261020,family_size=2)
    def ece(model):
        bins=result[model]['calibration']
        return sum(b['count']*abs(b['mean_probability']-b['observed_rate']) for b in bins if b['count'])/len(valid)
    floors=(len(valid)>=protocol['min_rows'] and valid.date_match.nunique()>=protocol['min_dates'] and valid.y.sum()>=protocol['min_positive_outcomes']
            and result['coverage']>=protocol['min_settlement_coverage'])
    scores=(comparison['ll']['delta']<=-protocol['minimum_log_loss_gain'] and comparison['br']['delta']<=-protocol['minimum_brier_gain']
            and comparison['ll']['family_adjusted_ci'][1]<0 and comparison['br']['family_adjusted_ci'][1]<0)
    calibration=ece('v2')<=.03 and ece('v2')-ece('v1')<=.005
    return dict(primary_end=str(end.date()),sample_floors_met=bool(floors),score_gate=bool(scores),calibration_gate=bool(calibration),
                comparison=comparison,ece_v1=ece('v1'),ece_v2=ece('v2'),promotion_authorized=False,
                remaining_gates=['external eligible-day coverage audit >=95%', 'segment robustness review', 'explicit human approval'])
