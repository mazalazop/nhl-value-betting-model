import warnings
import numpy as np
import pandas as pd


def merge_pp(source, pp, min_coverage=.90, policy='error'):
    if policy not in {'error','warn'} or not 0 <= min_coverage <= 1:
        raise ValueError('Invalid PP coverage policy')
    pp = pp.copy()
    pp['temps_pp'] = pd.to_numeric(pp['temps_pp'], errors='raise')
    observed = pp['temps_pp'].dropna()
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise ValueError('Invalid PP duration')
    if pp.duplicated(['id_match','id_joueur']).any(): raise ValueError('Duplicate PP keys')
    out=source.drop(columns=['temps_pp'],errors='ignore').merge(
        pp[['id_match','id_joueur','temps_pp']],on=['id_match','id_joueur'],how='left',validate='one_to_one')
    out['pp_observed']=out.temps_pp.notna()
    coverage=float(out.pp_observed.mean()) if len(out) else 1.
    groups=[c for c in ['season_source','game_type'] if c in out]
    per_group = (out.groupby(groups,dropna=False).pp_observed.mean().to_dict() if groups else {})
    report={'coverage':coverage,'per_group':{str(k):float(v) for k,v in per_group.items()},
            'missing_rows':int((~out.pp_observed).sum()),'min_coverage':min_coverage,'policy':policy}
    if min([coverage]+list(per_group.values())) < min_coverage:
        msg=f'PP coverage below {min_coverage:.1%}: {report}'
        if policy=='error': raise ValueError(msg)
        warnings.warn(msg,UserWarning)
    return out,report
