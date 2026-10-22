import pandas as pd
import pytest
from henachel.but_v2_validation import paired,metrics

def rows():
    return pd.DataFrame([dict(date_match=f'2024-01-{d:02}',id_match=d,id_joueur=p,
        y=p%2,probability=.2+p*.03) for d in range(1,10) for p in range(12)])

def test_paired_identical_has_zero_delta_and_ci():
    a=rows();result=paired(a,a,100)
    assert result['ll']['delta']==0 and result['br']['ci95']==[0,0]
    with pytest.raises(AssertionError):paired(a,a.iloc[1:])
    with pytest.raises(ValueError,match='Duplicate'):paired(pd.concat([a,a]),a)

def test_daily_top_k_not_global_or_percentile():
    result=metrics(rows())
    assert result['top5_n']==45 and result['top10_n']==90
    assert sum(x['count'] for x in result['calibration'])==result['n']
