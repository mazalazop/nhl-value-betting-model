"""Unchanged POINT algorithm and strict shared data contracts."""
import numpy as np
import pandas as pd

POINT_PARAMS=dict(loss='log_loss',learning_rate=.05,max_iter=300,max_depth=6,
                  min_samples_leaf=50,l2_regularization=1.,early_stopping=False,random_state=42)


def point_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**POINT_PARAMS)


def binary_labels(values):
    y=pd.to_numeric(pd.Series(values),errors='coerce')
    if y.isna().any() or not y.isin([0,1]).all(): raise ValueError('Labels must be observed binary 0/1')
    return y.astype(int)


def find_target(df,candidates):
    out=df.copy()
    for column in candidates:
        if column in out:
            out[column]=binary_labels(out[column]).to_numpy()
            if 'points' in out:
                points=pd.to_numeric(out.points,errors='coerce')
                if points.isna().any() or not np.isfinite(points).all() or (points<0).any() or (points%1!=0).any():raise ValueError('Invalid points')
                if not out[column].eq(points.ge(1).astype(int)).all():raise ValueError('Label disagrees with points')
            return out,column
    if 'points' in out:
        points=pd.to_numeric(out.points,errors='coerce')
        if points.isna().any() or not np.isfinite(points).all() or (points<0).any() or (points%1!=0).any():raise ValueError('Invalid points')
        out['target_point_1p']=points.ge(1).astype(int)
        return out,'target_point_1p'
    raise ValueError('No POINT target')


def find_date(df,candidates):
    for column in candidates:
        if column in df:
            out=df.copy();out[column]=pd.to_datetime(out[column],errors='raise')
            if out[column].isna().any():raise ValueError('Missing game date')
            return out.sort_values(column).reset_index(drop=True),column
    raise ValueError('No game date')
