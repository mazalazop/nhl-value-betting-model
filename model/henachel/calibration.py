"""Temporal calibrator selection shared by evaluation and production.

Policy auto: fit on the first half of calibration dates, choose using log loss
then Brier on later dates, refit the winner on the complete calibration window.
No test labels are accepted by this API. Small/single-class windows retain raw
probabilities with an explicit diagnostic; an empty window is an error.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

EPSILON = 1e-6


def validate_probabilities(values):
    p = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and between 0 and 1')
    return p


def clip_proba(values):
    return np.clip(validate_probabilities(values), EPSILON, 1-EPSILON)


def logit_clip(values):
    p = clip_proba(values)
    return np.log(p / (1-p))


def fit_sigmoid_calibrator(proba_fit, y_fit):
    return LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000, random_state=42).fit(
        logit_clip(proba_fit).reshape(-1,1), y_fit)


def predict_sigmoid_calibrator(calibrator, proba):
    return clip_proba(calibrator.predict_proba(logit_clip(proba).reshape(-1,1))[:,1])


def fit_isotonic_calibrator(proba_fit, y_fit):
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(validate_probabilities(proba_fit), y_fit)


def predict_isotonic_calibrator(calibrator, proba):
    return clip_proba(calibrator.predict(validate_probabilities(proba)))


@dataclass
class Calibration:
    method: str
    estimator: object = None

    def predict(self, values):
        p = validate_probabilities(values)
        if not len(p):
            return p
        if self.method == 'raw':
            return clip_proba(p)
        if self.method == 'sigmoid':
            return predict_sigmoid_calibrator(self.estimator, p)
        if self.method == 'isotonic':
            return predict_isotonic_calibrator(self.estimator, p)
        raise ValueError(f'Unknown calibrator: {self.method}')


def fit_method(method, p, y):
    if method == 'raw':
        return Calibration('raw')
    if method == 'sigmoid':
        return Calibration(method, fit_sigmoid_calibrator(p,y))
    if method == 'isotonic':
        return Calibration(method, fit_isotonic_calibrator(p,y))
    raise ValueError(f'Unknown calibration policy: {method}')


def select_calibrator(proba, labels, dates, policy='auto'):
    if policy not in {'auto','raw','sigmoid','isotonic'}:
        raise ValueError('Invalid calibration policy')
    p = validate_probabilities(proba)
    y = np.asarray(labels, dtype=float).reshape(-1)
    d = pd.DatetimeIndex(pd.to_datetime(dates, errors='raise')).normalize()
    if not len(p):
        raise ValueError('Calibration window is empty')
    if len(p) != len(y) or len(p) != len(d) or d.isna().any():
        raise ValueError('Invalid calibration dates/lengths')
    if not np.isin(y,[0,1]).all():
        raise ValueError('Calibration labels must be binary')
    unique = sorted(d.unique())
    info = {'policy':policy, 'rows':len(p), 'selection':[], 'fit_start':str(min(d)), 'calib_end':str(max(d))}
    if len(np.unique(y)) < 2 or len(unique) < 3:
        info.update(method='raw', reason='single_class' if len(np.unique(y)) < 2 else 'insufficient_dates')
        return Calibration('raw'),info
    cut = unique[len(unique)//2]
    fit = d < cut
    info.update(fit_end=str(max(d[fit])), eval_start=str(min(d[~fit])))
    if len(np.unique(y[fit])) < 2:
        info.update(method='raw',reason='single_class_fit')
        return Calibration('raw'),info
    methods = ['raw','sigmoid','isotonic'] if policy == 'auto' else [policy]
    scores=[]
    for method in methods:
        c=fit_method(method,p[fit],y[fit])
        q=c.predict(p[~fit])
        scores.append(dict(method=method, logloss=float(log_loss(y[~fit],q,labels=[0,1])),
                           brier=float(brier_score_loss(y[~fit],q))))
    scores.sort(key=lambda r:(r['logloss'],r['brier'],methods.index(r['method'])))
    selected=scores[0]['method']
    info.update(method=selected,reason='selected_on_temporal_calibration_eval',selection=scores)
    return fit_method(selected,p,y),info
