import numpy as np
import pandas as pd
import pytest
from conftest import load_script


def sample():
    p = np.linspace(.03, .97, 100)
    y = (np.arange(100) % 3 == 0).astype(int)
    dates = pd.date_range('2024-01-01', periods=100)
    return p, y, dates


def test_calibration_entry_points_share_implementation():
    c03 = load_script('03_calibrate_point_model')
    c05 = load_script('05_predict_upcoming_games')
    assert c03.select_calibrator is c05.select_calibrator


@pytest.mark.parametrize('method', ['raw', 'sigmoid', 'isotonic', 'auto'])
def test_reproducible_calibration(method):
    from henachel.calibration import select_calibrator
    p,y,dates = sample()
    a,info = select_calibrator(p,y,dates, policy=method)
    b,_ = select_calibrator(p,y,dates, policy=method)
    np.testing.assert_array_equal(a.predict(p), b.predict(p))
    assert np.isfinite(a.predict([0,1])).all()
    assert info['fit_end'] < info['eval_start']


@pytest.mark.parametrize('bad', [[np.nan,.5], [np.inf,.5], [-.1,.5], [1.1,.5]])
def test_invalid_probabilities_rejected(bad):
    from henachel.calibration import validate_probabilities
    with pytest.raises(ValueError):
        validate_probabilities(bad)


def test_empty_and_single_class_are_explicit():
    from henachel.calibration import select_calibrator
    with pytest.raises(ValueError, match='empty'):
        select_calibrator([],[],[])
    c,info = select_calibrator([.2,.3,.4], [0,0,0], pd.date_range('2024-01-01',periods=3))
    assert c.method == 'raw'
    assert info['reason'] == 'single_class'


def test_sigmoid_is_logistic_of_logit():
    from henachel.calibration import select_calibrator, logit_clip
    p,y,dates = sample()
    c,_ = select_calibrator(p,y,dates,policy='sigmoid')
    np.testing.assert_allclose(c.predict(p), c.estimator.predict_proba(logit_clip(p).reshape(-1,1))[:,1])


def test_test_labels_do_not_enter_calibration():
    from henachel.calibration import select_calibrator
    p,y,dates = sample()
    a,_ = select_calibrator(p[:80],y[:80],dates[:80])
    y[80:] = 1-y[80:]
    b,_ = select_calibrator(p[:80],y[:80],dates[:80])
    np.testing.assert_array_equal(a.predict(p[80:]),b.predict(p[80:]))


def test_overlapping_evaluation_window_rejected():
    from henachel.calibration import validate_evaluation_window
    with pytest.raises(ValueError):validate_evaluation_window(['2026-01-02'],['2026-01-01','2026-01-02'])
    validate_evaluation_window(['2026-01-01'],['2026-01-02'])
