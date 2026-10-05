import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


def experiment():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_point_depth.py'
    spec = importlib.util.spec_from_file_location('point_depth_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('overlaps_calibration', [False, True])
def test_experiment_restores_production_parameters_after_failed_fit(monkeypatch, overlaps_calibration):
    module = experiment()
    original = module.POINT_PARAMS.copy()
    cutoff = pd.Timestamp('2025-01-02')

    def fit(*args):
        assert module.POINT_PARAMS['max_depth'] == 6
        if not overlaps_calibration:
            raise RuntimeError('fixture training failure')
        return None, None, {'fit_calibration_split': {'calib_end': str(cutoff)}}

    monkeypatch.setattr(module.PROD, 'fit_point_model_and_calibrator', fit)
    with pytest.raises(AssertionError if overlaps_calibration else RuntimeError):
        module.fit_depth(pd.DataFrame(), pd.DataFrame(), 'target', 'date', cutoff, 6)
    assert module.POINT_PARAMS == original


def test_existing_experiment_cannot_be_overwritten(monkeypatch, tmp_path):
    module = experiment()
    marker = tmp_path / 'immutable.txt'
    marker.write_text('original')
    monkeypatch.setattr(sys, 'argv', ['evaluate_point_depth.py', '--features', 'unused.csv', '--output', str(tmp_path)])
    with pytest.raises(FileExistsError, match='overwrite'):
        module.main()
    assert marker.read_text() == 'original'
    assert list(tmp_path.iterdir()) == [marker]
