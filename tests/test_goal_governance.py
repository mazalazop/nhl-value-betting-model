import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import pytest
from henachel.goal_governance import ROOT, verify_lock, smoothed_benchmark


def test_registered_algorithm_matches_integrated_code():
    manifest, digest = verify_lock()
    assert manifest['name'] == 'HENACHEL_GOAL_OPERATIONAL_V1'
    assert len(manifest['features']) == 17
    assert len(digest) == 64


def test_algorithm_changes_require_comparability_review(tmp_path):
    manifest, _ = verify_lock()
    paths = list(manifest['file_hashes']) + [
        'docs/goal_operational_prospective/algorithm_v1.json',
        'docs/goal_operational_prospective/protocol.json',
    ]
    for path in paths:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    verify_lock(tmp_path)
    (tmp_path / 'model/henachel/goal.py').write_text('# changed')
    with pytest.raises(ValueError, match='comparability review'):
        verify_lock(tmp_path)


def test_smoothed_benchmark_excludes_calibration_and_future_labels():
    past = pd.DataFrame({'date_match': ['2026-01-01', '2026-01-02', '2026-02-01'], 'buts': [0, 1, 9]})
    future = pd.DataFrame({'season_games_before_match': [0, 10, np.nan], 'season_goal_hits_before_match': [0, 3, np.nan]})
    actual = smoothed_benchmark(past, future, '2026-01-02')
    np.testing.assert_allclose(actual, [.5, .4, np.nan], equal_nan=True)
    past.loc[2, 'buts'] = 0
    pd.testing.assert_series_equal(actual, smoothed_benchmark(past, future, '2026-01-02'))


def test_invalid_benchmark_counts_are_not_imputed():
    past = pd.DataFrame({'date_match': ['2026-01-01'], 'buts': [0]})
    future = pd.DataFrame({'season_games_before_match': [2], 'season_goal_hits_before_match': [3]})
    with pytest.raises(ValueError, match='counts'):
        smoothed_benchmark(past, future, '2026-01-02')
