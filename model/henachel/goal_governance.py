"""Frozen refit-algorithm guard and pregame comparison scores; no model selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def verify_lock(root=ROOT):
    folder=root/'docs/goal_operational_prospective'
    raw=(folder/'algorithm_v1.json').read_bytes()
    manifest=json.loads(raw);protocol=json.loads((folder/'protocol.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=protocol['algorithm_manifest_sha256']:
        raise ValueError('GOAL V1 manifest changed')
    for path,digest in manifest['file_hashes'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'GOAL V1 algorithm changed: {path}; comparability review required')
    return manifest,hashlib.sha256(raw).hexdigest()


def smoothed_benchmark(past,future,fit_end):
    fit=past[pd.to_datetime(past.date_match)<=pd.Timestamp(fit_end)]
    if fit.empty:raise ValueError('Missing benchmark fit partition')
    prevalence=fit.buts.ge(1).mean()
    games=pd.to_numeric(future.season_games_before_match,errors='raise')
    hits=pd.to_numeric(future.season_goal_hits_before_match,errors='raise')
    if ((games<0)|(hits<0)|(hits>games)).any():raise ValueError('Invalid prior-season counts')
    return (hits+10*prevalence)/(games+10)
