"""Canonical local ledger. Locks serialize reruns; atomic replacement preserves the old file on failure."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
import pandas as pd
from henachel.matching import COLUMNS


@contextmanager
def ledger_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


def atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as out:
            temporary=out.name;frame.to_csv(out,index=False);out.flush();os.fsync(out.fileno())
        os.replace(temporary,path)
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


def validate_ledger(frame):
    required=['bet_id','id_match','id_joueur','market','stat','threshold','bookmaker']
    if not set(required).issubset(frame): raise ValueError('Canonical ledger lacks identity columns; explicit migration required')
    if frame[required].isna().any().any() or frame.bet_id.astype(str).str.strip().eq('').any(): raise ValueError('Missing bet identity')
    if frame.bet_id.duplicated().any(): raise ValueError('Duplicate bet_id')
    keys=['id_match','id_joueur','bookmaker','market','stat','threshold']
    if frame.duplicated(keys).any(): raise ValueError('Duplicate natural bet identity; migration required')


def read_ledger(path):
    path=Path(path)
    frame=pd.read_csv(path,low_memory=False) if path.exists() else pd.DataFrame(columns=COLUMNS)
    validate_ledger(frame)
    return frame


def append_ledger(path,daily):
    validate_ledger(daily)
    with ledger_lock(path):
        old=read_ledger(path); old_ids=old.set_index('bet_id'); new_ids=daily.set_index('bet_id')
        common=old_ids.index.intersection(new_ids.index)
        for col in ['id_match','id_joueur','market','stat','threshold','bookmaker']:
            a,b=old_ids.loc[common,col],new_ids.loc[common,col]
            if col in {'id_match','id_joueur','threshold'}:
                a,b=pd.to_numeric(a),pd.to_numeric(b)
            if not a.eq(b).all(): raise ValueError(f'Conflicting bet identity: {col}')
        # Freeze original odds, probabilities and settlement; reruns only append new identities.
        additional=daily[~daily.bet_id.isin(old.bet_id)]
        combined=pd.concat([old,additional],ignore_index=True) if len(old) else additional.copy()
        for col in COLUMNS:
            if col not in combined: combined[col]=pd.Series(index=combined.index,dtype=object)
        validate_ledger(combined);atomic_csv(combined,path)
        return dict(rows_before=len(old),rows_added=len(additional),rows_after=len(combined))
