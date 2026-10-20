#!/usr/bin/env python3
"""Capture a future shadow slate or report independently supplied NHL outcomes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.but_v2_shadow import capture,report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    c=s.add_parser('capture')
    for name in ['history','slate','v1','output']:c.add_argument('--'+name,type=Path,required=True)
    r=s.add_parser('report')
    for name in ['capture','outcomes','output']:r.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.mode=='capture':
        paths={'history':a.history,'slate':a.slate,'v1':a.v1}
        provenance={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in paths.items()}
        capture(pd.read_csv(a.history,low_memory=False),pd.read_csv(a.slate),pd.read_csv(a.v1),a.output,provenance)
    else:report(a.capture,pd.read_csv(a.outcomes),a.output)
