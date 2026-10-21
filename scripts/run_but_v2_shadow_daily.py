#!/usr/bin/env python3
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.but_v2_daily import daily,seal
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:daily(a.root/'upstream',a.root/'state',json.loads((a.root/'upstream_run.json').read_text()),requests.Session(),Path('configs/but_v2_prospective.json'))
    finally:
        if (a.root/'state').exists():seal(a.root/'state')
