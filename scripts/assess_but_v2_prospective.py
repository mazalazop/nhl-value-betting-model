#!/usr/bin/env python3
"""One explicit, immutable final assessment after the fixed prospective window."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.but_v2_prospective import final_assessment
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--settlement',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(Path('configs/but_v2_prospective.json').read_text())
    result=final_assessment(pd.read_csv(a.settlement),protocol,pd.Timestamp.now(tz='UTC').tz_localize(None))
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
