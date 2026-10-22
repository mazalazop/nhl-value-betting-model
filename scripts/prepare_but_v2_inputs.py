#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.but_v2_inputs import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();slate,ref=prepare(a.root,a.output);print(f'Prepared {len(slate)} identical V1/V2 identities; no predictions emitted')
