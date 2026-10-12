"""Export small, reproducible structural results; leave OOF/models outside Git."""
import argparse
import json
from pathlib import Path
from research_structural import digest, write


def summarize(directory, destination):
    report=json.loads((directory/'summary.json').read_text())
    result={key:report[key] for key in ['target','candidate','production_approved','independent_confirmation','comparisons','diagnostic_contrasts','manifest']}
    result['reports']={}
    for name,row in report['reports'].items():
        summary={key:row[key] for key in ['metrics','ece','calibration','rows','dates','players','positions'] if key in row}
        if 'features' in row:summary['feature_count']=len(row['features'])
        if 'folds' in row:
            summary['folds']=[{key:fold[key] for key in ['metrics','ece','kept','fit_end','calibration_end','calibration','prediction_sha256','frozen_model'] if key in fold} for fold in row['folds']]
        result['reports'][name]=summary
    result['artifacts']={str(path):digest(path) for path in sorted(directory.iterdir()) if path.is_file()}
    write(destination,result)
    print(json.dumps(dict(target=result['target'],candidate=result['candidate'],metrics={name:{**r['metrics'],'ece':r['ece']} for name,r in result['reports'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summarize(a.directory,a.output)
