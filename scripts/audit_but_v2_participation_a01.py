"""Explicit exploratory audit only. No workflow hook, no study activation option."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import pandas as pd
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
from henachel.but_v2_participation_a01 import archive_evidence, audit, summarize, validate_preparation_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--original-report', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, help='JSON mapping game IDs to official envelopes; otherwise collect now')
    parser.add_argument('--previous-evidence', type=Path)
    parser.add_argument('--as-of', help='Required for offline replay, forbidden with collection')
    parser.add_argument('--output', type=Path, required=True, help='New separate directory, must not exist')
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1]/'configs/but_v2_a01_proposed.json'
    validate_preparation_config(json.loads(config_path.read_text()))
    manifest_path = args.predictions.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(args.predictions.read_bytes()).hexdigest() != manifest['predictions_sha256']:
        parser.error('Original capture checksum mismatch')
    if bool(args.evidence) != bool(args.as_of):
        parser.error('Offline evidence requires --as-of; online collection uses the real clock')
    # Never place output inside an existing scientific registry or beside predictions.
    destination = args.output.resolve()
    for ancestor in [destination, *destination.parents]:
        if (ancestor/'protocol_lock.json').exists() or (ancestor/'archive_manifest.json').exists():
            parser.error('Output must be outside the original registry')
    if destination == args.predictions.resolve().parent or args.predictions.resolve().is_relative_to(destination):
        parser.error('Output must be separate from input')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = pd.read_csv(args.predictions).to_dict('records')
    original = json.loads(args.original_report.read_text())
    errors = []
    if args.evidence:
        evidence = json.loads(args.evidence.read_text())
        as_of = args.as_of
    else:
        evidence = {}
        for gid in sorted({int(r['id_match']) for r in rows}):
            evidence[str(gid)] = {}
            for endpoint in ['boxscore', 'play-by-play']:
                url = f'https://api-web.nhle.com/v1/gamecenter/{gid}/{endpoint}'
                try:
                    response = requests.get(url, timeout=35)
                    response.raise_for_status()
                    evidence[str(gid)][endpoint] = dict(source=url, document=response.json(),
                        observed_at_utc=datetime.now(timezone.utc).isoformat())
                except (requests.RequestException, ValueError) as exc:
                    errors.append(dict(game_id=gid, source=url, error=str(exc),
                                       observed_at_utc=datetime.now(timezone.utc).isoformat()))
        as_of = datetime.now(timezone.utc).isoformat()
    previous = json.loads(args.previous_evidence.read_text()) if args.previous_evidence else None
    receipt = {}
    for label, snapshot in [('current', evidence), ('previous', previous or {})]:
        receipt[label] = {gid: {endpoint: archive_evidence(args.output/'evidence', envelope)
                        for endpoint, envelope in values.items()} for gid, values in snapshot.items()}
    classified = audit(rows, evidence, as_of, previous)
    metrics = summarize(classified, original['settled'], original['n'])
    # Explicitly check the unchanged report's numerator/denominator consistency.
    if original['coverage'] != metrics['original_coverage']:
        raise ValueError('Original coverage disagrees with supplied counts')
    for name, value in [('classifications', classified), ('metrics', metrics), ('evidence_index', receipt),
                        ('collection_errors', errors), ('snapshot', evidence),
                        ('input_provenance', dict(predictions_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                          original_report_sha256=hashlib.sha256(args.original_report.read_bytes()).hexdigest(),
                          original_report=original, as_of_utc=as_of, config=json.loads(config_path.read_text())))]:
        (args.output/(name+'.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
