"""Validate GitHub input values before writing its environment/output files."""
import datetime
import os
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


def resolve(env):
    run_date=env.get('INPUT_RUN_DATE') or datetime.datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    if datetime.date.fromisoformat(run_date).isoformat()!=run_date: raise ValueError('Expected YYYY-MM-DD NHL date')
    url=env.get('INPUT_HUB_URL') or 'https://www.unibet.fr/paris-hockey-sur-glace/etats-unis/nhl'
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname!='www.unibet.fr' or any(c in url for c in '\r\n'):
        raise ValueError('Expected an HTTPS Unibet URL without newlines')
    out=dict(RUN_DATE=run_date,HUB_URL=url)
    for key,default in [('HEADLESS','true'),('PUBLISH_TO_SHEET','false')]:
        value=env.get('INPUT_'+key) or default
        if value not in {'true','false'}: raise ValueError(f'Invalid {key}')
        out[key]=value
    for key,default in [('DISCOVERY_MAX_MATCHES','12'),('MIN_ROWS_PER_EVENT','8')]:
        value=env.get('INPUT_'+key) or default
        if not value.isdigit() or not 1<=int(value)<=100: raise ValueError(f'Invalid {key}')
        out[key]=str(int(value))
    return out

if __name__=='__main__':
    values=resolve(os.environ)
    with Path(os.environ['GITHUB_ENV']).open('a') as out:
        for key,value in values.items():out.write(f'{key}={value}\n')
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as out:
        for key in ['RUN_DATE','PUBLISH_TO_SHEET']:out.write(f'{key.lower()}={values[key]}\n')
