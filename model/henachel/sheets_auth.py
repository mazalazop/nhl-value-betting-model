"""Environment-only service-account authentication; never writes credentials to disk."""
import json
import os
import gspread
from google.oauth2.service_account import Credentials


def authorize_environment(*,readonly=False):
    raw=os.environ.pop('GOOGLE_CREDENTIALS',None)
    if not raw:raise ValueError('Missing GOOGLE_CREDENTIALS environment variable')
    scope='https://www.googleapis.com/auth/spreadsheets'+('.readonly' if readonly else '')
    try:
        info=json.loads(raw)
        credentials=Credentials.from_service_account_info(info,scopes=[scope])
    except Exception:
        # Parser/auth errors must never echo any secret input.
        raise ValueError('Invalid service account configuration') from None
    return gspread.authorize(credentials)
