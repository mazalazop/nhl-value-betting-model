from pathlib import Path
import subprocess
import sys

# Installer les dépendances si besoin
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gspread", "google-auth"])

import gspread
from google.oauth2.service_account import Credentials

repo_root = Path("/content/nhl_value_betting_work/nhl-value-betting-model-main")
creds_path = repo_root / "secrets" / "henachel-service-account.json"
sheet_id = "1cK3tplx1nrq9N32XTvh6Oee7pKYfzsaevQJJlwunfT4"

if not creds_path.exists():
    raise FileNotFoundError(f"JSON introuvable : {creds_path}")

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_file(str(creds_path), scopes=scopes)
gc = gspread.authorize(creds)

sh = gc.open_by_key(sheet_id)

print("Sheet OK :", sh.title)
print("Worksheets :", [ws.title for ws in sh.worksheets()])
