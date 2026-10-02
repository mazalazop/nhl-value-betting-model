"""Shared final boxscores. Recent games expire in one hour; older games in 30 days."""
import json
import os
from pathlib import Path
import tempfile
import pandas as pd
from henachel.data import game_state


class BoxscoreCache:
    def __init__(self,directory,full_refresh=False,now=None):
        self.directory=Path(directory);self.full_refresh=full_refresh
        self.now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
        self.hits=0;self.fetches=0

    def get(self,game_id,game_date,fetch):
        path=self.directory/f'{int(game_id)}.json'
        game_day=pd.Timestamp(game_date).tz_localize(None).normalize()
        ttl=pd.Timedelta(hours=1) if (self.now.tz_localize(None)-game_day).days<=7 else pd.Timedelta(days=30)
        if path.exists() and not self.full_refresh:
            try:
                cached=json.loads(path.read_text());age=self.now-pd.Timestamp(cached['fetched_at'])
                payload=cached['payload']
                if pd.Timedelta(0)<=age<=ttl and game_state(payload.get('gameState'))=='final' and int(payload.get('id'))==int(game_id):
                    self.hits+=1;return payload
            except (ValueError,KeyError,TypeError):pass
        self.fetches+=1;payload=fetch(game_id)
        if not isinstance(payload,dict) or game_state(payload.get('gameState'))!='final' or int(payload.get('id',-1))!=int(game_id):
            raise ValueError(f'Boxscore identity or final status unverified for {game_id}')
        self.directory.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',dir=self.directory,delete=False,suffix='.tmp') as out:
            json.dump(dict(fetched_at=self.now.isoformat(),payload=payload),out);temp=out.name
        os.replace(temp,path)
        return payload
