"""Read-only extraction from a verified durable backup; never writes to a registry."""
import argparse, hashlib, json, sys, tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from backup_but_v2_state import authenticated_store

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 if a.index and a.receipt:raise ValueError('Choose either index or receipt')
 store=authenticated_store()
 if a.receipt:
  receipt=json.loads(a.receipt.read_text());index=dict(files={receipt['source']:receipt['source_entry']},state_sha256=receipt['snapshot'],workflow_run_id=receipt['workflow_run_id'])
 else:index=json.loads(a.index.read_text()) if a.index else store.latest()
 files=[(n,v) for n,v in index['files'].items() if n.startswith('training_sources/') and n.endswith('.csv.gz')]
 name,item=max(files,key=lambda x:x[1]['asset_id'])
 response=store.get(f"releases/assets/{item['asset_id']}",headers={'Accept':'application/octet-stream'},stream=True)
 with tarfile.open(fileobj=response.raw,mode='r|gz') as archive:
  for member in archive:
   if member.name==name:
    payload=archive.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==item['sha256']
    (a.output/'history.csv.gz').write_bytes(payload);break
  else:raise ValueError('Missing source in verified bundle')
 response.close()
 (a.output/'durable-index.json').write_text(json.dumps(index,indent=2))
 (a.output/'receipt.json').write_text(json.dumps(dict(snapshot=index['state_sha256'],source=name,source_entry=item,workflow_run_id=index['workflow_run_id'],selection='latest training bundle asset id; dates validated in inventory'),indent=2))
 print(name,index['state_sha256'])
if __name__=='__main__':main()
