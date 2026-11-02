"""Read-only, checksum-verified reports from the already frozen durable index."""
import argparse,json,tarfile,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from backup_but_v2_state import authenticated_store

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 index=json.loads(a.index.read_text());store=authenticated_store()
 def read(name):
  item=index['files'][name];response=store.get(f"releases/assets/{item['asset_id']}",headers={'Accept':'application/octet-stream'},stream=True)
  with tarfile.open(fileobj=response.raw,mode='r|gz') as archive:
   for member in archive:
    if member.name==name:
     payload=archive.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==item['sha256'];break
   else:raise ValueError('Missing indexed file')
  response.close();return payload
 pointer=json.loads(read('latest_batch.json'));(a.output/'pointer.json').write_text(json.dumps(pointer))
 for filename in ['report.json','settlement.csv','checkpoint.json']:
  name=pointer['path']+'/cumulative/'+filename
  if name in index['files']:(a.output/filename).write_bytes(read(name))
 (a.output/'provenance.json').write_text(json.dumps(dict(snapshot=index['state_sha256'],workflow_run_id=index['workflow_run_id'],index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
