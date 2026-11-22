"""Download the pinned, original first-half decisions for candidate/bbox provenance."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/derived/ball_fp_reduction_20261116/sources/first_half_decisions.jsonl.gz'
OID='3bf2cc41f2bffa96180767963bc6bc953e4cd1902d1a7243d5ce2aa0d18375ee'
SIZE=43001808


def main():
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
        assert DEST.stat().st_size==SIZE and hashlib.sha256(DEST.read_bytes()).hexdigest()==OID
        print('Pinned reference already verified');return
    headers={'Accept':'application/vnd.git-lfs+json','Content-Type':'application/vnd.git-lfs+json'}
    data=json.dumps(dict(operation='download',transfers=['basic'],objects=[dict(oid=OID,size=SIZE)])).encode()
    url='https://github.com/jsihun1006/SNP.git/info/lfs/objects/batch'
    try:response=urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers),timeout=60)
    except urllib.error.HTTPError as error:
        if error.code not in (401,403):raise
        credential=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\npath=jsihun1006/SNP.git\n\n',text=True,capture_output=True,check=True)
        fields=dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
        headers['Authorization']='Basic '+base64.b64encode((fields['username']+':'+fields['password']).encode()).decode()
        response=urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers),timeout=60)
    obj=json.load(response)['objects'][0]
    if 'error' in obj:raise RuntimeError(obj['error'])
    action=obj['actions']['download'];request=urllib.request.Request(action['href'],headers=action.get('header',{}))
    temp=DEST.with_suffix('.partial');digest=hashlib.sha256();size=0
    with urllib.request.urlopen(request,timeout=120) as source,temp.open('wb') as output:
        while chunk:=source.read(4*1024*1024):output.write(chunk);digest.update(chunk);size+=len(chunk)
    assert size==SIZE and digest.hexdigest()==OID
    temp.replace(DEST);print(f'Original decisions verified: {size} bytes')


if __name__=='__main__':main()
