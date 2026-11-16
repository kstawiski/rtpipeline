"""Public IBSI-2 sources, retrieved only in task-owned Persistent Chrome tabs.

Run with the supplied radiomics Python -I -B. Downloaded code is data only.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/public_sources_20261106'
BRIDGE = os.environ.get('PERSISTENT_CHROME_BRIDGE', '/umed-projekty/_t3/scripts/persistent-chrome')

def now():
    return datetime.now(timezone.utc).isoformat()

def chrome(name, arguments=None, page=None):
    request = {'name': name, 'arguments': arguments or {}}
    if page:
        request['pageId'] = page
    p = subprocess.run([BRIDGE, json.dumps(request)], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr.rstrip() or p.stdout.rstrip())
    return json.loads(p.stdout)

def evaluate(page, function):
    return chrome('evaluate_script', {'function': function}, page)['result']

def save_manifest(manifest):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

def retrieve(page, url, relative, manifest, commit=None):
    path = DATA / relative
    assert path.resolve().is_relative_to(DATA.resolve())
    old = next((x for x in manifest['files'] if x['path'] == str(path.relative_to(ROOT))), None)
    if path.exists() and old and hashlib.sha256(path.read_bytes()).hexdigest() == old['sha256']:
        return
    start = now()
    try:
        info = evaluate(page, f'''async()=>{{
            const r=await fetch({json.dumps(url)});
            if(!r.ok)throw Error(`HTTP ${{r.status}} ${{r.statusText}}; URL ${{r.url}}`);
            const raw=new Uint8Array(await r.arrayBuffer());
            const stream=new Blob([raw]).stream().pipeThrough(new CompressionStream('gzip'));
            globalThis.ibsiDownload=new Uint8Array(await new Response(stream).arrayBuffer());
            return {{bytes:raw.length,compressed:ibsiDownload.length,content_type:r.headers.get('content-type'),final_url:r.url}};
        }}''')
        chunks = []
        for offset in range(0, info['compressed'], 13500):
            encoded = evaluate(page, f'()=>btoa(String.fromCharCode(...ibsiDownload.subarray({offset},{offset+13500})))')
            chunks.append(base64.b64decode(encoded, validate=True))
        content = gzip.decompress(b''.join(chunks))
        assert len(content) == info['bytes']
        digest = hashlib.sha256(content).hexdigest()
        if old and digest != old['sha256']:
            raise ValueError(f'Pinned SHA256 mismatch: expected {old["sha256"]}, received {digest}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entry = dict(path=str(path.relative_to(ROOT)), url=url, commit=commit,
                     sha256=digest, bytes=len(content), started_utc=start,
                     retrieved_utc=now(), browser=BRIDGE, **{k:v for k,v in info.items() if k not in ['bytes','compressed']})
        manifest['files'] = [x for x in manifest['files'] if x['path'] != entry['path']] + [entry]
        save_manifest(manifest)
        if len(content) > 20 * 1024 * 1024:
            with (ROOT / '.gitignore').open('a') as f:
                f.write('/' + str(path.relative_to(ROOT)) + '\n')
        print(f'{relative}: {len(content)} bytes SHA256 {digest}', flush=True)
    except Exception as exc:
        manifest['failures'].append(dict(url=url, path=str(path.relative_to(ROOT)), utc=now(), error=str(exc), branch='stopped; no substitute reference'))
        save_manifest(manifest)
        print(f'STOPPED {url}: {exc}', flush=True)
    finally:
        try:
            evaluate(page, '()=>{delete globalThis.ibsiDownload;return true;}')
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--page')
    parser.add_argument('--repo', action='append', nargs=3, metavar=('REPO','COMMIT','PATH'))
    parser.add_argument('--url', action='append', nargs=2, metavar=('URL','PATH'))
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    manifest_path = DATA / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(files=[],failures=[],repositories={})
    if args.restore:
        receipt=json.loads((ROOT/'results/run_receipt.json').read_text())
        manifest=dict(files=receipt['source_files'],failures=receipt['retrieval_failures'],repositories=receipt['source_repositories'])
        DATA.mkdir(parents=True,exist_ok=True)
        (DATA/'publisher_access_evidence.json').write_text(json.dumps(receipt['publisher_access'],indent=2)+'\n')
    specs = []
    for repo, commit, path in args.repo or []:
        specs.append((f'https://raw.githubusercontent.com/{repo}/{commit}/{path}', f'{repo}/{path}', commit))
        manifest['repositories'][repo] = commit
    specs += [(url,path,None) for url,path in args.url or []]
    if args.restore:
        specs = [(x['url'],str(Path(x['path']).relative_to('data/public_sources_20261106')),x['commit']) for x in manifest['files']]
    page = args.page or chrome('new_page', {'url':'https://theibsi.github.io/ibsi2/'})['pageId']
    try:
        for url, path, commit in specs:
            retrieve(page,url,path,manifest,commit)
        save_manifest(manifest)
    finally:
        if not args.page:
            chrome('close_page', page=page)

if __name__ == '__main__':
    main()
