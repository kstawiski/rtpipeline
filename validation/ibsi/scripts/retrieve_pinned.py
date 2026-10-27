"""Restore the exact public source files named in an existing run receipt."""
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from retrieve import chrome


def main():
    receipt = json.loads((ROOT / 'results/run_receipt.json').read_text())
    page = chrome('new_page', {'url': 'https://github.com/theibsi'})['pageId']
    manifest = {'retrieval_browser': 'Persistent Chrome main (Poland)',
                'repositories': receipt['source_repositories'], 'files': []}
    partial = None
    try:
        for item in receipt['source_files']:
            path = ROOT / item['path']
            assert path.is_relative_to(ROOT / 'data')
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']:
                manifest['files'].append(item)
                continue
            host = urlsplit(item['url'])
            if host.netloc != 'raw.githubusercontent.com':
                chrome('navigate_page', {'url': f'{host.scheme}://{host.netloc}/'}, page)
            expression = f'async()=>{{const r=await fetch({json.dumps(item["url"])}); if(!r.ok)throw Error(r.status);globalThis.ibsiDownload=new Uint8Array(await r.arrayBuffer());return ibsiDownload.length;}}'
            size = chrome('evaluate_script', {'function': expression}, page)['result']
            assert size == item['bytes'], (path, size, item['bytes'])
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.partial')
            with partial.open('wb') as f:
                for offset in range(0, size, 12000):
                    expression = f'()=>btoa(String.fromCharCode(...ibsiDownload.subarray({offset},{offset+12000})))'
                    f.write(base64.b64decode(chrome('evaluate_script', {'function': expression}, page)['result'], validate=True))
            assert hashlib.sha256(partial.read_bytes()).hexdigest() == item['sha256'], path
            partial.replace(path)
            partial = None
            entry = dict(item, retrieved_utc=datetime.now(timezone.utc).isoformat())
            manifest['files'].append(entry)
            chrome('evaluate_script', {'function': '()=>{delete globalThis.ibsiDownload;return true;}'}, page)
            print(f'Restored {item["path"]}', flush=True)
        (ROOT / 'data/public_sources_20261027/manifest_pinned.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Verified {len(manifest["files"])} pinned public source files.')
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
        chrome('close_page', page=page)


if __name__ == '__main__':
    main()
