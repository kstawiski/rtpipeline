"""Retrieve pinned public sources using task-owned Persistent Chrome (main)."""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = os.environ.get('PERSISTENT_CHROME_BRIDGE', 'persistent-chrome')


def chrome(name, arguments=None, page=None):
    request = {'name': name, 'arguments': arguments or {}}
    if page:
        request['pageId'] = page
    p = subprocess.run([BRIDGE, json.dumps(request)], text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f'Persistent Chrome {name} failed: {p.stderr.strip()} {p.stdout[:500]}')
    return json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--page', help='Existing task-owned CLI page, otherwise create and close one')
    parser.add_argument('--extra', nargs=3, action='append', metavar=('REPO', 'REVISION', 'PATH'))
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('--url', nargs=2, metavar=('URL', 'LOCAL_NAME'))
    args = parser.parse_args()
    owned = not args.page
    url_parts = urlsplit(args.url[0]) if args.url else None
    initial_url = f'{url_parts.scheme}://{url_parts.netloc}/' if url_parts else 'https://github.com/theibsi'
    page = args.page or chrome('new_page', {'url': initial_url})['pageId']
    directory = ROOT / 'data' / 'public_sources_20261027'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / args.manifest
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'retrieval_browser': 'Persistent Chrome main (Poland)', 'files': [], 'repositories': {}}
    specs = {
        'theibsi/data_sets': ('master', [
            f'ibsi_1_{phantom}_phantom/{p}'
            for phantom in ['digital', 'ct_radiomics']
            for p in ['README.md', 'LICENSE.md', 'nifti/image/phantom.nii.gz', 'nifti/mask/mask.nii.gz']]),
        'theibsi/ibsi-doc': ('master', ['README.md', 'docs/02_Image_processing.rst', 'docs/03_Image_features.rst', 'docs/05_Reference_data_sets.rst']),
        'theibsi/ibsi_1_data_analysis': ('master', ['README.md', 'data/compare_tables/20191014.xlsx', 'data/20181005_pyradiomics.xlsx', 'naming_tables/20190830.xlsx', 'tolerance/20180108_tolerance_table.csv', 'main_results.R', 'tables.R', 'validation.R']),
        'AIM-Harvard/pyradiomics': ('v3.0.1', ['docs/features.rst', 'docs/faq.rst', 'docs/changes.rst', 'labs/pyradiomics-dcm/resources/featuresDict_IBSIv7.tsv']),
        'theibsi/ibsi_identifier': ('master', ['ibsi_concepts.csv']),
    }
    if args.extra:
        specs = {}
        for repo, rev, path in args.extra:
            if repo not in specs:
                specs[repo] = (rev, [])
            specs[repo][1].append(path)
    if args.url:
        specs = {'official_ibsi_reference': ('unversioned website; SHA256 pins retrieved content', [args.url[1]])}
    try:
        for repo, (revision, files) in specs.items():
            if args.url:
                manifest['repositories'][repo] = {'commit': None, 'requested_revision': revision, 'landing_page': 'https://theibsi.github.io/ibsi1/'}
            elif repo not in manifest['repositories']:
                expression = f'async()=>{{const r=await fetch("https://api.github.com/repos/{repo}/commits/{revision}"); if(!r.ok) throw Error(r.status); const j=await r.json(); return {{commit:j.sha,date:j.commit.committer.date,url:j.html_url,requested_revision:{json.dumps(revision)}}};}}'
                manifest['repositories'][repo] = chrome('evaluate_script', {'function': expression}, page)['result']
            commit = manifest['repositories'][repo]['commit']
            for source_path in files:
                path = directory / repo / source_path
                known = {f['path']: f for m in directory.glob('manifest*.json') for f in json.loads(m.read_text())['files']}
                existing = known.get(str(path.relative_to(ROOT)))
                if path.exists() and existing and existing['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest():
                    if not any(f['path'] == existing['path'] for f in manifest['files']):
                        manifest['files'].append(existing)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                    continue
                url = args.url[0] if args.url else f'https://raw.githubusercontent.com/{repo}/{commit}/{source_path}'
                expression = f'async()=>{{const r=await fetch({json.dumps(url)}); if(!r.ok) throw Error(r.status);globalThis.ibsiDownload=new Uint8Array(await r.arrayBuffer()); return ibsiDownload.length;}}'
                length = chrome('evaluate_script', {'function': expression}, page)['result']
                path.parent.mkdir(parents=True, exist_ok=True)
                # Keep each bridge response under its bounded-output limit.
                partial = path.with_name(path.name + '.partial')
                with partial.open('wb') as stream:
                    for offset in range(0, length, 12000):
                        expression = f'()=>btoa(String.fromCharCode(...ibsiDownload.subarray({offset},{offset+12000})))'
                        chunk = chrome('evaluate_script', {'function': expression}, page)['result']
                        stream.write(base64.b64decode(chunk, validate=True))
                partial.replace(path)
                assert path.stat().st_size == length
                entry = {'path': str(path.relative_to(ROOT)), 'url': url, 'repository': repo, 'commit': commit, 'bytes': length, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
                manifest['files'].append(entry)
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                print(json.dumps({k: entry[k] for k in ['path', 'bytes', 'sha256']}), flush=True)
                chrome('evaluate_script', {'function': '()=>{delete globalThis.ibsiDownload; return true;}'}, page)
    finally:
        if owned:
            chrome('close_page', page=page)


if __name__ == '__main__':
    main()
