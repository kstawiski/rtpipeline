"""Retrieve the pinned IBSI sources selected for this experiment."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import DATA, chrome, evaluate, retrieve, save_manifest

PINS = {
    'data_sets': '6da96021bc91faf4c0cb7fd7fa56a4225d2064a8',
    'ibsi2_manual': 'b6ef5856f97a22af763cea5a424316f667a3de40',
    'ibsi_2_reference_data': '5404579fb3e0d17e8db421f0e82d64ce2432ec03',
    'ibsi_2_data_analysis': 'fde70ca61819799f673c07dca776deea065d2a7d',
}

def selected(repo, paths):
    if repo == 'data_sets':
        return [p for p in paths if p.startswith(('ibsi_2_digital_phantom/','ibsi_2_ct_radiomics_phantom/'))
                and ('/nifti/' in p or p.endswith(('README.md','LICENSE.md')))]
    if repo == 'ibsi_2_reference_data':
        return paths
    if repo == 'ibsi2_manual':
        return [p for p in paths if p in ['IBSI_Filters.tex','refs.bib'] or p.startswith('reference_table/')]
    return [p for p in paths if p in ['README.md','code/README.md','code/pairwise_diff_RMs_vs_CRM.m',
                                    'code/differenceMapNorm.m','results/analysis_with_tolerance.R']]

def main():
    manifest_path = DATA / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(files=[],failures=[],repositories={})
    page = chrome('new_page', {'url':'https://theibsi.github.io/ibsi2/'})['pageId']
    try:
        for repo, commit in PINS.items():
            url = f'https://api.github.com/repos/theibsi/{repo}/git/trees/{commit}?recursive=1'
            # Hash and retain repository metadata too. Every downloaded file is data.
            retrieve(page,url,f'metadata/{repo}_tree.json',manifest,commit)
            tree_path = DATA / f'metadata/{repo}_tree.json'
            if not tree_path.exists():
                continue
            tree = json.loads(tree_path.read_text())
            paths = selected(repo,[x['path'] for x in tree['tree'] if x['type']=='blob'])
            manifest['repositories']['theibsi/'+repo] = commit
            save_manifest(manifest)
            for p in paths:
                retrieve(page,f'https://raw.githubusercontent.com/theibsi/{repo}/{commit}/{p}',
                         f'theibsi/{repo}/{p}',manifest,commit)
    finally:
        chrome('close_page',page=page)

if __name__ == '__main__':
    main()
