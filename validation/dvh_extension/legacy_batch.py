"""Partition the 36-group, 120-case axial design."""
import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import run_benchmark


def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('shard',type=int,nargs='?',default=0)
    p.add_argument('shards',type=int,nargs='?',default=1)
    p.add_argument('--dicompyler-python',type=Path,default=Path(os.environ.get('DICOMPYLER_PYTHON',here/'.venv-dicompyler/bin/python')))
    args=p.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        p.error('Require shards >= 1 and 0 <= shard < shards')
    design=json.loads((here/'DESIGN.json').read_text())
    groups=[s['legacy_group'] for s in design['groups'] if s['rotation']['id']=='axial' and s['volume_cc']>=1]
    assert len(groups)==len(set(groups))==36
    selected=[g for i,g in enumerate(groups) if i%args.shards==args.shard]
    if not selected:p.error('No design groups selected')
    run_benchmark.run(Namespace(results=here/f'results/legacy-shard-{args.shard}',dicompyler_python=args.dicompyler_python,group=selected,limit=None))


if __name__=='__main__':main()
