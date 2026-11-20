"""Combine disjoint unchanged benchmark batches into the original 120-case table."""
from pathlib import Path
import argparse,csv,json,sys
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parent)
root=parser.parse_args().root.resolve();out=root/'results';dest=out/'legacy';dest.mkdir(parents=True,exist_ok=True)
parts=sorted(out.glob('legacy-shard-*'))
if not parts:raise ValueError('No legacy partitions found')
rows=[];inputs=[];receipts=[];groups=[]
for part in parts:
 rec=json.loads((part/'run_receipt.json').read_text())
 assert rec['status']=='completed' and not rec['errors'],part
 completed={g['group'] for g in rec['groups'] if g['status']=='completed'}
 receipts.append({'path':str(part.relative_to(root)/'run_receipt.json'),'receipt':rec})
 with (part/'metrics_long.csv').open() as h:
  r=csv.DictReader(h);fields=r.fieldnames;rows+=[row for row in r if row['group'] in completed]
 inputs+=[r for r in json.loads((part/'input_sha256.json').read_text()) if r['group'] in completed]
 groups+=[g for g in rec['groups'] if g['group'] in completed]
keys=[(r['case'],r['tool'],r['metric']) for r in rows]
assert len(keys)==len(set(keys))==4440
assert len(groups)==len({g['group'] for g in groups})==36
with (dest/'metrics_long.csv').open('w') as h:
 w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['case'],r['tool'],r['metric'])))
(dest/'input_sha256.json').write_text(json.dumps(inputs,indent=2)+'\n')
(dest/'run_receipt.json').write_text(json.dumps({'status':'completed','groups':groups,'parts':receipts,'case_count':120,'rows':4440,'note':'Disjoint partitions of the 120-case axial design.'},indent=2)+'\n')
print({'cases':120,'groups':36,'rows':len(rows),'input_files':len(inputs)})
