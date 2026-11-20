"""Exact decimal and serialized-input comparison with a supplied reference run."""
import argparse,csv,json,sys
from pathlib import Path
from decimal import Decimal
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,required=True,help='Reference directory containing metrics_long.csv and input_sha256.json')
p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results')
args=p.parse_args();out=args.results.resolve();old=args.reference.resolve()
def rows(p):return list(csv.DictReader(p.open()))
a=rows(old/'metrics_long.csv');b=rows(out/'legacy/metrics_long.csv')
key=lambda r:(r['case'],r['tool'],r['metric'])
ma={key(r):r for r in a};mb={key(r):r for r in b}
fields=['value','analytic','abs_err','rel_err','signed_err','status','rel_err_status','shape','volume_cc','phase','dose_mm','ct_mm','field']
num=set(['value','analytic','abs_err','rel_err','signed_err','volume_cc','dose_mm','ct_mm'])
diffs=[]
for k in sorted(set(ma)|set(mb)):
 if k not in ma or k not in mb:diffs.append({'case':k[0],'tool':k[1],'metric':k[2],'field':'row_presence','old':k in ma,'new':k in mb});continue
 for f in fields:
  av,bv=ma[k][f],mb[k][f]
  equal=Decimal(av)==Decimal(bv) if f in num and av and bv else av==bv
  if not equal:diffs.append(dict(case=k[0],tool=k[1],metric=k[2],field=f,old=av,new=bv))
p=out/'legacy_comparison_differences.csv'
with p.open('w') as h:
 w=csv.DictWriter(h,fieldnames=['case','tool','metric','field','old','new']);w.writeheader();w.writerows(diffs)
input_old=json.loads((old/'input_sha256.json').read_text());input_new=json.loads((out/'legacy/input_sha256.json').read_text())
km=lambda r:(r['group'],r['path'])
ha={km(r):(r['sha256'],r['bytes']) for r in input_old};hb={km(r):(r['sha256'],r['bytes']) for r in input_new}
report=dict(old_rows=len(a),new_rows=len(b),old_cases=len({r['case'] for r in a}),new_cases=len({r['case'] for r in b}),compared_fields=fields,differences=len(diffs),input_files_old=len(ha),input_files_new=len(hb),input_hashes_identical=ha==hb,passed=not diffs and set(ma)==set(mb) and ha==hb)
(out/'legacy_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(not report['passed'])
