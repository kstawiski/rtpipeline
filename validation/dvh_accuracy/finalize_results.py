"""Make retained synthetic benchmark evidence portable and omit host identity."""
import argparse
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

from run_benchmark import ROOT, public_value, sha, utc, write_json


def finalize(out):
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.suffix not in {'.json', '.csv', '.log', '.xml', '.md', '.patch'}:
            continue
        text = public_value(path.read_text())
        if path.suffix == '.json':
            # DVH curve files stay compact, as the pipeline writes them.
            layout = {} if path.name == 'dvh_curves.json' else {'indent': 2}
            text = json.dumps(public_value(json.loads(text)), ensure_ascii=False, **layout) + '\n'
        elif path.suffix == '.xml':
            root = ET.fromstring(text)
            for element in root.iter():
                element.attrib.pop('hostname', None)
            text = ET.tostring(root, encoding='unicode', xml_declaration=True)
        hostname = platform.node()
        if hostname:
            text = text.replace(hostname, '${HOSTNAME}')
        path.write_text(text)
    receipt_path = out / 'run_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if 'analytic_tests' in receipt:
        receipt['analytic_tests']['public_sha256'] = sha(out / 'analytic-tests.xml')
        receipt['analytic_tests']['suites'] = [s.attrib for s in ET.parse(out / 'analytic-tests.xml').iter('testsuite')]
    receipt['public_benchmark_code_sha256'] = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    receipt['provenance_paths'] = (
        'Repository-relative paths; interpreter selected with RTPIPELINE_PYTHON. '
        'Executed source hashes are retained separately from public script hashes.')
    receipt.setdefault('postprocessing', {})['portable_evidence'] = dict(
        ended_utc=utc(), status='completed', code_sha256=sha(Path(__file__)),
        command='"${RTPIPELINE_PYTHON:-python}" validation/dvh_accuracy/finalize_results.py --results '
                + str(out.relative_to(ROOT.parents[1])))
    write_json(receipt_path, receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    out = args.results.resolve()
    if not out.is_relative_to(ROOT):
        parser.error('Results must be inside the benchmark directory')
    finalize(out)
