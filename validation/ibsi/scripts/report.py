"""Render the clinician-facing results from the retained numeric tables."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def main():
    rows = list(csv.DictReader((OUT / 'feature_conformance.csv').open()))
    rows = [r for r in rows if not r['family'].startswith('Diagnostics')]
    families = list(csv.DictReader((OUT / 'family_summary.csv').open()))
    receipt = json.loads((OUT / 'run_receipt.json').read_text())
    production = json.loads((OUT / 'production_configuration.json').read_text())
    repos = receipt['source_repositories']
    commit = repos['AIM-Harvard/pyradiomics']['commit']
    doc_commit = repos['theibsi/ibsi-doc']['commit']
    mapping_url = f'https://github.com/AIM-Harvard/pyradiomics/blob/{commit}/labs/pyradiomics-dcm/resources/featuresDict_IBSIv7.tsv'
    faq_url = f'https://github.com/AIM-Harvard/pyradiomics/blob/{commit}/docs/faq.rst'
    doc_url = f'https://github.com/theibsi/ibsi-doc/blob/{doc_commit}/docs/05_Reference_data_sets.rst'
    feature_url = f'https://github.com/theibsi/ibsi-doc/blob/{doc_commit}/docs/03_Image_features.rst'
    def selected(arm, config):
        return [r for r in rows if (r['arm'], r['configuration']) == (arm, config)]
    def score(rr):
        w = sum(r['status'] == 'within tolerance' for r in rr)
        n = sum(r['status'] in ['within tolerance', 'outside tolerance'] for r in rr)
        return f'{w}/{n} ({100*w/n:.1f}%)' if n else 'Not run'
    def primary(rr, config):
        allowed = ['', '2.5D', '2.5D, direction-merged'] if config in 'AB' else ['', '3D', '3D, averaged']
        return [r for r in rr if r['aggregation'] in allowed]
    text = []
    text.append('RTpipeline CT radiomics: public IBSI 1 phantom conformance\n')
    text.append('Agreement depended on preprocessing and feature definition. With one aggregation per feature, the digital phantom matched 100/102 features (98.0%). Native CT extraction matched 101/102 in configuration A, 54/102 in B, 27/102 in C and 23/102 in D. These results do not demonstrate full IBSI 1 conformance.\n')
    text.append('Only the public IBSI digital phantom and the public lung CT phantom with its supplied GTV mask were used. No clinical cohort was accessed. The installed software was RTpipeline '+receipt['versions']['rtpipeline']+', PyRadiomics '+receipt['versions']['pyradiomics']+', NumPy '+receipt['versions']['numpy']+' and SimpleITK '+receipt['versions']['SimpleITK']+'.\n')
    text.append('The subset with one aggregation per feature contains 102 mapped Original-image outputs: GLCM and GLRLM are direction-averaged, with volumetric matrices for C–E and the digital phantom, and pooled axial matrices for A–B. Other texture families use the corresponding 3D or 2.5D matrix. This subset excludes the five PyRadiomics outputs without an IBSI 1 counterpart.\n')
    text.append('| Extraction | Configuration | One aggregation per feature | All computed comparison pairs | Reference coverage |\n|---|---|---:|---:|---:|')
    order = [('digital', 'digital')] + [('native', c) for c in 'ABCDE'] + [('ibsi_preprocessed', c) for c in 'ABCDE']
    for arm, config in order:
        rr = selected(arm, config)
        n = sum(r['status'] in ['within tolerance', 'outside tolerance'] for r in rr)
        nr = sum(r['has_consensus_reference'] == 'True' for r in rr)
        text.append(f'| {arm} | {config} | {score(primary(rr, config))} | {score(rr)} | {n}/{nr} ({100*n/nr:.1f}%) |')
    text.append('\nA comparison pair is one feature at one aggregation method. A feature evaluated using both averaged and merged matrices contributes two pairs. Reference coverage includes all standardized reference pairs in the workbook for that configuration; it excludes diagnostics and entries without a consensus value. It therefore also includes unsupported families and slice-wise aggregation methods. Proportions among computed features must be read together with this coverage. No pooled percentage across phantoms or configurations is used.\n')
    text.append('Native extraction and explicit IBSI preprocessing\n')
    text.append('Every extraction used RTpipeline’s `radiomics._extractor` and `radiomics_ct_contract.build_ct_extractors`, followed by the configured PyRadiomics `execute` method. Shape used the separate, unresegmented morphology mask. The numerical extraction ran inside the designated radiomics environment, so the conda-launch wrapper added no numerical operation and was not invoked. Clinical case ingestion, RTSTRUCT conversion and publication orchestration were outside this test.\n')
    text.append('Native A used the original spacing, the HU interval [−500, 400] and 25-HU bins. Native B requested 2 × 2 mm in-plane linear image interpolation, [−500, 400] and 32 bins. Native C requested 2-mm isotropic linear image interpolation, [−1000, 400] and 25-HU bins. Native D requested the same 3D interpolation, 3-SD outlier exclusion and 32 bins. E cannot be represented unchanged by a single standard PyRadiomics parameter file because it combines absolute-range and sigma resegmentation; its native rows are marked not computed. Native B–D are implementation comparisons with IBSI targets, not claims that the complete IBSI preprocessing protocol is expressible by native settings. [IBSI configurations]('+doc_url+'), [PyRadiomics preprocessing definitions]('+faq_url+').\n')
    text.append('The explicit preprocessing arm used the IBSI-centered grid with ceiling-rounded dimensions; bilinear interpolation for B, trilinear interpolation for C–D, and cubic B-spline image interpolation for E; linear floating-point mask interpolation with a ≥0.5 threshold; and nearest-integer HU rounding. Morphology and intensity masks remained separate. A–C applied their specified HU intervals. D excluded values beyond three population SDs of the unresegmented ROI. E intersected that outlier criterion with [−1000, 400]. These were fixed rules per configuration. PyRadiomics feature formulas and fixed-bin-width discretization were retained. No feature value was shifted to obtain agreement. All 300 preprocessing diagnostics matched the official workbook under its comparison rule; values are in `preprocessing_diagnostics.csv`.\n')
    text.append('The available whole-volume modes were 3D averaged and 3D merged GLCM/GLRLM matrices, and 2.5D direction-merged and fully merged axial matrices. The digital phantom covered both dimensionalities. A–B covered the two 2.5D modes; C–E covered the two 3D modes. In PyRadiomics, `force2D` suppresses through-plane links and pools matrices across slices. It does not perform the separate per-slice feature averaging used by the IBSI 2D reference variants. Those variants are explicitly not computed here.\n')
    text.append('Family results\n')
    short = {'Morphology': 'Morphology', 'Statistics': 'Intensity statistics', 'Intensity histogram': 'Intensity histogram',
             'Co-occurrence matrix': 'GLCM', 'Run length matrix': 'GLRLM', 'Size zone matrix': 'GLSZM',
             'Neighbourhood grey tone difference matrix': 'NGTDM', 'Neighbouring grey level dependence matrix': 'NGLDM / PyRadiomics GLDM'}
    lookup = {(r['arm'], r['configuration'], r['family']): r for r in families}
    for arm, configs in [('native', list('ABCDE')), ('ibsi_preprocessed', list('ABCDE'))]:
        text.append(arm+'; within tolerance / computed pairs.\n')
        columns = [('digital', 'digital')] + [(arm, c) for c in configs] if arm == 'native' else [(arm, c) for c in configs]
        text.append('| Family | '+' | '.join('Digital' if c == 'digital' else c for _, c in columns)+' |')
        text.append('|---|'+'---:|'*len(columns))
        for family, label in short.items():
            counts = []
            for aa, cc in columns:
                f = lookup[(aa, cc, family)]
                counts.append(f"{f['within_tolerance']}/{f['computed_with_reference']}" if f['computed_with_reference'] != '0' else '—')
            text.append('| '+label+' | '+' | '.join(counts)+' |')
        text.append('')
    text.append('`family_summary.csv` also reports uncomputed features, unmappable outputs, agreement against all reference pairs, reference coverage and strict unrounded comparisons. The enabled PyRadiomics feature set has no GLDZM, local-intensity or intensity-volume-histogram outputs. It also lacks much of the morphology and intensity-histogram reference set. Deprecated getters omitted by the production feature-class convention remain not computed; no substitute formulas were introduced. The five unmappable outputs are TotalEnergy, GLCM MCC and the three maximum 2D diameters in the shape class.\n')
    text.append('Feature definitions and remaining discrepancies\n')
    text.append('Feature identities follow the [published PyRadiomics IBSI dictionary]('+mapping_url+') and the [official IBSI identifier catalogue](https://github.com/theibsi/ibsi_identifier/blob/'+repos['theibsi/ibsi_identifier']['commit']+'/ibsi_concepts.csv). Class and getter assignments were checked against the installed v3.0.1 implementation. `feature_mapping.csv` records each assignment and its basis. IBSI GLDZM features were not equated to GLSZM features. Kurtosis was mapped to the corresponding IBSI identifier while retaining its documented definition difference.\n')
    text.append('PyRadiomics reports kurtosis rather than excess kurtosis. In the digital phantom, the raw value was 2.6453795193 against −0.355, with published tolerance 0. It was outside tolerance in every CT configuration as well. Subtracting three was not part of this validation. [PyRadiomics definition]('+faq_url+').\n')
    text.append('The digital phantom’s least-axis length was 8.4781094932 mm against 8.54 mm, with tolerance 0.05 mm; the workbook difference was 0.06 mm. The installed shape implementation normalizes its covariance by N. Using N−1 for this 74-voxel phantom gives 8.5359812196 mm, consistent with the reference. This normalization explanation is an inference from the implementation and reference value, not an asserted correction to either definition. [PyRadiomics shape source](https://github.com/AIM-Harvard/pyradiomics/blob/'+commit+'/radiomics/shape.py), [IBSI shape references]('+feature_url+').\n')
    text.append('Configuration C retains a fixed-bin-width difference after explicit preprocessing: PyRadiomics resets the lowest occupied bin to 1, whereas IBSI uses the specified −1000-HU lower bound. Here the minimum ROI intensity was −939 HU, so the bin indices differ by two. This affects gray-level-sensitive features: for example, direction-averaged GLCM joint average was 36.9779305122 against 39.0, tolerance 0.2. C had 32 outside-tolerance pairs, including kurtosis. A’s observed minimum of −500 HU coincided with its prescribed lower bound. [Discretization definitions]('+faq_url+').\n')
    text.append('Native resampling differs from IBSI in grid alignment, mask interpolation and intensity rounding. These differences propagate through shape and texture features. The explicit preprocessing arm isolates the retained PyRadiomics feature and binning conventions; its high agreement in B, D and E does not establish native pipeline agreement with those protocols. [Documented differences]('+faq_url+').\n')
    text.append('Comparison rule\n')
    text.append('The primary status reproduces the official workbook: `ABS(ROUNDDOWN(reference − result, 2 − floor(log10(abs(reference))))) ≤ tolerance`; for a zero reference the workbook uses zero decimal places. This truncation is material when tolerance is zero or a reference is rounded. `absolute_error` and `strict_raw_within_tolerance` retain the unrounded comparison. Missing consensus values are not assigned a conformance outcome. The unversioned [official reference workbook](https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx), linked from the [IBSI 1 results page](https://theibsi.github.io/ibsi1/), is pinned by SHA256 `3af20577b580f2161bd546e147338ff226abb1676488e2906f2ce908b7985710`. Participant-submission values and older tolerance tables were retained as source context, not used as comparison targets.\n')
    text.append('Default production CT configuration\n')
    text.append('| Setting | RTpipeline production default | IBSI benchmark runs |\n|---|---|---|\n| Discretization | Fixed width, 25 HU | Digital: width 1; A/C: width 25 HU; B/D/E: 32 bins |\n| Resampling | 1 × 1 × 1 mm | A/digital: none; B: 2 × 2 mm in plane; C–E: 2-mm isotropic |\n| Image interpolation | Cubic B-spline | B: linear in plane; C/D: trilinear; E: cubic B-spline |\n| Image signals | Original, LoG σ 1/2/3/5 mm, coif1 wavelets, level 1 | Original only |\n| Distances | [1, 2] | [1] |\n| Dimensionality | 3D | 2.5D for A/B; 3D for C–E; both for digital |\n| Normalization / energy shift | Off / 0 | Off / 0 |\n| Minimum ROI | 64 voxels; at least two dimensions | Same extraction admission thresholds |\n| Mask handling | Separate unresegmented morphology; nearest-neighbor native resampling | Explicit arm: linear mask interpolation, ≥0.5 threshold |\n')
    text.append('The production primary arm uses [−1000, 400] HU for targets and hollow pelvic organs, and [−500, 400] HU for solid soft-tissue/neural structures. Its sensitivity arm uses the unresegmented intensity mask. Shape is extracted independently from the unresegmented morphology mask on the Original image. Other ROI classes can be excluded by the production class policy. Full settings, enabled classes and the configured YAML hash are retained in `production_configuration.json`. This default configuration was recorded, not run as an IBSI configuration. The LoG/wavelet outputs and distance-2 production textures were not benchmarked by this IBSI 1 experiment.\n')
    text.append('Scope and reproducibility\n')
    text.append('This is a software comparison on two public test objects. It does not measure segmentation reproducibility, scanner robustness, clinical prediction or performance in patient cohorts. Local execution does not establish clinical or manuscript readiness. Exact URLs, commits, file hashes, installed-source hashes, effective settings and UTC times are in `run_receipt.json`; reproduction commands are in `../README.md`.\n')
    (OUT / 'summary.md').write_text('\n'.join(text))
    print('Wrote results/summary.md')


if __name__ == '__main__':
    main()
