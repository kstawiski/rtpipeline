"""Render the retained numerical tables without changing reference values."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'

def read(name): return list(csv.DictReader((OUT/name).open()))
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def counts(rows): return Counter(r['status'] for r in rows)

def main():
    config=read('configuration_map.csv');maps=read('response_map_agreement.csv')
    feature=read('feature_conformance.csv');attr=read('feature_attribution.csv')
    production=read('production_feature_coverage.csv')
    official=[r for r in feature if r['record_type']=='official reference']
    native=[r for r in maps if r['variant']=='native' and r['fraction_within_tolerance']!='']
    family=[]
    for scope,rows in [('native',official),('diagnostic adapter',attr)]:
        groups=defaultdict(list)
        for r in rows: groups[(r['configuration'],r['variant'],r['feature_family'])].append(r)
        for (cid,variant,fam),rs in groups.items():
            c=counts(rs);den=sum(r['reference'] not in ['','NA','NaN'] for r in rs)
            comparable=c['within tolerance']+c['outside tolerance']
            family.append(dict(scope=scope,configuration=cid,variant=variant,feature_family=fam,
                               assessed_pairs=len(rs),established_reference_pairs=den,
                               within_tolerance=c['within tolerance'],outside_tolerance=c['outside tolerance'],
                               not_computed=c['not computed'],not_mappable=c['not mappable'],
                               comparable_pairs=comparable,within_fraction_comparable=c['within tolerance']/comparable if comparable else '',
                               within_fraction_established=c['within tolerance']/den if den else ''))
    prodgroups=defaultdict(list)
    for r in production:prodgroups[(r['signal'],r['feature_family'])].append(r)
    for (signal,fam),rs in sorted(prodgroups.items()):
        family.append(dict(scope='production exact-reference coverage',configuration=signal,variant='production',feature_family=fam,
                           assessed_pairs=len(rs),established_reference_pairs=0,within_tolerance=0,outside_tolerance=0,
                           not_computed=0,not_mappable=len(rs),comparable_pairs=0,within_fraction_comparable='',within_fraction_established=''))
    write('family_summary.csv',family)
    bymap={(r['configuration'],r['variant']):r for r in maps}
    support=[]
    for signal in sorted({r['signal'] for r in production}):
        if signal=='wavelet-HHL':
            match=bymap[('6.a.1','native')]
            map_support='direct response-map support' if match['status']=='within tolerance' else 'direct configuration tested; native response map fails'
            why=f'IBSI 6.a.1: {100*float(match["fraction_within_tolerance"]):.6f}% voxels within tolerance; coif1 HHL, undecimated level 1, periodic boundary'
        elif signal.startswith('wavelet-'):
            map_support='none for this sub-band; partial family coverage'
            why='Only coif1 HHL has an unpooled phase-1 reference; no coif1 phase-2 configuration'
        elif signal in ['log-sigma-3-0-mm-3D','log-sigma-5-0-mm-3D']:
            cid='2.a' if '3-0' in signal else '2.b';match=bymap[(cid,'native')]
            map_support='none demonstrated; partial configuration overlap and native map fails'
            why=f'IBSI {cid} tests this physical scale on a 2-mm grid; native normalisation, boundary and recursive kernel differ'
        else:
            map_support='none at exact scale; partial family coverage'
            why='No phase-1 reference at this scale; phase-2 3D LoG uses sigma 1.5 mm'
        support.append(dict(signal=signal,computed_feature_count=sum(r['signal']==signal for r in production),
                            response_map_support=map_support,feature_value_support='none: no exact phase-2 production configuration',reason=why))
    write('production_support.csv',support)
    coif=bymap[('6.a.1','native')]
    c=counts(official);computed=c['within tolerance']+c['outside tolerance']
    first=(f'The production coif1 (Coiflet 1) HHL response map met the IBSI 2 voxel rule at all {int(coif["voxel_count"]):,} voxels '
           f'(maximum absolute difference {float(coif["maximum_absolute_difference"]):.6g}; tolerance {float(coif["absolute_tolerance"]):.6g}). '
           if coif['status']=='within tolerance' else
           f'The production coif1 (Coiflet 1) HHL response map failed the IBSI 2 voxel rule: {100*float(coif["fraction_within_tolerance"]):.6f}% of voxels were within tolerance, '
           f'with a maximum absolute difference of {float(coif["maximum_absolute_difference"]):.6g} against tolerance {float(coif["absolute_tolerance"]):.6g}. ')
    text='# RTpipeline production filters against IBSI 2\n\n'+first
    text+=(f'Native LoG failed both overlapping phase-1 scales. None of the {len(production):,} production filtered feature values '
           '(93 features from each of 12 filtered images) has an exact phase-2 reference configuration.\n\n')
    text+='## Native response maps\n\n'
    text+=('A response map is the filtered image. LoG denotes the Laplacian of Gaussian, with scale sigma expressed in millimetres. '
           'Wavelet sub-band labels specify high-pass (H) or low-pass (L) filtering along the x, y and z image axes; HHL therefore applies high-pass filters in x and y and a low-pass filter in z.\n\n')
    text+='Each comparison includes all 262,144 voxels. Passing requires every voxel to lie within 1% of the reference intensity range. A high voxel fraction can still fail this rule.\n\n'
    text+=mdtable(['IBSI configuration','Filter','Filter expressibility','Voxels within tolerance','Maximum absolute difference','Absolute tolerance','All voxels pass'],[
        [r['configuration'],next(x['filter_family'] for x in config if x['phase']=='1' and x['configuration']==r['configuration']),r['expressibility'],
         f'{100*float(r["fraction_within_tolerance"]):.6f}%',f'{float(r["maximum_absolute_difference"]):.6g}',f'{float(r["absolute_tolerance"]):.6g}',r['all_voxels_within_tolerance']]
        for r in native])+'\n\n'
    text+=('Only 6.a.1 is fully expressible as an unchanged native filter configuration. Seven configurations share a native family but differ in at least one parameter; their comparisons quantify those differences. '
           'The other 28 phase-1 configurations are not expressible. All 36 are retained in configuration_map.csv, and all 33 published response-map files were retrieved.\n\n')
    text+='## Phase 2 feature comparisons\n\n'
    text+=(f'Across 396 assessed feature–configuration pairs, {c["within tolerance"]} were within tolerance, {c["outside tolerance"]} outside tolerance, '
           f'{c["not computed"]} not computed and {c["not mappable"]} not mappable. Among {computed} comparable native values, {c["within tolerance"]} were within tolerance. '
           'These counts combine the unfiltered controls and partial filtered configurations. Production coverage is reported separately below.\n\n')
    text+='A denotes the 2D processing arm; B denotes the resampled 3D processing arm.\n\n'
    text+=mdtable(['Configuration','Within tolerance','Outside tolerance','Not computed','Not mappable','Established references'],[
        [r['configuration'],r['within_tolerance'],r['outside_tolerance'],r['not_computed'],r['not_mappable'],r['established_reference_pairs']]
        for r in family if r['scope']=='native'])+'\n\n'
    text+=('The official CSV contains 324 rows with 323 consensus values. It omits 72 Riesz feature–configuration pairs; the quartile coefficient of dispersion for 8.B has no consensus value. '
           'Those 73 gaps are explicit and have no substituted reference values. The reference family is intensity statistics only. Texture matrices, histogram entropy/uniformity and 25-HU discretisation have no phase-2 targets. '
           'PyRadiomics does not compute median absolute deviation, coefficient of variation or quartile coefficient of dispersion. Its kurtosis differs from IBSI excess kurtosis and is classified as not mappable, with the unadjusted output retained.\n\n')
    text+='## Attribution\n\n'
    text+=('The diagnostic sequences change one declared factor at each step and use the same setting for all features. The CSVs retain each parent comparison. '
           'Grid alignment and interpolation boundary handling were tested separately. The boundary control retains the native lattice, cubic B-spline CT interpolation, '
           'linear mask interpolation at a 0.5 threshold and CT rounding; it adds three replicated source edge voxels and an outside value of -1000 HU. '
           'The centred-grid step retains those boundary settings while changing grid alignment and extent.\n\n')
    text+=('On the 2-mm digital grid, removing LoG scale normalisation and converting physical derivatives to voxel-coordinate derivatives made 2.b pass at every voxel. '
           'Changing its boundary extension reduced the maximum absolute difference from '
           f'{float(bymap[("2.b","voxel_derivative_units")]["maximum_absolute_difference"]):.6g} to '
           f'{float(bymap[("2.b","boundary_no_scale_voxel_units")]["maximum_absolute_difference"]):.6g}; both were already within tolerance. '
           'For 2.a, one voxel remained outside tolerance after the normalisation and unit changes. Truncating the recursive kernel at 4 sigma retained that failure; '
           'substituting sampled Gaussian-derivative coefficients at the same cutoff removed it. This identifies kernel approximation as the residual difference on the impulse phantom.\n\n')
    text+=('Orientation averaging brought both pooled first-level wavelet configurations, 5.a.2 and 6.a.2, to 100% voxel agreement. '
           'For the second-level Haar configurations, boundary extension and orientation averaging were insufficient; adding the prescribed kernel dilation brought 7.a.1 and 7.a.2 to 100%. '
           'The unpooled db2 impulse configuration 5.a.1 passed with both native periodic and explicit zero extension: its central impulse does not challenge the boundary-condition difference.\n\n')
    text+=mdtable(['Configuration','Diagnostic change','Voxels within tolerance','Maximum absolute difference'],[
        [r['configuration'],r['variant'],f'{100*float(r["fraction_within_tolerance"]):.6f}%',f'{float(r["maximum_absolute_difference"]):.6g}']
        for r in maps if r['variant']!='native'])+'\n\n'
    text+='CT feature attribution, reported over the same 14 natively mappable statistics per computed configuration:\n\n'
    text+=mdtable(['Configuration','Variant','Within tolerance','Outside tolerance'],[
        [r['configuration'],r['variant'],r['within_tolerance'],r['outside_tolerance']]
        for r in family if r['scope']=='diagnostic adapter'])+'\n\n'
    fc={(r['configuration'],r['variant']):r for r in family if r['scope'] in ['native','diagnostic adapter']}
    text+=(f'The unfiltered 3D control matched {fc[("1.B","native")]["within_tolerance"]} of 14 comparable statistics with native preprocessing and '
           f'{fc[("1.B","centered_grid")]["within_tolerance"]} after the prescribed centred grid, linear mask interpolation and CT rounding. '
           f'For 3.B (LoG sigma 1.5 mm), agreement was {fc[("3.B","native")]["within_tolerance"]}/14 natively, '
           f'{fc[("3.B","scale_normalisation_off")]["within_tolerance"]}/14 after the preprocessing sequence and removal of scale normalisation, '
           f'{fc[("3.B","recursive_kernel_cutoff4")]["within_tolerance"]}/14 after imposing the finite kernel support, and '
           f'{fc[("3.B","finite_sampled_kernel")]["within_tolerance"]}/14 with sampled coefficients. '
           'Linear CT interpolation did not improve agreement with the prescribed spline configuration.\n\n')
    text+=(f'For first-level db3, orientation averaging produced {fc[("6.A","orientation_pooling")]["within_tolerance"]}/14 matches in 2D and '
           f'{fc[("6.B","orientation_pooling")]["within_tolerance"]}/14 in 3D. At level 2, the corresponding dilated-transform adapters produced '
           f'{fc[("7.A","atrous_level2")]["within_tolerance"]}/14 and {fc[("7.B","atrous_level2")]["within_tolerance"]}/14. '
           'These results attribute the native wavelet differences to orientation pooling and the level rule under the fixed preprocessing sequence.\n\n')
    text+='## Production coverage\n\n'
    text+=mdtable(['Production signal','Computed features','Response-map evidence','Feature-value evidence'],[
        [r['signal'],r['computed_feature_count'],r['response_map_support'],r['feature_value_support']] for r in support])+'\n\n'
    text+=('The principal message for clinicians is that reproducing one standardized filtered image does not validate the feature values from all filtered images. '
           'This benchmark provides no basis for a clinical decision threshold or a claim that the full production filtered feature set meets IBSI 2.\n\n')
    text+='## Sources and method\n\n'
    text+=('Reference configurations and tolerances follow [Whybra et al., Radiology 2024](https://doi.org/10.1148/radiol.231319), '
           'the [official IBSI filter manual](https://github.com/theibsi/ibsi2_manual/blob/b6ef5856f97a22af763cea5a424316f667a3de40/IBSI_Filters.tex), '
           'and [official reference objects](https://github.com/theibsi/ibsi_2_reference_data/tree/5404579fb3e0d17e8db421f0e82d64ce2432ec03). '
           'The receipt pins every downloaded file by URL, commit and SHA256. RTpipeline constructs the native extractor; phase-1 maps use the same PyRadiomics image generators. '
           'Phase 2 A has no interpolation, while B uses native 1-mm B-spline resampling and an official pre-filter −1000 to 400 HU window. '
           'The separate production arm uses the production YAML unchanged. No settings were fitted to individual features.\n\n')
    text+='## Limitations\n\n'
    text+=('This benchmark uses public synthetic objects and one public CT phantom. It does not establish clinical validity, generalisability or manuscript readiness. '
           'The publisher full text and supplement could not be retrieved: main and US browsers required sign-in, and no supplement file link was exposed. URLs, UTC times and verbatim access messages are retained in run_receipt.json; that branch stopped. '
           'The official GitHub reference objects and manual were retrieved independently. '
           'CT diagnostic filtering retains a 40-voxel margin after full-grid resampling; input precision is preserved for boundary and rotation tests. '
           'The official manual states no Riesz consensus, although the repository retains three Riesz response-map files; none is used to assert Riesz conformance.\n')
    (OUT/'summary.md').write_text(text)
    print('Wrote summary, family summary and production support tables.')

if __name__=='__main__':main()
