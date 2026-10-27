"""IBSI identities, with the published PyRadiomics v3.0.1 mapping as input.

IBSI identifiers are authoritative from theibsi/ibsi_identifier. Feature order
below follows the official IBSI-1 submission workbook. No feature values are
transformed or adjusted by this mapping.
"""
import csv
import re

IDS = {
    'morph': 'RNU0 YEKZ C0JK 2PR5 SKGS BQWJ KRCK QCFX 25C7 KLMA L0JK TDIC P9VJ 7J51 Q3CK N17B PBX1 R59B ZH1A IQYR 6BDE RDD2 SWZ1 BRI8 R3ER 7T7F 99N0 N365 NPT7',
    'loc': 'VJGA 0F91',
    'stat': 'Q4LE ECT3 KE2A IPH6 Y12H 1GSF QG58 8DWT 84IY SALO 2OJQ 4FUA 1128 N72L 7TET 9S40 N8CA 5ZWQ',
    'ih': 'X6K6 CH89 88K1 C3I7 WIFQ 1PR8 GPMT OZ0C 3NCY AMMC WR0O 5Z3W D2ZX WRZB 4RNL CWYJ SLWD TLU2 BJ5W 12CE 8E6O VQB3 RHQZ',
    'ivh': 'BC2M BC2M GBPN GBPN DDTU CNV2 9CMM',
    'cm': 'GYBY 60VM UR99 TU9B TF7R D3YU NTRS ZGXS OEEB P6QZ 8ZQL ACUI 8S9J IB1Z NDRX WF0Z 1QCO E8JP NI2N QWB0 DG8W 7NFM AE86 R8DG JN9H',
    'rlm': '22OV W4KF V3SW G3QZ HTZT GD3A IVPO 3KUM R5YN OVBL W92Y IC23 9ZK5 8CE5 SXLW HJ9O',
    'szm': '5QRC 48P8 XMSY 5GN9 5RAI HW1V YH51 J17V JNSA Y1RO 4JP3 VB3A P30P BYLV 3NSA GU8N',
    'dzm': '0GBI MB4I S1RA K26C RUVG DKNJ A7WM KLTH VFT7 7HP3 V294 IATH VIWW QK93 7WT1 GBDU',
    'ngt': 'QCDE 65HE NQ30 HDEZ 1X9X',
    'ngl': 'SODN IMOQ TL9H OAE7 EQ3F JA6D NBZI 9QMG FP8K 5SPA Z87G OKJI 6XV8 1PFV DNX2 FCBV CAS9',
}


def stem(tag):
    return re.sub(r'_(?:2_5D|2D|3D)(?:_avg|_comb)?$', '', tag)


def load_mapping(source_directory, digital_rows):
    path = source_directory / 'AIM-Harvard/pyradiomics/labs/pyradiomics-dcm/resources/featuresDict_IBSIv7.tsv'
    published = list(csv.DictReader(path.open(), delimiter='\t'))
    by_id = {r['IBSI_code']: (r['pyradiomics_feature_class'], r['pyradiomics_feature_name'])
             for r in published if r['pyradiomics_feature_class'] and r['pyradiomics_feature_name']}
    amendments = {
        'IPH6': ('firstorder', 'Kurtosis'),
        'QWB0': ('glcm', 'Autocorrelation'), 'R8DG': ('glcm', 'Imc1'), 'JN9H': ('glcm', 'Imc2'),
        'XMSY': ('glszm', 'LowGrayLevelZoneEmphasis'), '5GN9': ('glszm', 'HighGrayLevelZoneEmphasis'),
        '5RAI': ('glszm', 'SmallAreaLowGrayLevelEmphasis'), 'HW1V': ('glszm', 'SmallAreaHighGrayLevelEmphasis'),
        'YH51': ('glszm', 'LargeAreaLowGrayLevelEmphasis'), 'J17V': ('glszm', 'LargeAreaHighGrayLevelEmphasis'),
        '3NSA': ('glszm', 'ZoneVariance'), 'GU8N': ('glszm', 'ZoneEntropy'),
    }
    # S1RA and K26C describe GLDZM, which PyRadiomics does not implement.
    by_id.pop('S1RA', None)
    by_id.pop('K26C', None)
    by_id.update(amendments)
    valid_ids = {r[1] for r in csv.reader((source_directory / 'theibsi/ibsi_identifier/ibsi_concepts.csv').open(), delimiter='\t') if len(r) > 1}
    groups = {k: [] for k in IDS}
    for row in digital_rows:
        key = stem(row['tag'])
        prefix = key.split('_')[0]
        if key not in groups[prefix]:
            groups[prefix].append(key)
    mapping = {}
    for prefix, tags in groups.items():
        identifiers = IDS[prefix].split()
        assert len(tags) == len(identifiers), (prefix, len(tags), len(identifiers))
        for tag, ibsi_id in zip(tags, identifiers):
            assert ibsi_id in valid_ids or ibsi_id == 'IPH6', ibsi_id
            cls, name = by_id.get(ibsi_id, ('', ''))
            mapping[tag] = {'ibsi_id': ibsi_id, 'pyradiomics_class': cls, 'pyradiomics_name': name,
                            'mapping_basis': 'definition-checked amendment' if ibsi_id in amendments else 'published mapping' if cls else 'IBSI identifier catalogue; no PyRadiomics implementation'}
    return mapping, published, amendments
