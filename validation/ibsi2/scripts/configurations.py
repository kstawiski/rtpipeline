"""Configuration transcription from the pinned official IBSI-2 manual.

All IDs are retained, including unsupported filters and absent consensus objects.
Native means an unchanged PyRadiomics image operation selected by RTpipeline.
"""
from pathlib import Path

PYRAD = '/home/konrad/micromamba/envs/rtpipeline-radiomics/lib/python3.10/site-packages/radiomics/'
EVIDENCE = {
    'base': 'rtpipeline/radiomics.py:352,371,623,654,671; radiomics/featureextractor.py:325,330,337',
    'LoG': 'radiomics/imageoperations.py:690,708,757,758,759,762',
    'Wavelet': 'radiomics/imageoperations.py:805,807,810,831,832,833,834,842,845,852,859,874',
    'none': 'radiomics/imageoperations.py:678; radiomics/featureextractor.py:325',
    'unsupported': 'radiomics/__init__.py:100; radiomics/imageoperations.py (get*Image inventory)',
    'preprocessing': 'radiomics/imageoperations.py:420,505,518,546,555; rtpipeline/radiomics_params.yaml:18,21,60',
}

def all_configurations():
    rows=[]
    def add(phase, cid, family, phantom='', dimensions=3, boundary='mirror', sigma='',
            wavelet='', band='', level='', rotation='no', pooling='none', support='',
            status='not expressible', differences='filter family absent from PyRadiomics image operations'):
        native = 'LoG' if family=='LoG' else 'Wavelet' if wavelet else 'none' if family=='none' else 'unsupported'
        rows.append(dict(phase=phase,configuration=cid,filter_family=family,phantom=phantom,
                         dimensions=dimensions,boundary_condition=boundary,padding_support=support,
                         sigma_mm=sigma,log_scale_normalisation='none (IBSI equation)' if family=='LoG' else 'not applicable',
                         wavelet_family=wavelet or 'not applicable',decimation='undecimated' if wavelet else 'not applicable',
                         level=level or 'not applicable',subband=band or 'not applicable',rotation_invariance=rotation,
                         pooling=pooling,expressibility=status,parameter_differences=differences,
                         native_operation=native,
                         native_filter_dimensions=3 if native=='LoG' else dimensions if native=='Wavelet' else 'not applicable',
                         native_boundary_condition=('periodic' if native=='Wavelet' else 'fixed SimpleITK recursive edge continuation; no boundary setter exposed' if native=='LoG' else 'not applicable'),
                         native_padding=('generator adds one periodic sample on odd-length axes; production resampling retains padDistance=5 ROI margin' if native=='Wavelet' else
                                         'generator has no explicit padding; production resampling retains padDistance=5 ROI margin' if native=='LoG' else 'not applicable'),
                         native_log_scale_normalisation='sigma_mm squared (enabled)' if native=='LoG' else 'not applicable',
                         reference_log_derivative_coordinates='voxel coordinates; unnormalised second derivatives' if family=='LoG' else 'not applicable',
                         native_log_derivative_coordinates='physical coordinates before native sigma_mm^2 normalisation' if family=='LoG' else 'not applicable',
                         native_wavelet_level_rule=('level 1 kernel repeated on preceding approximation; no dilation at later levels' if native=='Wavelet' else 'not applicable'),
                         native_rotation_and_pooling='none; isotropic kernel' if native=='LoG' else 'no rotation averaging or pooling' if native=='Wavelet' else 'not applicable',
                         code_evidence=EVIDENCE['base']+'; '+EVIDENCE[native],
                         preprocessing_differences=('none; no phase-1 preprocessing' if phase==1 else
                         'B: native grid origin/crop, nearest-neighbour mask and no HU rounding differ from IBSI; A: no interpolation required'),
                         reference_basis='IBSI_Filters.tex:2216 (tab:benchmark_filter_settings)' if phase==1 else 'IBSI_Filters.tex:2523,2552 (lung CT processing and filter configuration tables)'))
        if family=='LoG':
            coordinate_difference='; physical-to-voxel derivative-coordinate conversion multiplies unnormalised second derivatives by spacing_mm^2 on isotropic grids'
            if phase==1:
                coordinate_difference+=' (factor 4 on the phase-1 2-mm digital phantoms)'
            elif cid.endswith('B'):
                coordinate_difference+=' (factor 1 on the phase-2 B 1-mm grid; no additional numerical discrepancy)'
            rows[-1]['parameter_differences']+=coordinate_difference
    for i,b in enumerate(['zero','nearest','periodic','mirror'],1):
        add(1,f'1.a.{i}','mean','checkerboard',boundary=b,support='M=15 voxels')
    add(1,'1.b.1','mean','impulse',2,'zero',support='M=15 voxels')
    for cid,p,d,b,s in [('2.a','impulse',3,'zero',3.),('2.b','checkerboard',3,'mirror',5.),('2.c','checkerboard',2,'mirror',5.)]:
        add(1,cid,'LoG',p,d,b,s,support='cutoff 4 sigma; finite sampled kernel',
            status='partly expressible' if d==3 else 'not expressible',
            differences=('native recursive infinite-impulse approximation; fixed edge extension; sigma^2 scale normalisation; no cutoff control'
                         if d==3 else 'LoG on the production 3D CT remains 3D; force2D affects textures/wavelets, not LoG'))
    for letter,p,d,b,k in [('a','impulse',3,'zero','E5L5S5'),('b','checkerboard',3,'mirror','E3W5R5'),('c','checkerboard',2,'mirror','L5S5')]:
        for i in range(1,4):
            add(1,f'3.{letter}.{i}','Laws',p,d,b,band=k+('; energy delta=7' if i==3 else ''),
                rotation='right-angle' if i>1 else 'no',pooling='max' if i>1 else 'none')
    for letter,p,b,s,w,g,t,delta in [('a','impulse','zero',10,4,.5,'pi/3','pi/4'),('b','sphere','mirror',20,8,2.5,'5pi/4','pi/8')]:
        for i in [1,2]:
            add(1,f'4.{letter}.{i}','Gabor',p,2,b,s,band=f'modulus; lambda={w}; gamma={g}; theta={t if i==1 else delta}',
                rotation='in-plane; orthogonal-plane average' if i==2 else 'no',pooling='average' if i==2 else 'none')
    for i in [1,2]:
        add(1,f'5.a.{i}','separable wavelet','impulse',boundary='zero',wavelet='db2',band='LHL',level=1,
            rotation='right-angle' if i==2 else 'no',pooling='average' if i==2 else 'none',status='partly expressible',
            differences='native periodic extension instead of zero'+('; no orientation pooling' if i==2 else ''))
    for i in [1,2]:
        add(1,f'6.a.{i}','separable wavelet','sphere',boundary='periodic',wavelet='coif1',band='HHL',level=1,
            rotation='right-angle' if i==2 else 'no',pooling='average' if i==2 else 'none',
            status='expressible' if i==1 else 'partly expressible',differences='none' if i==1 else 'native operation does not pool orientations')
    for i,band in [(1,'LLL'),(2,'HHH')]:
        add(1,f'7.a.{i}','separable wavelet','checkerboard',boundary='mirror',wavelet='haar',band=band,level=2,
            rotation='right-angle',pooling='average',status='partly expressible',
            differences='native periodic extension; no orientation pooling; repeated level-1 kernels instead of a-trous dilation at level 2')
    for i in [1,2,3]:
        add(1,f'8.a.{i}','Simoncelli','checkerboard',boundary='periodic',band='B',level=i)
    for cid,p in [('9.a','impulse'),('9.b.1','sphere'),('9.b.2','sphere')]:
        add(1,cid,'Riesz LoG',p,boundary='zero',sigma=3.,band='(1,0,0)' if cid=='9.a' else '(0,2,0)',
            support='cutoff 4 sigma; structure alignment sigma=1 mm' if cid.endswith('2') else 'cutoff 4 sigma')
    for cid,p,b in [('10.a','impulse','zero'),('10.b.1','pattern_1','nearest'),('10.b.2','pattern_1','nearest')]:
        add(1,cid,'Riesz Simoncelli',p,boundary=b,band='B; (1,0,0)' if cid=='10.a' else 'B; (0,2,0)',level=1,
            support='structure alignment sigma=1 mm' if cid.endswith('2') else '')
    for suffix,dim in [('A',2),('B',3)]:
        for number,family in [(1,'none'),(2,'mean'),(3,'LoG'),(4,'Laws'),(5,'Gabor'),(6,'separable wavelet'),
                              (7,'separable wavelet'),(8,'Simoncelli'),(9,'Simoncelli'),(10,'Riesz Simoncelli'),(11,'Riesz Simoncelli')]:
            kw=dict()
            if number==1:
                kw.update(status='expressible',differences='no filtering; preprocessing differences assessed separately')
            if number==2:
                kw.update(support='M=5 voxels')
            if number==3:
                kw.update(sigma=1.5,support='cutoff 4 sigma; finite sampled kernel',
                          status='partly expressible' if dim==3 else 'not expressible',
                          differences='sigma^2 normalisation; recursive rather than finite sampled kernel; fixed boundary' if dim==3 else 'no slice-wise LoG setting for the production 3D CT')
            if number==4:
                kw.update(band='L5E5' if dim==2 else 'L5E5E5',rotation='right-angle',pooling='max',support='energy delta=7')
            if number==5:
                kw.update(sigma=5,band='modulus; lambda=2; gamma=1.5; delta theta=pi/8',rotation='in-plane',pooling='average; orthogonal planes for B')
            if number in [6,7]:
                kw.update(wavelet='db3',band=('LH' if dim==2 else 'LLH') if number==6 else 'H'*dim,
                          level=1 if number==6 else 2,rotation='right-angle',pooling='average',status='partly expressible',
                          differences='no orientation pooling; periodic instead of mirror extension'+('; level-2 kernels are not dilated' if number==7 else ''))
            if number in [8,9,10,11]:
                kw.update(band='B'+('; Riesz (0,2)' if dim==2 else '; Riesz (0,2,0)') if number in [10,11] else 'B',
                          level=2 if number==9 else 1,support='structure alignment sigma=1 mm' if number==11 else '')
            add(2,f'{number}.{suffix}',family,'CT GTV',2 if number==5 else dim,**kw)
    assert len([r for r in rows if r['phase']==1])==36
    assert len([r for r in rows if r['phase']==2])==22
    return rows
