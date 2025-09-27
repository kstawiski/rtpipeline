# Pelvic Custom Structures Template

## Overview

The RTpipeline now includes a comprehensive pelvic custom structures template that automatically creates clinically relevant anatomical structures for pelvic radiotherapy planning. This template is based on the Boolean operations implemented in the `05b Boolean operations.ipynb` notebook and includes structures commonly used in pelvic RT planning.

## Automatic Usage

**The pelvic template is now used by default** when running RTpipeline. No additional configuration is required - the system will automatically create all defined pelvic structures when processing data.

```bash
# The pelvic template is used automatically
rtpipeline --dicom-root /path/to/dicom --outdir ./output

# To use a custom configuration instead
rtpipeline --dicom-root /path/to/dicom --outdir ./output --custom-structures my_custom.yaml
```

## Pelvic Structures Created

### 🩸 Vascular Structures

| Structure | Description | Source Structures | Margin |
|-----------|-------------|-------------------|---------|
| `iliac_vess` | Combined iliac vessels | iliac arteries + veins (L/R) | - |
| `iliac_area` | Iliac planning area | iliac_vess | 7mm |
| `major_vessels` | All major blood vessels | aorta, IVC, iliac_vess, portal vein | - |
| `major_vessels_5mm` | Major vessels PRV | major_vessels | 5mm |

### 🦴 Bone Structures

| Structure | Description | Source Structures | Margin |
|-----------|-------------|-------------------|---------|
| `pelvic_bones` | Core pelvic skeleton | sacrum, hip_left, hip_right, S1 | - |
| `pelvic_bones_3mm` | Bone marrow sparing | pelvic_bones | 3mm |
| `pelvic_bones_extended` | Extended pelvic bones | pelvic_bones + L4-L5 + femurs | - |
| `lumbar_spine` | Lumbar vertebrae | L1-L5 vertebrae | - |
| `bone_marrow_region` | Bone marrow areas | pelvic_bones_3mm + spine + femurs | 2mm |

### 💪 Muscle Structures

| Structure | Description | Source Structures |
|-----------|-------------|-------------------|
| `gluteus_muscles` | All gluteal muscles | gluteus maximus/medius/minimus (L/R) |
| `iliopsoas_muscles` | Hip flexor muscles | iliopsoas_left + iliopsoas_right |
| `pelvic_muscles` | Combined pelvic muscles | gluteus + iliopsoas + autochthon |

### 🔄 Bowel Structures

| Structure | Description | Source Structures | Margin |
|-----------|-------------|-------------------|---------|
| `bowel_bag` | All bowel structures | colon + small_bowel + duodenum | - |
| `bowel_PRV` | Bowel planning risk volume | bowel_bag | 5mm |
| `bladder_bowel_interface` | Interface region | bladder ∩ bowel_bag | 5mm |

### ⚠️ Organs at Risk (OAR)

| Structure | Description | Source Structures | Margin |
|-----------|-------------|-------------------|---------|
| `pelvic_OARs` | Critical pelvic OARs | bladder + bowel_bag + femurs | - |
| `pelvic_OARs_PRV` | OAR planning margins | pelvic_OARs | 3mm |
| `kidneys_combined` | Both kidneys | kidney_left + kidney_right | - |
| `kidneys_PRV` | Kidney planning volume | kidneys_combined | 5mm |

### 🎯 Analysis Structures

| Structure | Description | Source Structures |
|-----------|-------------|-------------------|
| `normal_tissue` | Non-critical normal tissue | body - pelvic_OARs_PRV |

## Integration with Analysis

All custom structures are automatically included in:

1. **DVH Analysis**: Custom structures appear in DVH metrics with source "Custom"
2. **Radiomics Analysis**: Features extracted from all custom structures
3. **Visualization**: Custom structures appear in HTML reports

## Structure Dependencies

The template is designed to work with TotalSegmentator output. Key dependencies:

### Required TotalSegmentator Structures
- `iliac_artery_left/right`
- `iliac_vena_left/right`
- `sacrum`, `hip_left/right`, `vertebrae_S1`
- `colon`, `small_bowel`, `duodenum`
- `urinary_bladder`
- `femur_left/right`
- `kidney_left/right`
- Various muscle groups

### Missing Structures
If any source structures are missing, the system will:
- Skip that specific custom structure with a warning
- Continue processing other structures
- Log which structures were successfully created

## Clinical Applications

### Pelvic Radiotherapy Planning
- **Gynecologic**: Cervical, endometrial, ovarian cancers
- **Genitourinary**: Prostate, bladder, rectal cancers
- **Bone marrow sparing**: IMRT planning with reduced hematologic toxicity

### Specific Use Cases

1. **Iliac Vessel Constraints**
   - `iliac_area` (7mm expansion) for dose constraints
   - Helps avoid vascular complications

2. **Bone Marrow Sparing**
   - `pelvic_bones_3mm` for active bone marrow regions
   - `bone_marrow_region` for comprehensive marrow areas

3. **Bowel Sparing**
   - `bowel_PRV` (5mm) for motion uncertainty
   - `bladder_bowel_interface` for overlap analysis

4. **Normal Tissue Analysis**
   - `normal_tissue` for dose-volume analysis outside OARs

## Output Files

Custom structures are saved as:
- **DICOM**: `RS_custom.dcm` in each course directory
- **DVH**: Included in `dvh_metrics.xlsx`
- **Radiomics**: Included in `radiomics_features_CT.xlsx`

## Customization

To modify the pelvic template or create your own:

1. **Copy the template**:
   ```bash
   cp custom_structures_pelvic.yaml my_pelvic_config.yaml
   ```

2. **Edit as needed** (add/remove structures, change margins)

3. **Use custom config**:
   ```bash
   rtpipeline --custom-structures my_pelvic_config.yaml ...
   ```

## Troubleshooting

### Common Issues

1. **Structure not created**: Check that source structures exist in TotalSegmentator output
2. **Margin too large**: Reduce margin if structure becomes too large
3. **Missing dependencies**: Ensure TotalSegmentator segmentation completed successfully

### Logs

Check logs for custom structure creation:
```
INFO: Added custom structure: iliac_vess
INFO: Added custom structure: iliac_area
WARNING: Source structure 'missing_structure' not found...
```

## Examples

### Check What Was Created
```python
# In analysis scripts
import pandas as pd
dvh_df = pd.read_excel('course_dir/dvh_metrics.xlsx')
custom_structures = dvh_df[dvh_df['Segmentation_Source'] == 'Custom']['ROI_Name'].unique()
print("Custom structures created:", custom_structures)
```

### DVH Analysis
```python
# Filter for specific custom structures
iliac_area_dvh = dvh_df[dvh_df['ROI_Name'] == 'iliac_area']
pelvic_bones_dvh = dvh_df[dvh_df['ROI_Name'] == 'pelvic_bones_3mm']
```

## Validation

Test the pelvic configuration:
```bash
python test_pelvic_config.py
```

This validates:
- ✅ Configuration loads correctly
- ✅ All expected structures are defined
- ✅ Structure categories are properly organized
- ✅ Important clinical structures are present

## Template Source

This template is based on the clinical pelvic RT structures created in the `05b Boolean operations.ipynb` notebook, specifically:

- Iliac vessel combinations used in multiple RT centers
- Pelvic bone structures for bone marrow sparing techniques
- OAR combinations for dose constraint applications
- Margin selections based on clinical planning standards

The template ensures consistency across all pelvic RT cases processed through RTpipeline.