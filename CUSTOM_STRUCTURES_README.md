# Custom Structures in RTpipeline

## Overview

The custom structures feature allows you to create new structures from existing ones using boolean operations (union, intersection, subtraction, XOR) and apply margins. These custom structures are automatically included in all downstream analyses including DVH calculations and radiomics feature extraction.

## Features

- **Boolean Operations**: Combine structures using union, intersection, subtraction, and XOR operations
- **Margin Operations**: Add uniform or asymmetric margins to structures
- **TotalSegmentator Integration**: Use automatically segmented structures from TotalSegmentator in custom structure definitions
- **Full Pipeline Integration**: Custom structures are included in DVH and radiomics analyses

## Configuration

Custom structures are defined in a YAML configuration file. See `custom_structures_example.yaml` for a complete example.

### Basic Structure

```yaml
custom_structures:
  - name: "Structure_Name"
    operation: "union"  # union, intersection, subtract, xor
    source_structures: ["Source1", "Source2"]
    margin: 5  # Optional: uniform margin in mm
    description: "Optional description"
```

### Margin Options

Margins can be specified as:
- **Uniform**: Single value applies to all directions
- **Asymmetric**: Different values for each direction

```yaml
# Uniform margin
margin: 5  # 5mm in all directions

# Asymmetric margin
margin:
  anterior_mm: 10
  posterior_mm: 5
  left_mm: 7
  right_mm: 7
  superior_mm: 15
  inferior_mm: 5
```

## Usage

### Command Line

Add the `--custom-structures` parameter to your rtpipeline command:

```bash
rtpipeline --dicom-root /path/to/dicom \
           --outdir ./output \
           --custom-structures custom_structures.yaml
```

### Available Operations

1. **Union**: Combines structures (logical OR)
   ```yaml
   - name: "Combined_OARs"
     operation: "union"
     source_structures: ["Heart", "Lungs", "Esophagus"]
   ```

2. **Intersection**: Overlap between structures (logical AND)
   ```yaml
   - name: "PTV_Lung_Overlap"
     operation: "intersection"
     source_structures: ["PTV", "Lung_L", "Lung_R"]
   ```

3. **Subtraction**: Remove one structure from another
   ```yaml
   - name: "Ring_Structure"
     operation: "subtract"
     source_structures: ["PTV_10mm", "PTV_5mm"]
   ```

4. **XOR**: Exclusive OR (symmetric difference)
   ```yaml
   - name: "PTV_Boundary"
     operation: "xor"
     source_structures: ["PTV", "PTV_2mm"]
   ```

## Examples

### Example 1: PTV with Margin
```yaml
- name: "PTV_5mm"
  operation: "union"
  source_structures: ["PTV"]
  margin: 5
  description: "PTV with 5mm uniform margin"
```

### Example 2: Ring Structure for Dose Evaluation
```yaml
- name: "Ring_5-10mm"
  operation: "subtract"
  source_structures: ["PTV_10mm", "PTV_5mm"]
  description: "Ring structure 5-10mm from PTV"
```

### Example 3: OAR Avoidance Structure
```yaml
- name: "OAR_avoid"
  operation: "union"
  source_structures: ["SpinalCord", "Heart", "Esophagus"]
  margin: 3
  description: "Combined OARs with 3mm margin"
```

## Source Structures

You can use structures from:
- **Manual contours**: ROI names from manual RS.dcm files
- **TotalSegmentator**: Automatically segmented structures (e.g., "liver", "heart", "lung_upper_lobe_left")
- **Previously defined custom structures**: Custom structures can be used as sources for other custom structures

## Output

Custom structures are:
1. Saved as `RS_custom.dcm` in each course directory
2. Included in DVH calculations (marked as "Custom" source)
3. Included in radiomics feature extraction
4. Available in all downstream analyses

## Testing

Run the test script to verify the implementation:

```bash
python test_custom_structures.py
```

## Notes

- Structure names are case-sensitive
- Operations are applied in the order structures are listed
- For subtract operations, the first structure is the base, and subsequent structures are subtracted from it
- Margins are applied after boolean operations
- Invalid or missing source structures will be skipped with a warning

## Troubleshooting

If custom structures are not being created:
1. Check that the YAML file path is correct
2. Verify that source structure names match exactly (case-sensitive)
3. Check the log files for warnings about missing structures
4. Ensure RS.dcm or RS_auto.dcm exists as a base for custom structures