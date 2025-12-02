import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

import pydicom
import pydicom_seg
import SimpleITK as sitk

# -----------------------------------------------------
# 1) DVH Analysis
# -----------------------------------------------------

def load_dvh_data(patient_dir):
    """
    Load DVH metrics from 'dvh_metrics.xlsx' in the patient directory.
    Returns a pandas DataFrame or None if not found.
    """
    dvh_file = os.path.join(patient_dir, "dvh_metrics.xlsx")
    if not os.path.exists(dvh_file):
        print(f"DVH file not found in {patient_dir}")
        return None
    df = pd.read_excel(dvh_file)
    return df

def extract_dvh_curve(row):
    """
    Extract DVH curve data points from columns named like 'V{dose}Gy (cm³)'.
    Returns (dose_list, rel_vol_list).
    """
    pattern = re.compile(r"V(\d+)Gy \(cm³\)")
    total_vol = row.get("Volume (cm³)", 0.0)
    doses, rel_vol = [], []
    for col in row.index:
        m = pattern.match(col)
        if m and total_vol > 0:
            dose_val = float(m.group(1))
            vol_cc = row[col]
            doses.append(dose_val)
            rel_vol.append((vol_cc / total_vol) * 100.0)
    # Sort by ascending dose
    if doses:
        doses, rel_vol = (list(t) for t in zip(*sorted(zip(doses, rel_vol))))
    return doses, rel_vol

def plot_dvh_curves(df, title):
    """
    Plot DVH curves for all structures in 'df' with dose vs. volume%.
    Legend is placed outside the plot to avoid overlap.
    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7,5))
    for _, row in df.iterrows():
        doses, rel_vol = extract_dvh_curve(row)
        ax.plot(doses, rel_vol, label=row["ROI_Name"])
    ax.set_xlabel("Dose (Gy)")
    ax.set_ylabel("Volume (%)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    return fig

def compute_coverage_and_integral(dvh_df):
    """
    Compute coverage metrics (D95Gy, D2Gy, etc.) and integral dose (mean_dose x volume).
    Returns a list of dictionaries, one per structure row in dvh_df.
    """
    results = []
    for _, row in dvh_df.iterrows():
        structure = row["ROI_Name"]
        presc = row.get("PrescribedDose_Gy", np.nan)
        d95 = row.get("D95Gy", np.nan)
        d98 = row.get("D98Gy", np.nan)
        d2  = row.get("D2Gy",  np.nan)
        mean_dose = row.get("DmeanGy", np.nan)
        vol_cc = row.get("Volume (cm³)", np.nan)
        
        # e.g. if presc=50 => look for "V50Gy (%)"
        v_presc = np.nan
        if not np.isnan(presc):
            presc_col = f"V{int(round(presc))}Gy (%)"
            v_presc = row.get(presc_col, np.nan)
        
        integral_dose = np.nan
        if not np.isnan(mean_dose) and not np.isnan(vol_cc):
            integral_dose = mean_dose * vol_cc
        
        results.append({
            "Structure": structure,
            "PrescribedDose_Gy": presc,
            "D95Gy": d95,
            "D98Gy": d98,
            "D2Gy": d2,
            "V_prescription_%": v_presc,
            "Volume_cm3": vol_cc,
            "DmeanGy": mean_dose,
            "IntegralDose_Gy_cm3": integral_dose
        })
    return results

def fig_to_base64(fig):
    """
    Convert a matplotlib Figure into a base64-encoded PNG string for embedding in HTML.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

# -----------------------------------------------------
# 2) Reading DICOM-RT Segmentation & CT with SimpleITK
# -----------------------------------------------------

def load_ct_sitk(dicom_folder):
    """
    Read a DICOM CT series from 'dicom_folder' using SimpleITK and return a sitk.Image.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        print(f"No DICOM series found in {dicom_folder}")
        return None
    file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_ids[0])
    reader.SetFileNames(file_names)
    ct_image = reader.Execute()
    return ct_image

def load_dicom_rt_seg(rt_seg_path):
    """
    Read a DICOM-RT segmentation file (TotalSegmentator.dcm) using pydicom-seg.
    Returns a SimpleITK image (label map) or None if fails.
    """
    if not os.path.exists(rt_seg_path):
        print(f"RT segmentation file not found: {rt_seg_path}")
        return None
    ds = pydicom.dcmread(rt_seg_path)
    reader = pydicom_seg.SegmentReader()
    try:
        seg_image = reader.read(ds)  # returns a SimpleITK label image
        return seg_image
    except Exception as e:
        print("Error reading RT segmentation with pydicom-seg:", e)
        return None

def overlay_ct_seg_sitk(ct_image, seg_image, patient_id):
    """
    Convert both sitk.Images to NumPy, then display mid axial/coronal/sagittal slices
    with segmentation overlaid. seg_image is assumed to be in the same geometry as ct_image.
    Returns a matplotlib Figure.
    """
    # Convert SITK => NumPy arrays. Format: [z, y, x].
    ct_array = sitk.GetArrayFromImage(ct_image)
    seg_array = sitk.GetArrayFromImage(seg_image)
    
    # If shapes differ, the segmentation might not match the CT. Possibly resample seg -> ct.
    if seg_array.shape != ct_array.shape:
        print(f"Warning: seg shape {seg_array.shape} != CT shape {ct_array.shape}")
        # We can resample if needed, but if the RT-SEG references the same CT,
        # they should typically match. We'll skip here to highlight mismatch.
    
    Nz, Ny, Nx = ct_array.shape
    mid_z = Nz // 2
    mid_y = Ny // 2
    mid_x = Nx // 2
    
    # Extract mid slices
    axial_ct = ct_array[mid_z, :, :]
    axial_seg = seg_array[mid_z, :, :]
    coronal_ct = ct_array[:, mid_y, :]
    coronal_seg = seg_array[:, mid_y, :]
    sagittal_ct = ct_array[:, :, mid_x]
    sagittal_seg = seg_array[:, :, mid_x]
    
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    cmap_seg = plt.cm.get_cmap('jet')
    
    # Axial
    axes[0].imshow(axial_ct, cmap='gray', vmin=-200, vmax=200)
    mask_axial = np.ma.masked_where(axial_seg == 0, axial_seg)
    axes[0].imshow(mask_axial, cmap=cmap_seg, alpha=0.4)
    axes[0].set_title("Axial (Z)")
    axes[0].axis("off")
    
    # Coronal
    axes[1].imshow(coronal_ct.T, cmap='gray', vmin=-200, vmax=200)
    mask_coronal = np.ma.masked_where(coronal_seg.T == 0, coronal_seg.T)
    axes[1].imshow(mask_coronal, cmap=cmap_seg, alpha=0.4)
    axes[1].set_title("Coronal (Y)")
    axes[1].axis("off")
    
    # Sagittal
    axes[2].imshow(sagittal_ct.T, cmap='gray', vmin=-200, vmax=200)
    mask_sagittal = np.ma.masked_where(sagittal_seg.T == 0, sagittal_seg.T)
    axes[2].imshow(mask_sagittal, cmap=cmap_seg, alpha=0.4)
    axes[2].set_title("Sagittal (X)")
    axes[2].axis("off")
    
    fig.suptitle(f"CT with DICOM-RT Segmentation – Patient {patient_id}")
    plt.tight_layout()
    return fig

# -----------------------------------------------------
# 3) Main Loop: DVH Analysis + RT-Seg Overlay + HTML
# -----------------------------------------------------

def generate_html_report(base_dir="../Data_Organized"):
    """
    Loop over patient folders in 'base_dir', read dvh_metrics.xlsx, compute coverage, plot DVHs,
    load CT from DICOM, load RT segmentation from 'TotalSegmentator.dcm', overlay them,
    and generate an HTML report with embedded images + summary metrics.
    """
    coverage_log = []
    patient_ids = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for patient_id in patient_ids:
        patient_dir = os.path.join(base_dir, patient_id)
        print(f"Processing patient {patient_id} ...")
        
        # 1) Load DVH data
        dvh_df = load_dvh_data(patient_dir)
        if dvh_df is None:
            continue
        
        # 2) Possibly filter data: e.g., separate targets vs. OARs if you want
        # For simplicity, we'll just treat dvh_df as one set. Or do your own logic here.
        
        # 3) Plot DVH
        fig_dvh = plot_dvh_curves(dvh_df, f"DVH – Patient {patient_id}")
        dvh_b64 = fig_to_base64(fig_dvh)
        
        # 4) Compute coverage & integral dose
        metrics = compute_coverage_and_integral(dvh_df)
        for m in metrics:
            m["Patient"] = patient_id
        coverage_log.extend(metrics)
        
        # 5) Load CT and RT segmentation
        ct_dicom_dir = os.path.join(patient_dir, "CT_DICOM")
        rt_seg_path = os.path.join(patient_dir, "TotalSegmentator.dcm")
        
        ct_image = load_ct_sitk(ct_dicom_dir)
        seg_image = load_dicom_rt_seg(rt_seg_path)
        
        overlay_b64 = None
        if ct_image is not None and seg_image is not None:
            fig_overlay = overlay_ct_seg_sitk(ct_image, seg_image, patient_id)
            overlay_b64 = fig_to_base64(fig_overlay)
        else:
            print("CT or RT segmentation not available. Skipping overlay.")
        
        # 6) Build HTML summary
        summary_lines = [f"<h2>Patient {patient_id} – DVH Metrics Summary</h2>",
                         "<ul>"]
        for m in metrics:
            summary_lines.append(
                f"<li><strong>{m['Structure']}</strong>: "
                f"D95={m['D95Gy']:.2f} Gy, D2={m['D2Gy']:.2f} Gy, "
                f"MeanDose={m['DmeanGy']:.2f} Gy, IntegralDose={m['IntegralDose_Gy_cm3']:.1f} Gy·cm³</li>"
            )
        summary_lines.append("</ul>")
        summary_html = "\n".join(summary_lines)
        
        # 7) Generate final HTML for this patient
        output_dir = os.path.join(patient_dir, "DVH_Analysis")
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, f"{patient_id}_DVH_Report.html")
        
        html_content = f"""<html>
<head>
  <meta charset="utf-8">
  <title>DVH Analysis – Patient {patient_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .section {{ margin-bottom: 40px; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>DVH Analysis Report for Patient {patient_id}</h1>
  <div class="section">
    <h2>DVH Plot</h2>
    <img src="data:image/png;base64,{dvh_b64}" />
  </div>
"""
        if overlay_b64:
            html_content += f"""<div class="section">
    <h2>CT & RT-Seg Overlay</h2>
    <img src="data:image/png;base64,{overlay_b64}" />
  </div>
"""
        html_content += f"""<div class="section">
    <h2>Summary Metrics</h2>
    {summary_html}
  </div>
</body>
</html>"""
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {html_path}")
    
    # 8) Save coverage metrics for all patients
    coverage_df = pd.DataFrame(coverage_log)
    coverage_csv = os.path.join(base_dir, "DVH_coverage_summary.csv")
    coverage_df.to_csv(coverage_csv, index=False)
    print(f"Coverage metrics saved to {coverage_csv}")


# ------------------------------------------
# Example Usage:
# generate_html_report(base_dir="../Data_Organized")
# ------------------------------------------



generate_dvh_html_report(base_dir="../Data_Organized")

