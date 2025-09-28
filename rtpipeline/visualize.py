from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import plotly.graph_objs as go
import plotly.io as pio
from PIL import Image
import base64 as _b64
import io as _io

logger = logging.getLogger(__name__)


def _load_dvh_df(course_dir: Path) -> pd.DataFrame | None:
    f = course_dir / "dvh_metrics.xlsx"
    if not f.exists():
        return None
    return pd.read_excel(f)


def _extract_curve(row: pd.Series) -> Tuple[List[float], List[float]]:
    patt = re.compile(r"V(\d+)Gy \(cm³\)")
    total = float(row.get("Volume (cm³)", 0.0) or 0.0)
    doses: List[float] = []
    rel: List[float] = []
    for col in row.index:
        m = patt.match(str(col))
        if m and total > 0:
            d = float(m.group(1))
            v = float(row[col])
            doses.append(d)
            rel.append((v / total) * 100.0)
    if doses:
        pairs = sorted(zip(doses, rel), key=lambda t: t[0])
        doses, rel = [p[0] for p in pairs], [p[1] for p in pairs]
    return doses, rel


def _fig_to_b64(fig) -> str:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    enc = _b64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return enc


def _plotly_dvh(df: pd.DataFrame, title: str) -> str:
    traces = []
    for _, row in df.iterrows():
        doses, rel = _extract_curve(row)
        name = str(row.get("ROI_Name", "ROI"))
        if not doses:
            continue
        traces.append(
            go.Scatter(x=doses, y=rel, mode='lines', name=name, hovertemplate='Dose: %{x} Gy<br>Vol: %{y:.2f}%<extra></extra>')
        )
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Dose (Gy)'),
        yaxis=dict(title='Volume (%)'),
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.2),
        margin=dict(l=60, r=20, t=50, b=80),
    )
    fig = go.Figure(data=traces, layout=layout)
    # Inline JS for a standalone HTML
    return pio.to_html(fig, include_plotlyjs='inline', full_html=False)


# -----------------------------
# Axial viewer helpers
# -----------------------------

def _window_ct(arr: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    a = np.clip(arr.astype('float32'), lo, hi)
    a = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return a


def _png_b64(img: np.ndarray) -> str:
    im = Image.fromarray(img)
    buf = _io.BytesIO()
    im.save(buf, format='PNG')
    return _b64.b64encode(buf.getvalue()).decode('ascii')


def _rgba_mask_b64(mask: np.ndarray, color: Tuple[int, int, int], alpha: int = 120) -> str:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if mask.any():
        rgba[mask] = [color[0], color[1], color[2], alpha]
    im = Image.fromarray(rgba, mode='RGBA')
    buf = _io.BytesIO()
    im.save(buf, format='PNG')
    return _b64.b64encode(buf.getvalue()).decode('ascii')


def _roi_palette(names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    base = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
        (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
        (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    ]
    pal: Dict[str, Tuple[int, int, int]] = {}
    for i, n in enumerate(names):
        pal[n] = base[i % len(base)]
    return pal


def _load_ct_sitk(ct_dir: Path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        return None
    files = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


def _load_dicom_rt_seg(seg_path: Path):
    if not seg_path.exists():
        return None
    try:
        import pydicom_seg
    except Exception as e:
        logger.debug("pydicom-seg unavailable: %s", e)
        return None
    ds = pydicom.dcmread(str(seg_path))
    try:
        if str(ds.SOPClassUID) != "1.2.840.10008.5.1.4.1.1.66.4":
            logger.debug("Not a DICOM-SEG storage: %s", getattr(ds, 'SOPClassUID', None))
            return None
    except Exception:
        return None
    reader = pydicom_seg.SegmentReader()
    try:
        return reader.read(ds)
    except Exception as e:
        logger.debug("pydicom-seg failed: %s", e)
        return None


def _overlay_fig(ct_img, seg_img, title: str):
    ct = sitk.GetArrayFromImage(ct_img)
    seg = sitk.GetArrayFromImage(seg_img)
    if seg.shape != ct.shape:
        logger.warning("Seg shape %s != CT shape %s", seg.shape, ct.shape)
    Nz, Ny, Nx = ct.shape
    mz, my, mx = Nz // 2, Ny // 2, Nx // 2
    ax_ct = ct[mz]
    ax_seg = seg[mz]
    cor_ct = ct[:, my, :]
    cor_seg = seg[:, my, :]
    sag_ct = ct[:, :, mx]
    sag_seg = seg[:, :, mx]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.cm.get_cmap("jet")
    axes[0].imshow(ax_ct, cmap="gray", vmin=-200, vmax=200)
    axes[0].imshow(np.ma.masked_where(ax_seg == 0, ax_seg), cmap=cmap, alpha=0.4)
    axes[0].set_title("Axial")
    axes[0].axis("off")
    axes[1].imshow(cor_ct.T, cmap="gray", vmin=-200, vmax=200)
    axes[1].imshow(np.ma.masked_where(cor_seg.T == 0, cor_seg.T), cmap=cmap, alpha=0.4)
    axes[1].set_title("Coronal")
    axes[1].axis("off")
    axes[2].imshow(sag_ct.T, cmap="gray", vmin=-200, vmax=200)
    axes[2].imshow(np.ma.masked_where(sag_seg.T == 0, sag_seg.T), cmap=cmap, alpha=0.4)
    axes[2].set_title("Sagittal")
    axes[2].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_course(course_dir: Path) -> Optional[Path]:
    dvh_df = _load_dvh_df(course_dir)
    if dvh_df is None:
        logger.info("No DVH metrics for %s; skipping visualization", course_dir)
        return None

    dvh_div = _plotly_dvh(dvh_df, f"DVH – {course_dir.name}")

    # Generate DVH-only HTML (no CT overlay section)
    html = [
        "<html><head><meta charset='utf-8'><title>DVH Report</title>",
        "<style>body{font-family:Arial, sans-serif;margin:20px}img{max-width:100%}</style>",
        "</head><body>",
        f"<h1>DVH Report – {course_dir.parent.name} / {course_dir.name}</h1>",
        "<h2>DVH Plot</h2>",
        dvh_div,
    ]
    html.append("</body></html>")
    out = course_dir / "DVH_Report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out

# New Axial review generator
def generate_axial_review(course_dir: Path) -> Optional[Path]:
    try:
        from rt_utils import RTStructBuilder
    except Exception:
        RTStructBuilder = None
    ct_dir = course_dir / 'CT_DICOM'
    if not ct_dir.exists():
        return None
    ct_img = _load_ct_sitk(ct_dir)
    if ct_img is None:
        return None
    ct_arr = sitk.GetArrayFromImage(ct_img)
    zs, h, w = ct_arr.shape
    # Window to 8-bit
    ct_png = []
    for z in range(zs):
        sl = _window_ct(ct_arr[z], wl=40.0, ww=400.0)
        ct_png.append(_png_b64(sl))
    # Manual overlays
    overlays_manual = {}
    rs_manual = course_dir / 'RS.dcm'
    if RTStructBuilder is not None and rs_manual.exists():
        try:
            rt = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rs_manual))
            for roi in rt.get_roi_names():
                try:
                    if hasattr(rt, 'get_mask_for_roi'):
                        m3d = rt.get_mask_for_roi(roi)
                    elif hasattr(rt, 'get_roi_mask'):
                        m3d = rt.get_roi_mask(roi)
                    elif hasattr(rt, 'get_roi_mask_by_name'):
                        m3d = rt.get_roi_mask_by_name(roi)
                    else:
                        m3d = None
                except Exception:
                    m3d = None
                if m3d is None:
                    continue
                # Ensure [z,y,x] orientation and boolean mask
                ct_z, ct_y, ct_x = ct_arr.shape
                if m3d.shape != (ct_z, ct_y, ct_x):
                    if m3d.shape == (ct_y, ct_x, ct_z):
                        m3d = np.transpose(m3d, (2, 0, 1))
                    elif m3d.shape == (ct_x, ct_y, ct_z):
                        m3d = np.transpose(m3d, (2, 1, 0))
                m3d = m3d.astype(bool)
                pal = _roi_palette([roi])
                color = pal[roi]
                slmap = {}
                for z in range(m3d.shape[0]):
                    m = m3d[z]
                    if not m.any():
                        continue
                    slmap[z] = _rgba_mask_b64(m, color)
                if slmap:
                    overlays_manual[roi] = slmap
        except Exception:
            pass
    # Auto overlays
    overlays_auto = {}
    rs_auto = course_dir / 'RS_auto.dcm'
    if RTStructBuilder is not None and rs_auto.exists():
        try:
            rt = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rs_auto))
            for roi in rt.get_roi_names():
                try:
                    if hasattr(rt, 'get_mask_for_roi'):
                        m3d = rt.get_mask_for_roi(roi)
                    elif hasattr(rt, 'get_roi_mask'):
                        m3d = rt.get_roi_mask(roi)
                    elif hasattr(rt, 'get_roi_mask_by_name'):
                        m3d = rt.get_roi_mask_by_name(roi)
                    else:
                        m3d = None
                except Exception:
                    m3d = None
                if m3d is None:
                    continue
                # Ensure [z,y,x] orientation and boolean mask
                ct_z, ct_y, ct_x = ct_arr.shape
                if m3d.shape != (ct_z, ct_y, ct_x):
                    if m3d.shape == (ct_y, ct_x, ct_z):
                        m3d = np.transpose(m3d, (2, 0, 1))
                    elif m3d.shape == (ct_x, ct_y, ct_z):
                        m3d = np.transpose(m3d, (2, 1, 0))
                m3d = m3d.astype(bool)
                pal = _roi_palette([roi])
                color = pal[roi]
                slmap = {}
                for z in range(m3d.shape[0]):
                    m = m3d[z]
                    if not m.any():
                        continue
                    slmap[z] = _rgba_mask_b64(m, color)
                if slmap:
                    overlays_auto[roi] = slmap
        except Exception:
            pass

    # HTML
    import json
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Axial Viewer</title>")
    html.append("<style>body{font-family:Arial,sans-serif;margin:10px}#view{position:relative;display:inline-block}#ct{display:block} .ov{position:absolute;left:0;top:0} .controls{margin-bottom:10px} .panel{display:inline-block;vertical-align:top;margin-right:20px} img{image-rendering:pixelated}</style>")
    html.append("</head><body>")
    html.append(f"<h2>Axial Viewer – {course_dir.parent.name} / {course_dir.name}</h2>")
    html.append(f"Slice: <input type='range' id='slice' min='0' max='{zs-1}' value='{zs//2}' step='1' oninput='update()'/> <span id='slab'></span> ")
    html.append(" Opacity: <input type='range' id='opacity' min='0' max='1' value='0.4' step='0.05' oninput='updateOpacity()'/><br/>")
    html.append("<div class='panel'><h3>Manual (RS)</h3><div id='manualList'></div></div>")
    html.append("<div class='panel'><h3>Auto (RS_auto)</h3><div id='autoList'></div></div>")
    html.append("<div id='view'>")
    html.append(f"<img id='ct' width='{w}' height='{h}' src='data:image/png;base64,{ct_png[zs//2]}'/>")
    html.append("<div id='overlays'></div>")
    html.append("</div>")
    html.append("<script>")
    html.append(f"const CT_SLICES = {json.dumps(ct_png)};")
    html.append(f"const OVERLAYS_MANUAL = {json.dumps(overlays_manual)};")
    html.append(f"const OVERLAYS_AUTO = {json.dumps(overlays_auto)};")
    html.append('''
const overlaysDiv = document.getElementById('overlays');
const ctImg = document.getElementById('ct');
const sliceInput = document.getElementById('slice');
const slab = document.getElementById('slab');
const opacityInput = document.getElementById('opacity');
function safeId(s){ return s.replace(/[^A-Za-z0-9_-]/g,'_'); }
function buildList(containerId, data, prefix){
  const cont = document.getElementById(containerId);
  cont.innerHTML = '';
  const names = Object.keys(data).sort();
  names.forEach(n=>{
    const id = prefix + '_' + safeId(n);
    const cb = document.createElement('input'); cb.type='checkbox'; cb.id=id; cb.value=n; cb.onchange=updateOverlays;
    // Preselect common targets/OARs
    const nl = n.toLowerCase();
    cb.checked = (nl.includes('ptv') || nl.includes('ctv') || nl.includes('prostate') || nl.includes('odbytnica') || nl.includes('rectum') || nl.includes('bladder') || nl.includes('pecherz'));
    const lbl = document.createElement('label'); lbl.htmlFor=id; lbl.innerText=n;
    cont.appendChild(cb); cont.appendChild(lbl); cont.appendChild(document.createElement('br'));
  });
}
function currentSlice(){ return parseInt(sliceInput.value); }
function update(){
  const z = currentSlice();
  slab.innerText = z + ' / ' + (CT_SLICES.length-1);
  ctImg.src = 'data:image/png;base64,' + CT_SLICES[z];
  updateOverlays();
}
function updateOpacity(){
  const op = parseFloat(opacityInput.value);
  const imgs = overlaysDiv.querySelectorAll('img');
  imgs.forEach(im=>{ im.style.opacity = op; });
}
function updateOverlays(){
  overlaysDiv.innerHTML='';
  const z = currentSlice();
  const op = parseFloat(opacityInput.value);
  const addImgs = (data, prefix)=>{
    Object.keys(data).forEach(name=>{
      const sel = document.getElementById(prefix+'_'+safeId(name));
      if (!sel || !sel.checked) return;
      const sl = data[name];
      const b64 = sl[z];
      if (!b64) return;
      const im = document.createElement('img');
      im.className='ov'; im.style.opacity=op; im.src = 'data:image/png;base64,' + b64;
      overlaysDiv.appendChild(im);
    });
  }
  addImgs(OVERLAYS_MANUAL, 'm'); addImgs(OVERLAYS_AUTO, 'a');
}
buildList('manualList', OVERLAYS_MANUAL, 'm');
buildList('autoList', OVERLAYS_AUTO, 'a');
update();
''')
    html.append("</script>")
    html.append("</body></html>")
    out = course_dir / 'Axial.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    return out
