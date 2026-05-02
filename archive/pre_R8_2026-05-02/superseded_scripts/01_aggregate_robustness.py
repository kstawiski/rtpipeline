#!/usr/bin/env python3
"""Step 1 v2: Aggregate robustness parquets with streaming per-cohort writes.

Writes ONE parquet per cohort under data/robustness_by_cohort/<cohort>.parquet
(avoids OOM on 527M-row combined DataFrame). Step 02 can read via pa.dataset.

Path convention
---------------
Each per-course parquet lives at:
    /home/kgs24/rtpipeline_manuscript/<Cohort>/data/<patient>/<course>/radiomics_robustness_ct.parquet

So Path.parts indices are:
    parts[-1] = 'radiomics_robustness_ct.parquet'  (file)
    parts[-2] = <course>                           (course directory, e.g. '2021-06')
    parts[-3] = <patient>                          (patient directory, numeric ID)
    parts[-4] = 'data'                             (constant literal)
    parts[-5] = <Cohort>                           (cohort directory name)

HISTORICAL BUG (fixed 2026-04-18):
    Prior version used parts[-4] for patient_id and parts[-3] for course_id,
    which made patient_id the literal string 'data' for every row and stored
    the true patient_id in course_id. This silently broke every patient-level
    grouping downstream (clustered bootstraps, patient-level random effects,
    LOCO-by-patient stratification). The correct indices are parts[-3] for
    patient_id and parts[-2] for course_id.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

COHORTS = {
    "Prostata":         {"region": "Pelvis", "site": "Lodz"},
    "Odbytnice":        {"region": "Pelvis", "site": "Lodz"},
    "Immunodozymetria": {"region": "Thorax", "site": "Gdansk+Lodz"},
    "PlucaRCHT":        {"region": "Thorax", "site": "Lodz"},
    "Hipokampy":        {"region": "Brain",  "site": "Lodz"},
    "GBM":              {"region": "Brain",  "site": "Lodz"},
    "LCTSC":            {"region": "Thorax", "site": "Multi-US"},
    "NSCLC_Interobserver": {"region": "Thorax", "site": "MAASTRO"},
    "NSCLC_Radiomics":  {"region": "Thorax", "site": "MAASTRO"},
    "RIDER":            {"region": "Thorax", "site": "MSKCC"},
}

DATA_ROOT = Path("/home/kgs24/rtpipeline_manuscript")
OUTPUT_DIR = Path("/home/kgs24/rtpipeline_manuscript/analysis/data")
PART_DIR = OUTPUT_DIR / "robustness_by_cohort"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PART_DIR.mkdir(parents=True, exist_ok=True)


def read_one(args):
    path, cohort, region, site = args
    try:
        df = pd.read_parquet(path)
        df["cohort"] = cohort
        df["body_region"] = region
        df["site"] = site
        parts = path.parts
        # Path convention: .../<Cohort>/data/<patient>/<course>/radiomics_robustness_ct.parquet
        # parts[-2] = course, parts[-3] = patient (NOT parts[-4], which is the literal 'data').
        df["patient_id"] = parts[-3]
        df["course_id"] = parts[-2]
        return df
    except Exception as e:
        return ("FAIL", str(path), str(e))


def process_cohort(cohort, info, parquets):
    t0 = time.time()
    tasks = [(p, cohort, info["region"], info["site"]) for p in parquets]
    out_path = PART_DIR / f"{cohort}.parquet"
    writer = None
    schema = None
    total_rows = 0
    fails = 0

    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, res in enumerate(ex.map(read_one, tasks)):
            if isinstance(res, tuple) and res[0] == "FAIL":
                fails += 1
                continue
            df = res
            total_rows += len(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema, compression="snappy")
            else:
                # Align schema in case of column-order differences
                table = table.select(schema.names).cast(schema)
            writer.write_table(table)
            del df, table
            if (i + 1) % 25 == 0:
                print(f"  {cohort}: wrote {i+1}/{len(tasks)} (rows={total_rows:,})", flush=True)

    if writer is not None:
        writer.close()

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9 if out_path.exists() else 0
    print(f"[{cohort}] wrote {total_rows:,} rows in {elapsed:.1f}s "
          f"({size_gb:.2f} GB, fails={fails})", flush=True)
    return {"cohort": cohort, "rows": total_rows, "size_gb": size_gb, "fails": fails}


def main():
    print("=== Step 1 v2: Aggregate robustness (streaming per-cohort) ===", flush=True)
    t_all = time.time()
    summary = []
    for cohort, info in COHORTS.items():
        base = DATA_ROOT / cohort / "data"
        if not base.exists():
            print(f"Skip {cohort}: no dir", flush=True)
            continue
        parquets = sorted(base.glob("*/*/radiomics_robustness_ct.parquet"))
        print(f"\n{cohort} ({info['region']}): {len(parquets)} parquets", flush=True)
        if not parquets:
            continue
        rec = process_cohort(cohort, info, parquets)
        summary.append(rec)

    total_time = time.time() - t_all
    total_rows = sum(r["rows"] for r in summary)
    total_size = sum(r["size_gb"] for r in summary)
    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE: {total_rows:,} rows across {len(summary)} cohorts "
          f"in {total_time:.0f}s ({total_size:.2f} GB total)", flush=True)
    for r in summary:
        print(f"  {r['cohort']}: {r['rows']:,} rows, {r['size_gb']:.2f} GB, fails={r['fails']}", flush=True)

    # Write summary manifest
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "robustness_by_cohort_summary.csv", index=False)
    print(f"\nManifest: {OUTPUT_DIR / 'robustness_by_cohort_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
