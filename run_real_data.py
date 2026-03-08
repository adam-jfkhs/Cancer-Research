"""
Run the full TDA-CAR-T pipeline on REAL clinical data.

Usage:
    # Step 1: Download data (one-time)
    python -m src.data.geo_download --dataset GSE197268

    # Step 2: Run analysis
    python run_real_data.py --dataset GSE197268

    # Or run the whole thing in one shot:
    python run_real_data.py --dataset GSE197268 --download
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))

from src.data.geo_download import (
    DATASETS,
    download_geo_metadata,
    download_geo_supplementary,
    load_real_dataset,
    parse_response_labels,
    print_instructions,
)
from src.preprocessing.single_cell import (
    load_cart_dataset,
    extract_exhaustion_signature,
    compute_embedding,
    prepare_for_tda,
)
from src.tda.persistent_homology import (
    compute_persistence,
    persistence_statistics,
    topological_feature_matrix,
)
from src.mapper.cart_mapper import (
    build_mapper_graph,
    detect_topological_features,
)


def run_real_analysis(dataset_id: str, download: bool = False):
    """Full pipeline: download → preprocess → TDA → classify → visualize.

    Parameters
    ----------
    dataset_id : str
        GEO accession (e.g. 'GSE197268')
    download : bool
        If True, download data first.
    """
    print("=" * 60)
    print(f"TDA-CAR-T: Real Data Analysis — {dataset_id}")
    print(f"  {DATASETS.get(dataset_id, {}).get('paper', 'Unknown dataset')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Download (if requested)
    # ------------------------------------------------------------------
    if download:
        print("\n[1/6] Downloading from GEO...")
        meta = download_geo_metadata(dataset_id)
        download_geo_supplementary(dataset_id)
    else:
        print("\n[1/6] Loading cached metadata...")
        meta = download_geo_metadata(dataset_id)

    # ------------------------------------------------------------------
    # 2. Parse clinical labels
    # ------------------------------------------------------------------
    print("\n[2/6] Parsing clinical response labels...")
    response_labels = parse_response_labels(meta, dataset_id)

    n_r = sum(1 for v in response_labels.values() if v == "responder")
    n_nr = sum(1 for v in response_labels.values() if v == "non_responder")
    print(f"  Found {n_r} responders, {n_nr} non-responders")

    if n_r == 0 or n_nr == 0:
        print("\n  WARNING: Need both responders and non-responders for comparison.")
        print("  Check data/raw/{dataset_id}/clinical_metadata.csv and annotate manually.")
        print("  Column needed: 'response' with values 'CR'/'PR' or 'NR'/'PD'")

    # ------------------------------------------------------------------
    # 3. Load and preprocess scRNA-seq
    # ------------------------------------------------------------------
    print("\n[3/6] Loading scRNA-seq data...")
    adata = load_real_dataset(dataset_id)

    print("  Running QC and normalization...")
    adata = load_cart_dataset(adata)
    print(f"  After QC: {adata.shape[0]} cells × {adata.shape[1]} genes")

    print("  Computing embeddings (PCA + UMAP)...")
    adata = compute_embedding(adata)

    print("  Scoring exhaustion signature...")
    try:
        exhaustion = extract_exhaustion_signature(adata)
        print(f"  Exhaustion score range: [{exhaustion.min():.2f}, {exhaustion.max():.2f}]")
    except ValueError as e:
        print(f"  Warning: {e}")
        print("  Continuing without exhaustion scores...")

    # ------------------------------------------------------------------
    # 4. Per-patient TDA (group by PATIENT, not by GSM sample)
    # ------------------------------------------------------------------
    print("\n[4/6] Computing persistent homology per patient...")

    # For GSE197268: group by patient_id (each patient has multiple timepoints)
    # We use infusion product samples for TDA (the CAR-T cells before infusion)
    if "sample_id" not in adata.obs.columns:
        print("  Warning: No sample_id found. Treating all cells as one sample.")
        adata.obs["sample_id"] = "all_cells"

    # Build mapping: GSM → patient_id and timepoint from metadata
    gsm_to_patient = {}
    gsm_to_timepoint = {}
    if "patient_id" in meta.columns:
        for sid, row in meta.iterrows():
            gsm_to_patient[sid] = row.get("patient_id", sid)
            gsm_to_timepoint[sid] = row.get("timepoint", "unknown")

    # Prefer infusion product samples (the CAR-T cells), then fall back to d7
    # Group samples by patient
    patient_samples = {}
    for sid in adata.obs["sample_id"].unique():
        patient = gsm_to_patient.get(sid, sid)
        tp = gsm_to_timepoint.get(sid, "unknown")
        if patient not in patient_samples:
            patient_samples[patient] = []
        patient_samples[patient].append((sid, tp))

    print(f"  Found {len(patient_samples)} patients with data")

    # For each patient, pick the best sample (prefer infusion > d7 > baseline)
    TIMEPOINT_PRIORITY = {"infusion": 0, "d7_cart": 1, "d7": 2, "baseline": 3, "d14": 4, "retreatment": 5, "unknown": 6}

    point_clouds = []
    labels = []
    sample_names = []
    patient_ids = []

    for patient, samples in sorted(patient_samples.items()):
        # Sort by priority: infusion first
        samples.sort(key=lambda x: TIMEPOINT_PRIORITY.get(x[1], 99))
        best_gsm, best_tp = samples[0]

        mask = adata.obs["sample_id"] == best_gsm
        adata_sub = adata[mask]

        if adata_sub.shape[0] < 50:
            print(f"  Skipping {patient} ({best_tp}): only {adata_sub.shape[0]} cells")
            continue

        # Extract point cloud from PCA
        pc = adata_sub.obsm["X_pca"][:, :20]
        point_clouds.append(pc)

        # Get response label for this patient
        from src.data.geo_download import GSE197268_PATIENT_RESPONSE
        label = GSE197268_PATIENT_RESPONSE.get(patient, response_labels.get(best_gsm, "unknown"))
        labels.append(label)
        sample_names.append(best_gsm)
        patient_ids.append(patient)
        print(f"  {patient} ({best_tp}, {best_gsm}): {adata_sub.shape[0]} cells → {label}")

    # ------------------------------------------------------------------
    # 5. Topological feature extraction + classification
    # ------------------------------------------------------------------
    print(f"\n[5/6] Extracting topological features from {len(point_clouds)} samples...")

    feature_df = topological_feature_matrix(
        point_clouds, labels=labels, maxdim=1
    )
    feature_df["sample_id"] = sample_names
    feature_df["patient_id"] = patient_ids

    # Save features
    output_dir = _project_root / "data" / "results" / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(output_dir / "topological_features.csv", index=False)
    print(f"  Saved: {output_dir / 'topological_features.csv'}")

    # Classification (if we have both groups)
    has_both = "responder" in labels and "non_responder" in labels
    if has_both and sum(l != "unknown" for l in labels) >= 4:
        print("\n  Running response classification...")
        from src.pipeline import _classify_response
        # Filter to labeled samples only
        mask = feature_df["label"].isin(["responder", "non_responder"])
        labeled_df = feature_df[mask].copy()

        if len(labeled_df) >= 4:
            result = _classify_response(labeled_df)
            print(f"  LOO Accuracy: {result['accuracy']:.2f} (+/- {result['accuracy_std']:.2f})")
            print(f"  Top features:")
            for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1])[:5]:
                print(f"    {feat}: {imp:.3f}")
        else:
            print("  Not enough labeled samples for classification.")
            result = None
    else:
        print("\n  Skipping classification (need both responders and non-responders).")
        result = None

    # ------------------------------------------------------------------
    # 6. Mapper analysis on representative samples
    # ------------------------------------------------------------------
    print("\n[6/6] Building Mapper graphs...")

    mapper_results = {}
    for group in ["responder", "non_responder"]:
        indices = [i for i, l in enumerate(labels) if l == group]
        if not indices:
            continue
        # Use the sample with most cells
        best_idx = max(indices, key=lambda i: len(point_clouds[i]))
        pc = point_clouds[best_idx]

        print(f"  {group} representative: {sample_names[best_idx]} ({len(pc)} cells)")
        graph = build_mapper_graph(pc)
        features = detect_topological_features(graph)
        mapper_results[group] = {
            "sample": sample_names[best_idx],
            "graph": graph,
            "features": features,
        }
        print(f"    Nodes: {len(graph['nodes'])}, "
              f"Branches: {features.get('branch_count', 0)}, "
              f"Loops: {features.get('loop_count', 0)}, "
              f"Components: {features.get('component_count', 0)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Dataset: {dataset_id}")
    print(f"  Samples analyzed: {len(point_clouds)}")
    print(f"  Results saved to: {output_dir}")

    if has_both:
        # Compare topological features between groups
        r_mask = feature_df["label"] == "responder"
        nr_mask = feature_df["label"] == "non_responder"

        if r_mask.any() and nr_mask.any():
            h1_cols = [c for c in feature_df.columns if "H1" in c]
            print("\n  TOPOLOGICAL COMPARISON (Responder vs Non-Responder):")
            print(f"  {'Feature':<30} {'Responder':>12} {'Non-Resp':>12} {'Diff':>8}")
            print(f"  {'-'*62}")
            for col in h1_cols:
                r_val = feature_df.loc[r_mask, col].mean()
                nr_val = feature_df.loc[nr_mask, col].mean()
                diff = r_val - nr_val
                print(f"  {col:<30} {r_val:>12.3f} {nr_val:>12.3f} {diff:>+8.3f}")

    print(f"""
NEXT STEPS:
  1. Open topological_features.csv to inspect per-patient features
  2. Run 'python run_visualizations.py' to generate 3D plots
  3. Compare H1 features between responders and non-responders
  4. The key finding: responders should show MORE H1 persistence (loops)
     than non-responders (fragmented exhaustion)
""")

    return {
        "feature_df": feature_df,
        "mapper_results": mapper_results,
        "classification": result,
        "adata": adata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TDA-CAR-T on real clinical data")
    parser.add_argument("--dataset", type=str, default="GSE197268",
                        help="GEO accession (default: GSE197268)")
    parser.add_argument("--download", action="store_true",
                        help="Download data before analysis")
    parser.add_argument("--info", action="store_true",
                        help="Print instructions for getting data")
    args = parser.parse_args()

    if args.info:
        print_instructions()
    else:
        run_real_analysis(args.dataset, download=args.download)
