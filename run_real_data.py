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


def run_from_features(dataset_id: str):
    """Resume from saved topological_features.csv — skips all heavy computation."""
    features_path = _project_root / "data" / "results" / dataset_id / "topological_features.csv"
    if not features_path.exists():
        print(f"ERROR: No saved features at {features_path}")
        print("Run without --resume first to generate features.")
        sys.exit(1)

    print("=" * 60)
    print(f"TDA-CAR-T: Resume from features — {dataset_id}")
    print("=" * 60)

    feature_df = pd.read_csv(features_path)
    print(f"  Loaded {len(feature_df)} samples from {features_path}")

    # --- Classification ---
    labels = feature_df["label"].tolist()
    has_both = "responder" in labels and "non_responder" in labels

    if has_both and sum(l != "unknown" for l in labels) >= 4:
        print("\n  Running response classification...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        mask = feature_df["label"].isin(["responder", "non_responder"])
        labeled_df = feature_df[mask].copy()

        if len(labeled_df) >= 4:
            X = labeled_df.drop(columns=["sample_id", "patient_id", "label"], errors="ignore")
            X = X.select_dtypes(include=[np.number])
            X = X.loc[:, (X != 0).any(axis=0)].fillna(0)
            y = (labeled_df["label"] == "responder").astype(int)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring="accuracy")
            clf.fit(X, y)

            print(f"  LOO Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
            print(f"  Top features:")
            for feat, imp in sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])[:5]:
                print(f"    {feat}: {imp:.3f}")
    else:
        print("\n  Skipping classification (need both responders and non-responders).")

    # --- Statistical tests ---
    if has_both:
        from scipy.stats import mannwhitneyu, ttest_ind
        from itertools import compress

        r_mask = feature_df["label"] == "responder"
        nr_mask = feature_df["label"] == "non_responder"
        n_r = r_mask.sum()
        n_nr = nr_mask.sum()

        if r_mask.any() and nr_mask.any():
            # Test all topological features (H0 and H1)
            topo_cols = [c for c in feature_df.columns
                         if c.startswith("H0_") or c.startswith("H1_")]

            print(f"\n  STATISTICAL TESTS (n_responder={n_r}, n_non_responder={n_nr})")
            print(f"  {'Feature':<25} {'Resp mean':>10} {'NR mean':>10} {'Cohen d':>8} {'MW-U p':>10} {'t-test p':>10} {'Sig':>5}")
            print(f"  {'-'*80}")

            results = []
            for col in topo_cols:
                r_vals = feature_df.loc[r_mask, col].values.astype(float)
                nr_vals = feature_df.loc[nr_mask, col].values.astype(float)

                r_mean = np.mean(r_vals)
                nr_mean = np.mean(nr_vals)

                # Cohen's d (pooled std)
                pooled_std = np.sqrt(
                    ((len(r_vals) - 1) * np.var(r_vals, ddof=1) +
                     (len(nr_vals) - 1) * np.var(nr_vals, ddof=1))
                    / (len(r_vals) + len(nr_vals) - 2)
                )
                cohens_d = (r_mean - nr_mean) / pooled_std if pooled_std > 0 else 0.0

                # Mann-Whitney U (non-parametric, better for small n)
                try:
                    mw_stat, mw_p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
                except ValueError:
                    mw_p = 1.0

                # Welch's t-test
                try:
                    t_stat, t_p = ttest_ind(r_vals, nr_vals, equal_var=False)
                except Exception:
                    t_p = 1.0

                sig = ""
                if mw_p < 0.001:
                    sig = "***"
                elif mw_p < 0.01:
                    sig = "**"
                elif mw_p < 0.05:
                    sig = "*"
                elif mw_p < 0.10:
                    sig = "."

                results.append((col, r_mean, nr_mean, cohens_d, mw_p, t_p, sig))
                print(f"  {col:<25} {r_mean:>10.3f} {nr_mean:>10.3f} {cohens_d:>+8.3f} {mw_p:>10.4f} {t_p:>10.4f} {sig:>5}")

            # Multiple comparison correction (Benjamini-Hochberg)
            p_vals = [r[4] for r in results]
            n_tests = len(p_vals)
            sorted_idx = np.argsort(p_vals)
            bh_corrected = np.zeros(n_tests)
            for rank, idx in enumerate(sorted_idx, 1):
                bh_corrected[idx] = p_vals[idx] * n_tests / rank
            # Enforce monotonicity
            for i in range(n_tests - 2, -1, -1):
                bh_corrected[sorted_idx[i]] = min(
                    bh_corrected[sorted_idx[i]],
                    bh_corrected[sorted_idx[i + 1]] if i + 1 < n_tests else 1.0
                )
            bh_corrected = np.clip(bh_corrected, 0, 1)

            print(f"\n  AFTER BENJAMINI-HOCHBERG CORRECTION:")
            print(f"  {'Feature':<25} {'Raw p':>10} {'Adj p (BH)':>12} {'Sig':>5}")
            print(f"  {'-'*55}")
            for i, (col, r_mean, nr_mean, d, mw_p, t_p, _) in enumerate(results):
                adj_p = bh_corrected[i]
                sig = ""
                if adj_p < 0.001:
                    sig = "***"
                elif adj_p < 0.01:
                    sig = "**"
                elif adj_p < 0.05:
                    sig = "*"
                elif adj_p < 0.10:
                    sig = "."
                print(f"  {col:<25} {mw_p:>10.4f} {adj_p:>12.4f} {sig:>5}")

            # Summary
            sig_raw = sum(1 for r in results if r[4] < 0.05)
            sig_adj = sum(1 for p in bh_corrected if p < 0.05)
            large_d = sum(1 for r in results if abs(r[3]) >= 0.8)
            medium_d = sum(1 for r in results if 0.5 <= abs(r[3]) < 0.8)

            print(f"\n  SUMMARY:")
            print(f"    Features tested: {n_tests}")
            print(f"    Significant (raw p<0.05): {sig_raw}")
            print(f"    Significant (BH adj p<0.05): {sig_adj}")
            print(f"    Large effect size (|d|>=0.8): {large_d}")
            print(f"    Medium effect size (0.5<=|d|<0.8): {medium_d}")
            print(f"    Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10")

    print("\n" + "=" * 60)
    print("DONE (resumed from saved features)")
    print("=" * 60)


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
        # Inline classifier to avoid importing src.pipeline (which has heavy deps)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        def _classify_response(df):
            X = df.drop(columns=["sample_id", "patient_id", "label"], errors="ignore")
            X = X.select_dtypes(include=[np.number])
            X = X.loc[:, (X != 0).any(axis=0)].fillna(0)
            y = (df["label"] == "responder").astype(int)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring="accuracy")
            clf.fit(X, y)
            return {
                "accuracy": float(scores.mean()),
                "accuracy_std": float(scores.std()),
                "feature_importances": dict(zip(X.columns, clf.feature_importances_)),
                "n_samples": len(y),
            }

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
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved features (skip data loading + TDA)")
    parser.add_argument("--info", action="store_true",
                        help="Print instructions for getting data")
    args = parser.parse_args()

    if args.info:
        print_instructions()
    elif args.resume:
        run_from_features(args.dataset)
    else:
        run_real_analysis(args.dataset, download=args.download)
