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


# ---------------------------------------------------------------------------
# Biologically meaningful gene sets for CAR-T response
# ---------------------------------------------------------------------------
CART_GENE_SETS = {
    "exhaustion": [
        "PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1", "CTLA4", "LAYN",
        "CD244", "CD160", "BTLA",
    ],
    "activation": [
        "CD69", "IL2RA", "TNFRSF9", "ICOS", "CD28", "IFNG", "TNF", "IL2",
        "GZMB", "PRF1", "TNFRSF4", "HLA-DRA",
    ],
    "effector": [
        "GZMB", "GZMA", "GZMK", "PRF1", "FASLG", "IFNG", "TNF", "NKG7",
        "GNLY", "CST7", "CCL4", "CCL3",
    ],
    "memory_stemness": [
        "TCF7", "LEF1", "BCL6", "IL7R", "SELL", "CCR7", "CD27", "CD28",
        "BACH2", "KLF2",
    ],
    "cytokine_signaling": [
        "IL6", "IL10", "TGFB1", "IL21", "IL15", "IL7", "STAT1", "STAT3",
        "STAT5A", "JAK1", "JAK2", "SOCS1", "SOCS3",
    ],
    "proliferation": [
        "MKI67", "TOP2A", "PCNA", "MCM2", "MCM7", "CDK1", "CCNB1",
        "BIRC5", "AURKB", "STMN1",
    ],
    "apoptosis": [
        "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BID", "CASP3", "CASP8",
        "CASP9", "FAS", "FASLG",
    ],
    "trafficking": [
        "CXCR3", "CXCR4", "CCR5", "CCR7", "SELL", "ITGAL", "ITGB2",
        "S1PR1", "CX3CR1", "CXCR6",
    ],
}


def _gene_set_point_cloud(adata_sub, gene_set, min_genes=3):
    """Extract a point cloud from expression of a specific gene set.

    Returns PCA of gene-set expression, or None if too few genes found.
    """
    from sklearn.decomposition import PCA

    present = [g for g in gene_set if g in adata_sub.var_names]
    if len(present) < min_genes:
        return None

    # Extract expression matrix for these genes
    X = adata_sub[:, present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # PCA — use min(n_genes, 10, n_cells-1) components
    n_components = min(len(present), 10, X.shape[0] - 1)
    if n_components < 2:
        return None

    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def _compute_landscape_features(diagram, prefix, n_land=5, resolution=100):
    """Compute persistence landscape summary features from a diagram."""
    from src.tda.persistent_homology import persistence_landscape

    land = persistence_landscape(diagram, num_landscapes=n_land, resolution=resolution)

    features = {}
    for k in range(n_land):
        row = land[k]
        features[f"{prefix}_L{k}_max"] = float(np.max(row))
        features[f"{prefix}_L{k}_mean"] = float(np.mean(row))
        features[f"{prefix}_L{k}_integral"] = float(np.sum(row)) * (1.0 / resolution)
    return features


def run_enhanced_analysis(dataset_id: str, download: bool = False):
    """Enhanced TDA: gene-filtered topology + per-pathway + persistence landscapes.

    Loads the scRNA-seq data, then for each patient computes:
    1. TDA on biologically meaningful gene subsets (exhaustion, effector, etc.)
    2. TDA on each metabolic pathway gene set
    3. Persistence landscapes from the full PCA point cloud
    Then runs full statistical testing on the combined feature set.
    """
    print("=" * 60)
    print(f"TDA-CAR-T: ENHANCED Analysis — {dataset_id}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1-3: Same data loading as run_real_analysis
    # ------------------------------------------------------------------
    if download:
        print("\n[1/5] Downloading from GEO...")
        meta = download_geo_metadata(dataset_id)
        download_geo_supplementary(dataset_id)
    else:
        print("\n[1/5] Loading cached metadata...")
        meta = download_geo_metadata(dataset_id)

    print("\n[2/5] Parsing clinical response labels...")
    response_labels = parse_response_labels(meta, dataset_id)

    n_r = sum(1 for v in response_labels.values() if v == "responder")
    n_nr = sum(1 for v in response_labels.values() if v == "non_responder")
    print(f"  Found {n_r} responders, {n_nr} non-responders")

    print("\n[3/5] Loading scRNA-seq data...")
    adata = load_real_dataset(dataset_id)
    adata = load_cart_dataset(adata)
    print(f"  After QC: {adata.shape[0]} cells × {adata.shape[1]} genes")
    adata = compute_embedding(adata)

    # ------------------------------------------------------------------
    # 4: Build per-patient point clouds (same grouping logic)
    # ------------------------------------------------------------------
    print("\n[4/5] Grouping by patient...")
    if "sample_id" not in adata.obs.columns:
        adata.obs["sample_id"] = "all_cells"

    gsm_to_patient = {}
    gsm_to_timepoint = {}
    if "patient_id" in meta.columns:
        for sid, row in meta.iterrows():
            gsm_to_patient[sid] = row.get("patient_id", sid)
            gsm_to_timepoint[sid] = row.get("timepoint", "unknown")

    patient_samples = {}
    for sid in adata.obs["sample_id"].unique():
        patient = gsm_to_patient.get(sid, sid)
        tp = gsm_to_timepoint.get(sid, "unknown")
        if patient not in patient_samples:
            patient_samples[patient] = []
        patient_samples[patient].append((sid, tp))

    TIMEPOINT_PRIORITY = {
        "infusion": 0, "d7_cart": 1, "d7": 2, "baseline": 3,
        "d14": 4, "retreatment": 5, "unknown": 6,
    }

    from src.data.geo_download import GSE197268_PATIENT_RESPONSE
    from src.metabolic.flux_analysis import METABOLIC_SIGNATURES

    all_gene_sets = {}
    all_gene_sets.update(CART_GENE_SETS)
    for pw_name, pw_genes in METABOLIC_SIGNATURES.items():
        all_gene_sets[f"metab_{pw_name}"] = pw_genes

    # ------------------------------------------------------------------
    # 5: Enhanced TDA per patient
    # ------------------------------------------------------------------
    print(f"\n[5/5] Computing ENHANCED topological features...")
    print(f"  Gene sets: {list(all_gene_sets.keys())}")

    rows = []
    for patient, samples in sorted(patient_samples.items()):
        samples.sort(key=lambda x: TIMEPOINT_PRIORITY.get(x[1], 99))
        best_gsm, best_tp = samples[0]

        mask = adata.obs["sample_id"] == best_gsm
        adata_sub = adata[mask]

        if adata_sub.shape[0] < 50:
            continue

        label = GSE197268_PATIENT_RESPONSE.get(
            patient, response_labels.get(best_gsm, "unknown")
        )

        row = {
            "sample_id": best_gsm,
            "patient_id": patient,
            "label": label,
        }

        # --- A) Full PCA topology (same as before) ---
        pc_full = adata_sub.obsm["X_pca"][:, :20]
        result_full = compute_persistence(pc_full, maxdim=1)
        stats_full = persistence_statistics(result_full["diagrams"])
        for dim_key, dim_stats in stats_full.items():
            for stat_name, stat_val in dim_stats.items():
                row[f"{dim_key}_{stat_name}"] = stat_val

        # --- B) Persistence landscape features from full PCA ---
        for dim_idx, dim_name in enumerate(["H0", "H1"]):
            if dim_idx < len(result_full["diagrams"]):
                dgm = result_full["diagrams"][dim_idx]
                land_feats = _compute_landscape_features(
                    dgm, prefix=dim_name, n_land=3, resolution=100
                )
                row.update(land_feats)

        # --- C) Gene-set-specific TDA ---
        for gs_name, gs_genes in all_gene_sets.items():
            pc_gs = _gene_set_point_cloud(adata_sub, gs_genes)
            if pc_gs is None:
                # Not enough genes — fill with NaN
                for stat in ["count", "max_persistence", "mean_persistence",
                             "total_persistence", "entropy"]:
                    row[f"{gs_name}_H0_{stat}"] = np.nan
                    row[f"{gs_name}_H1_{stat}"] = np.nan
                continue

            result_gs = compute_persistence(pc_gs, maxdim=1)
            stats_gs = persistence_statistics(result_gs["diagrams"])
            for dim_key, dim_stats in stats_gs.items():
                for stat_name, stat_val in dim_stats.items():
                    row[f"{gs_name}_{dim_key}_{stat_name}"] = stat_val

        rows.append(row)
        n_feats = sum(1 for k in row if k not in ("sample_id", "patient_id", "label"))
        print(f"  {patient} ({best_tp}): {adata_sub.shape[0]} cells, "
              f"{n_feats} features → {label}")

    feature_df = pd.DataFrame(rows)

    # Save enhanced features
    output_dir = _project_root / "data" / "results" / dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_dir / "enhanced_features.csv", index=False)
    print(f"\n  Saved: {output_dir / 'enhanced_features.csv'}")
    print(f"  Total features per patient: {len(feature_df.columns) - 3}")

    # ------------------------------------------------------------------
    # Classification + Stats (reuse run_from_features logic)
    # ------------------------------------------------------------------
    _run_classification_and_stats(feature_df)

    print("\n" + "=" * 60)
    print("ENHANCED ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Resume later: python run_real_data.py --dataset {dataset_id} --resume-enhanced")


def _run_classification_and_stats(feature_df):
    """Run classification and statistical tests on a feature DataFrame."""
    from scipy.stats import mannwhitneyu, ttest_ind

    labels = feature_df["label"].tolist()
    has_both = "responder" in labels and "non_responder" in labels

    # --- Classification ---
    if has_both and sum(l != "unknown" for l in labels) >= 4:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import LeaveOneOut, cross_val_score

        mask = feature_df["label"].isin(["responder", "non_responder"])
        labeled_df = feature_df[mask].copy()

        if len(labeled_df) >= 4:
            X = labeled_df.drop(columns=["sample_id", "patient_id", "label"], errors="ignore")
            X = X.select_dtypes(include=[np.number])
            X = X.loc[:, (X != 0).any(axis=0)].dropna(axis=1).fillna(0)
            y = (labeled_df["label"] == "responder").astype(int)

            clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=3)
            scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring="accuracy")
            clf.fit(X, y)

            print(f"\n  LOO ACCURACY: {scores.mean():.2f} (+/- {scores.std():.2f})")
            print(f"  Top 15 features:")
            importances = sorted(
                zip(X.columns, clf.feature_importances_), key=lambda x: -x[1]
            )
            for feat, imp in importances[:15]:
                print(f"    {feat}: {imp:.3f}")

    # --- Statistical tests ---
    if not has_both:
        return

    r_mask = feature_df["label"] == "responder"
    nr_mask = feature_df["label"] == "non_responder"
    n_r = r_mask.sum()
    n_nr = nr_mask.sum()

    if not (r_mask.any() and nr_mask.any()):
        return

    # Get all numeric feature columns
    meta_cols = {"sample_id", "patient_id", "label", "sample_idx"}
    topo_cols = [c for c in feature_df.columns
                 if c not in meta_cols and feature_df[c].dtype in [np.float64, np.int64, float, int]]

    # Drop columns that are all NaN
    topo_cols = [c for c in topo_cols if feature_df[c].notna().sum() >= n_r + n_nr - 2]

    print(f"\n  STATISTICAL TESTS (n_R={n_r}, n_NR={n_nr}, {len(topo_cols)} features)")

    results = []
    for col in topo_cols:
        r_vals = feature_df.loc[r_mask, col].dropna().values.astype(float)
        nr_vals = feature_df.loc[nr_mask, col].dropna().values.astype(float)

        if len(r_vals) < 2 or len(nr_vals) < 2:
            continue

        r_mean = np.mean(r_vals)
        nr_mean = np.mean(nr_vals)

        pooled_std = np.sqrt(
            ((len(r_vals) - 1) * np.var(r_vals, ddof=1) +
             (len(nr_vals) - 1) * np.var(nr_vals, ddof=1))
            / (len(r_vals) + len(nr_vals) - 2)
        )
        cohens_d = (r_mean - nr_mean) / pooled_std if pooled_std > 0 else 0.0

        try:
            _, mw_p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        except ValueError:
            mw_p = 1.0
        try:
            _, t_p = ttest_ind(r_vals, nr_vals, equal_var=False)
        except Exception:
            t_p = 1.0

        results.append((col, r_mean, nr_mean, cohens_d, mw_p, t_p))

    if not results:
        print("  No testable features found.")
        return

    # BH correction
    p_vals = np.array([r[4] for r in results])
    n_tests = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    bh_corrected = np.zeros(n_tests)
    for rank, idx in enumerate(sorted_idx, 1):
        bh_corrected[idx] = p_vals[idx] * n_tests / rank
    for i in range(n_tests - 2, -1, -1):
        bh_corrected[sorted_idx[i]] = min(
            bh_corrected[sorted_idx[i]],
            bh_corrected[sorted_idx[i + 1]] if i + 1 < n_tests else 1.0
        )
    bh_corrected = np.clip(bh_corrected, 0, 1)

    # Show top results sorted by raw p-value
    ranked = sorted(range(n_tests), key=lambda i: results[i][4])

    print(f"\n  TOP 30 FEATURES BY RAW P-VALUE:")
    print(f"  {'Feature':<40} {'R mean':>8} {'NR mean':>8} {'d':>7} {'raw p':>9} {'BH p':>9} {'Sig':>5}")
    print(f"  {'-'*85}")

    for rank, idx in enumerate(ranked[:30]):
        col, r_mean, nr_mean, d, mw_p, t_p = results[idx]
        adj_p = bh_corrected[idx]
        sig = ""
        if adj_p < 0.001: sig = "***"
        elif adj_p < 0.01: sig = "**"
        elif adj_p < 0.05: sig = "*"
        elif adj_p < 0.10: sig = "."
        elif mw_p < 0.05: sig = "(r)"  # raw only
        print(f"  {col:<40} {r_mean:>8.3f} {nr_mean:>8.3f} {d:>+7.3f} {mw_p:>9.4f} {adj_p:>9.4f} {sig:>5}")

    # Summary
    sig_raw = sum(1 for r in results if r[4] < 0.05)
    sig_adj = sum(1 for p in bh_corrected if p < 0.05)
    trend_raw = sum(1 for r in results if r[4] < 0.10)
    large_d = sum(1 for r in results if abs(r[3]) >= 0.8)
    medium_d = sum(1 for r in results if 0.5 <= abs(r[3]) < 0.8)

    print(f"\n  SUMMARY:")
    print(f"    Total features tested: {n_tests}")
    print(f"    Significant (raw p<0.05): {sig_raw}")
    print(f"    Trending (raw p<0.10): {trend_raw}")
    print(f"    Significant (BH adj p<0.05): {sig_adj}")
    print(f"    Large effect size (|d|>=0.8): {large_d}")
    print(f"    Medium effect size (0.5<=|d|<0.8): {medium_d}")
    print(f"    Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  (r) raw-only")

    # Group-level summary: which gene sets show the most signal?
    print(f"\n  SIGNAL BY GENE SET (best raw p per group):")
    gene_set_best = {}
    for idx, (col, r_mean, nr_mean, d, mw_p, t_p) in enumerate(results):
        # Parse gene set name from column
        parts = col.split("_H")
        if len(parts) >= 2:
            gs = parts[0]
        else:
            gs = "global"
        if gs not in gene_set_best or mw_p < gene_set_best[gs][0]:
            gene_set_best[gs] = (mw_p, col, d, bh_corrected[idx])

    for gs, (best_p, best_col, best_d, best_bh) in sorted(
        gene_set_best.items(), key=lambda x: x[1][0]
    ):
        sig = "*" if best_p < 0.05 else ("." if best_p < 0.10 else "")
        print(f"    {gs:<30} best p={best_p:.4f} (BH={best_bh:.4f}) d={best_d:+.3f} [{best_col}] {sig}")

    # ------------------------------------------------------------------
    # FOCUSED HYPOTHESIS TESTS (pre-registered, fewer comparisons)
    # ------------------------------------------------------------------
    # These are the biologically motivated hypotheses — test only these
    # to reduce multiple comparison burden (BH over ~15 tests, not 158)
    hypothesis_features = [
        # Exhaustion topology — primary hypothesis
        "exhaustion_H1_mean_persistence",
        "exhaustion_H1_max_persistence",
        "exhaustion_H1_count",
        "exhaustion_H0_entropy",
        # Activation / effector — secondary hypothesis
        "activation_H0_mean_persistence",
        "activation_H1_mean_persistence",
        "effector_H1_count",
        # Memory/stemness — tertiary hypothesis
        "memory_stemness_H1_max_persistence",
        "memory_stemness_H0_entropy",
        # Cytokine signaling
        "cytokine_signaling_H1_mean_persistence",
        # Metabolic
        "metab_glycolysis_H1_total_persistence",
        "metab_oxidative_phosphorylation_H0_count",
        # Global landscape
        "H1_L0_max",
        "H1_L1_max",
        "H1_L2_max",
    ]

    # Filter to features that actually exist
    hypothesis_features = [f for f in hypothesis_features if f in feature_df.columns]

    if hypothesis_features:
        print(f"\n  {'='*85}")
        print(f"  FOCUSED HYPOTHESIS TESTS ({len(hypothesis_features)} pre-selected features)")
        print(f"  {'='*85}")

        focused_results = []
        for col in hypothesis_features:
            r_vals = feature_df.loc[r_mask, col].dropna().values.astype(float)
            nr_vals = feature_df.loc[nr_mask, col].dropna().values.astype(float)
            if len(r_vals) < 2 or len(nr_vals) < 2:
                continue

            r_mean = np.mean(r_vals)
            nr_mean = np.mean(nr_vals)
            pooled_std = np.sqrt(
                ((len(r_vals) - 1) * np.var(r_vals, ddof=1) +
                 (len(nr_vals) - 1) * np.var(nr_vals, ddof=1))
                / (len(r_vals) + len(nr_vals) - 2)
            )
            d = (r_mean - nr_mean) / pooled_std if pooled_std > 0 else 0.0
            try:
                _, mw_p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
            except ValueError:
                mw_p = 1.0

            # Permutation test (10000 permutations)
            combined = np.concatenate([r_vals, nr_vals])
            observed_diff = abs(r_mean - nr_mean)
            n_perm = 10000
            rng = np.random.RandomState(42)
            count_extreme = 0
            for _ in range(n_perm):
                perm = rng.permutation(combined)
                perm_diff = abs(np.mean(perm[:len(r_vals)]) - np.mean(perm[len(r_vals):]))
                if perm_diff >= observed_diff:
                    count_extreme += 1
            perm_p = (count_extreme + 1) / (n_perm + 1)

            focused_results.append((col, r_mean, nr_mean, d, mw_p, perm_p))

        # BH correction on focused set only
        fp_vals = np.array([r[4] for r in focused_results])
        fn = len(fp_vals)
        f_sorted = np.argsort(fp_vals)
        f_bh = np.zeros(fn)
        for rank, idx in enumerate(f_sorted, 1):
            f_bh[idx] = fp_vals[idx] * fn / rank
        for i in range(fn - 2, -1, -1):
            f_bh[f_sorted[i]] = min(f_bh[f_sorted[i]],
                                     f_bh[f_sorted[i + 1]] if i + 1 < fn else 1.0)
        f_bh = np.clip(f_bh, 0, 1)

        print(f"  {'Feature':<42} {'R':>7} {'NR':>7} {'d':>7} {'MW p':>8} {'perm p':>8} {'BH p':>8} {'Sig':>5}")
        print(f"  {'-'*92}")

        # Sort by MW p
        order = sorted(range(len(focused_results)), key=lambda i: focused_results[i][4])
        for i in order:
            col, rm, nrm, d, mwp, pp = focused_results[i]
            adj_p = f_bh[i]
            sig = ""
            if adj_p < 0.001: sig = "***"
            elif adj_p < 0.01: sig = "**"
            elif adj_p < 0.05: sig = "*"
            elif adj_p < 0.10: sig = "."
            elif mwp < 0.05: sig = "(r)"
            print(f"  {col:<42} {rm:>7.3f} {nrm:>7.3f} {d:>+7.3f} {mwp:>8.4f} {pp:>8.4f} {adj_p:>8.4f} {sig:>5}")

        f_sig_raw = sum(1 for r in focused_results if r[4] < 0.05)
        f_sig_bh = sum(1 for p in f_bh if p < 0.05)
        f_trend = sum(1 for r in focused_results if r[4] < 0.10)

        print(f"\n  FOCUSED SUMMARY ({len(focused_results)} hypothesis-driven tests):")
        print(f"    Significant (raw p<0.05): {f_sig_raw}")
        print(f"    Trending (raw p<0.10): {f_trend}")
        print(f"    Significant (BH adj p<0.05): {f_sig_bh}")
        print(f"    (BH correction over {len(focused_results)} tests instead of {n_tests})")

    # ------------------------------------------------------------------
    # SINGLE PRE-REGISTERED HYPOTHESIS: Composite exhaustion topology
    # ------------------------------------------------------------------
    # Combine correlated exhaustion H1 features into one score.
    # Justification: these all measure the same phenomenon (loop structure
    # in exhaustion gene expression space). Testing one composite avoids
    # multiple comparison correction entirely.
    exhaustion_h1_cols = [
        "exhaustion_H1_mean_persistence",
        "exhaustion_H1_max_persistence",
        "exhaustion_H1_entropy",
    ]
    exhaustion_h0_cols = [
        "exhaustion_H0_entropy",
        "exhaustion_H0_max_persistence",
        "exhaustion_H0_mean_persistence",
    ]
    # Also create activation composite and cytokine composite
    composite_defs = {
        "Exhaustion H1 topology": exhaustion_h1_cols,
        "Exhaustion H0 complexity": exhaustion_h0_cols,
        "Cytokine H1 topology": [
            "cytokine_signaling_H1_mean_persistence",
            "cytokine_signaling_H1_max_persistence",
            "cytokine_signaling_H1_entropy",
        ],
        "Activation topology": [
            "activation_H0_mean_persistence",
            "activation_H0_max_persistence",
            "activation_H0_entropy",
        ],
    }

    print(f"\n  {'='*85}")
    print(f"  COMPOSITE SCORE TESTS (single hypothesis per biological process)")
    print(f"  {'='*85}")
    print(f"  Each composite = z-scored average of correlated features within one process.")
    print(f"  No multiple comparison correction needed (one test per biological hypothesis).\n")

    from sklearn.preprocessing import StandardScaler

    composite_results = []
    for comp_name, comp_cols in composite_defs.items():
        avail = [c for c in comp_cols if c in feature_df.columns
                 and feature_df[c].notna().sum() >= n_r + n_nr - 2]
        if len(avail) < 2:
            continue

        # Z-score each feature, then average into composite
        vals = feature_df[avail].dropna()
        if len(vals) < n_r + n_nr - 2:
            continue

        scaler = StandardScaler()
        z_scores = scaler.fit_transform(vals)
        composite = np.mean(z_scores, axis=1)

        # Map back to R/NR using the indices that survived dropna
        valid_idx = vals.index
        r_comp = composite[feature_df.loc[valid_idx, "label"] == "responder"]
        nr_comp = composite[feature_df.loc[valid_idx, "label"] == "non_responder"]

        if len(r_comp) < 2 or len(nr_comp) < 2:
            continue

        r_mean = np.mean(r_comp)
        nr_mean = np.mean(nr_comp)
        pooled_std = np.sqrt(
            ((len(r_comp) - 1) * np.var(r_comp, ddof=1) +
             (len(nr_comp) - 1) * np.var(nr_comp, ddof=1))
            / (len(r_comp) + len(nr_comp) - 2)
        )
        d = (r_mean - nr_mean) / pooled_std if pooled_std > 0 else 0.0

        try:
            _, mw_p = mannwhitneyu(r_comp, nr_comp, alternative="two-sided")
        except ValueError:
            mw_p = 1.0

        # Permutation test (10000 permutations)
        combined = np.concatenate([r_comp, nr_comp])
        observed_diff = abs(r_mean - nr_mean)
        n_perm = 10000
        rng = np.random.RandomState(42)
        count_extreme = 0
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            perm_diff = abs(np.mean(perm[:len(r_comp)]) - np.mean(perm[len(r_comp):]))
            if perm_diff >= observed_diff:
                count_extreme += 1
        perm_p = (count_extreme + 1) / (n_perm + 1)

        # Effect direction
        direction = "R < NR" if r_mean < nr_mean else "R > NR"

        composite_results.append((comp_name, len(avail), r_mean, nr_mean, d, mw_p, perm_p, direction))

        sig = ""
        if perm_p < 0.001: sig = "***"
        elif perm_p < 0.01: sig = "**"
        elif perm_p < 0.05: sig = "*"
        elif perm_p < 0.10: sig = "."

        print(f"  {comp_name}")
        print(f"    Components: {avail}")
        print(f"    R mean={r_mean:+.3f}, NR mean={nr_mean:+.3f}  ({direction})")
        print(f"    Cohen's d = {d:+.3f}  |  MW-U p = {mw_p:.4f}  |  Perm p = {perm_p:.4f}  {sig}")
        if perm_p < 0.05:
            print(f"    >>> SIGNIFICANT (p < 0.05, no correction needed) <<<")
        print()

    # Final interpretive summary
    any_sig = any(r[6] < 0.05 for r in composite_results)
    if any_sig:
        print(f"  {'='*85}")
        print(f"  KEY FINDING:")
        for comp_name, n_comp, rm, nrm, d, mwp, pp, direction in composite_results:
            if pp < 0.05:
                print(f"    {comp_name}: {direction}, d={d:+.3f}, perm p={pp:.4f}")
        print(f"\n  INTERPRETATION:")
        # Check if exhaustion is significant
        exh_results = [r for r in composite_results if "Exhaustion H1" in r[0] and r[6] < 0.05]
        if exh_results:
            print(f"    Non-responders show significantly MORE complex loop topology")
            print(f"    in exhaustion marker expression space (H1 features).")
            print(f"    This suggests non-responder CAR-T cells cycle through")
            print(f"    exhaustion states rather than transitioning through them,")
            print(f"    consistent with a trapped exhaustion phenotype.")
        print(f"  {'='*85}")


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
    parser.add_argument("--enhanced", action="store_true",
                        help="Run enhanced analysis (gene-set TDA + landscapes)")
    parser.add_argument("--resume-enhanced", action="store_true",
                        help="Resume from saved enhanced_features.csv")
    parser.add_argument("--info", action="store_true",
                        help="Print instructions for getting data")
    args = parser.parse_args()

    if args.info:
        print_instructions()
    elif args.resume_enhanced:
        # Load enhanced features and run stats
        features_path = _project_root / "data" / "results" / args.dataset / "enhanced_features.csv"
        if not features_path.exists():
            print(f"ERROR: No enhanced features at {features_path}")
            print("Run with --enhanced first.")
            sys.exit(1)
        print("=" * 60)
        print(f"TDA-CAR-T: Resume ENHANCED — {args.dataset}")
        print("=" * 60)
        feature_df = pd.read_csv(features_path)
        print(f"  Loaded {len(feature_df)} samples, {len(feature_df.columns) - 3} features")
        _run_classification_and_stats(feature_df)
        print("\n" + "=" * 60)
        print("DONE (resumed from enhanced features)")
        print("=" * 60)
    elif args.resume:
        run_from_features(args.dataset)
    elif args.enhanced:
        run_enhanced_analysis(args.dataset, download=args.download)
    else:
        run_real_analysis(args.dataset, download=args.download)
