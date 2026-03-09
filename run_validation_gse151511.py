"""External validation of TDA-CAR-T findings on GSE151511 (Deng et al. 2020).

Tests ONLY the pre-specified exhaustion and cytokine-signaling topological
features that showed strongest signal in GSE197268 (primary cohort).

Usage:
    # With data already downloaded:
    python run_validation_gse151511.py

    # Download first:
    python run_validation_gse151511.py --download

    # Resume from saved features:
    python run_validation_gse151511.py --resume
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
    GSE151511_PATIENT_RESPONSE,
    GSE151511_PATIENT_HISTOLOGY,
    GSE151511_RAW_RESPONSE,
    GSE151511_GSM_TO_PATIENT,
    download_geo_metadata,
    download_geo_supplementary,
    load_real_dataset,
    parse_response_labels,
)
from src.preprocessing.single_cell import (
    load_cart_dataset,
    extract_exhaustion_signature,
    compute_embedding,
)
from src.tda.persistent_homology import (
    compute_persistence,
    persistence_statistics,
    persistence_landscape,
)

DATASET_ID = "GSE151511"

# ---------------------------------------------------------------------------
# Gene sets — identical to those used in GSE197268 analysis (run_real_data.py)
# ---------------------------------------------------------------------------
EXHAUSTION_GENES = [
    "PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1", "CTLA4", "LAYN",
    "CD244", "CD160", "BTLA",
]
CYTOKINE_SIGNALING_GENES = [
    "IL6", "IL10", "TGFB1", "IL21", "IL15", "IL7", "STAT1", "STAT3",
    "STAT5A", "JAK1", "JAK2", "SOCS1", "SOCS3",
]
ACTIVATION_GENES = [
    "CD69", "IL2RA", "TNFRSF9", "ICOS", "CD28", "IFNG", "TNF", "IL2",
    "GZMB", "PRF1", "TNFRSF4", "HLA-DRA",
]
MEMORY_STEMNESS_GENES = [
    "TCF7", "LEF1", "BCL6", "IL7R", "SELL", "CCR7", "CD27", "CD28",
    "BACH2", "KLF2",
]

GENE_SETS = {
    "exhaustion": EXHAUSTION_GENES,
    "cytokine_signaling": CYTOKINE_SIGNALING_GENES,
    "activation": ACTIVATION_GENES,
    "memory_stemness": MEMORY_STEMNESS_GENES,
}


def _gene_set_point_cloud(adata_sub, gene_set, min_genes=3):
    """Extract PCA point cloud from expression of a specific gene set."""
    from sklearn.decomposition import PCA

    present = [g for g in gene_set if g in adata_sub.var_names]
    if len(present) < min_genes:
        return None, present

    X = adata_sub[:, present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    n_components = min(len(present), 10, X.shape[0] - 1)
    if n_components < 2:
        return None, present

    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), present


def _compute_landscape_features(diagram, prefix, n_land=3, resolution=100):
    """Compute persistence landscape summary features."""
    land = persistence_landscape(diagram, num_landscapes=n_land, resolution=resolution)
    features = {}
    for k in range(n_land):
        row = land[k]
        features[f"{prefix}_L{k}_max"] = float(np.max(row))
        features[f"{prefix}_L{k}_mean"] = float(np.mean(row))
        features[f"{prefix}_L{k}_integral"] = float(np.sum(row)) * (1.0 / resolution)
    return features


def compute_patient_features(adata, meta, response_labels):
    """Compute topological features for each patient in GSE151511.

    Returns DataFrame with one row per patient.
    """
    print("\n[4/5] Computing topological features per patient...")

    rows = []
    compatibility_notes = []

    for gsm_id, patient_id in sorted(GSE151511_GSM_TO_PATIENT.items()):
        label = GSE151511_PATIENT_RESPONSE.get(patient_id, "unknown")
        if label == "unknown":
            print(f"  Skipping {patient_id} ({gsm_id}): NE (not evaluable)")
            continue

        mask = adata.obs["sample_id"] == gsm_id
        adata_sub = adata[mask]

        if adata_sub.shape[0] < 50:
            print(f"  Skipping {patient_id}: only {adata_sub.shape[0]} cells")
            compatibility_notes.append(
                f"{patient_id}: skipped, {adata_sub.shape[0]} cells (< 50 minimum)"
            )
            continue

        row = {
            "sample_id": gsm_id,
            "patient_id": patient_id,
            "label": label,
            "histology": GSE151511_PATIENT_HISTOLOGY.get(patient_id, "unknown"),
            "raw_response": GSE151511_RAW_RESPONSE.get(patient_id, "unknown"),
            "n_cells": adata_sub.shape[0],
        }

        # --- A) Full PCA topology ---
        pc_full = adata_sub.obsm["X_pca"][:, :20]
        result_full = compute_persistence(pc_full, maxdim=1)
        stats_full = persistence_statistics(result_full["diagrams"])
        for dim_key, dim_stats in stats_full.items():
            for stat_name, stat_val in dim_stats.items():
                row[f"{dim_key}_{stat_name}"] = stat_val

        # --- B) Persistence landscape features ---
        for dim_idx, dim_name in enumerate(["H0", "H1"]):
            if dim_idx < len(result_full["diagrams"]):
                dgm = result_full["diagrams"][dim_idx]
                land_feats = _compute_landscape_features(
                    dgm, prefix=dim_name, n_land=3, resolution=100
                )
                row.update(land_feats)

        # --- C) Gene-set-specific TDA ---
        for gs_name, gs_genes in GENE_SETS.items():
            pc_gs, genes_found = _gene_set_point_cloud(adata_sub, gs_genes)
            n_found = len(genes_found)
            row[f"{gs_name}_genes_found"] = n_found
            row[f"{gs_name}_genes_total"] = len(gs_genes)

            if pc_gs is None:
                for stat in ["count", "max_persistence", "mean_persistence",
                             "total_persistence", "entropy"]:
                    row[f"{gs_name}_H0_{stat}"] = np.nan
                    row[f"{gs_name}_H1_{stat}"] = np.nan
                if n_found < 3:
                    compatibility_notes.append(
                        f"{patient_id}/{gs_name}: only {n_found}/{len(gs_genes)} genes found"
                    )
                continue

            result_gs = compute_persistence(pc_gs, maxdim=1)
            stats_gs = persistence_statistics(result_gs["diagrams"])
            for dim_key, dim_stats in stats_gs.items():
                for stat_name, stat_val in dim_stats.items():
                    row[f"{gs_name}_{dim_key}_{stat_name}"] = stat_val

        rows.append(row)
        print(f"  {patient_id}: {adata_sub.shape[0]} cells → {label} "
              f"({GSE151511_PATIENT_HISTOLOGY.get(patient_id, '?')})")

    feature_df = pd.DataFrame(rows)
    return feature_df, compatibility_notes


def run_prespecified_tests(feature_df, compatibility_notes):
    """Run ONLY pre-specified hypothesis tests from GSE197268 findings.

    Tests only:
    1. Exhaustion H1 composite score (primary hypothesis)
    2. Cytokine signaling H1 composite score (secondary hypothesis)

    Reports effect direction, effect sizes, p-values, and compatibility issues.
    """
    from scipy.stats import mannwhitneyu
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 85)
    print("EXTERNAL VALIDATION: PRE-SPECIFIED HYPOTHESIS TESTS")
    print(f"Dataset: GSE151511 (Deng et al., Nature Medicine 2020)")
    print(f"Primary cohort: GSE197268 (Haradhvala et al., Nature Medicine 2022)")
    print("=" * 85)

    r_mask = feature_df["label"] == "responder"
    nr_mask = feature_df["label"] == "non_responder"
    n_r = r_mask.sum()
    n_nr = nr_mask.sum()

    print(f"\nSample sizes: {n_r} responders (CR), {n_nr} non-responders (PD/PR)")
    print(f"Histology: {dict(feature_df['histology'].value_counts())}")

    # --- Compatibility report ---
    print(f"\n{'='*85}")
    print("COMPATIBILITY NOTES")
    print(f"{'='*85}")

    # Gene overlap check
    for gs_name in GENE_SETS:
        found_col = f"{gs_name}_genes_found"
        total_col = f"{gs_name}_genes_total"
        if found_col in feature_df.columns:
            found_vals = feature_df[found_col].dropna()
            if len(found_vals) > 0:
                mean_found = found_vals.mean()
                total = feature_df[total_col].iloc[0] if total_col in feature_df.columns else len(GENE_SETS[gs_name])
                pct = mean_found / total * 100
                status = "OK" if pct >= 50 else "LOW"
                print(f"  {gs_name}: {mean_found:.0f}/{total:.0f} genes detected "
                      f"({pct:.0f}%) [{status}]")

    if compatibility_notes:
        print(f"\n  Per-patient notes:")
        for note in compatibility_notes:
            print(f"    - {note}")

    # --- Dataset comparison ---
    print(f"\n{'='*85}")
    print("DATASET COMPARISON: GSE197268 vs GSE151511")
    print(f"{'='*85}")
    print(f"  {'':30} {'GSE197268':>15} {'GSE151511':>15}")
    print(f"  {'-'*60}")
    print(f"  {'Paper':30} {'Haradhvala 2022':>15} {'Deng 2020':>15}")
    print(f"  {'CAR-T product':30} {'axi-cel/tisa-cel':>15} {'axi-cel':>15}")
    print(f"  {'Disease':30} {'LBCL':>15} {'LBCL':>15}")
    print(f"  {'Response definition':30} {'6mo relapse':>15} {'3mo PET/CT':>15}")
    print(f"  {'N responders':30} {'17':>15} {str(n_r):>15}")
    print(f"  {'N non-responders':30} {'15':>15} {str(n_nr):>15}")
    print(f"  {'Data type':30} {'10x 5p/3p':>15} {'CapID 10x 5p':>15}")
    print(f"  {'Genome build':30} {'GRCh38 (likely)':>15} {'hg19':>15}")

    print(f"\n  Key differences:")
    print(f"    1. Response definition: 6-month vs 3-month assessment")
    print(f"    2. CAR-T product: mixed axi-cel/tisa-cel vs axi-cel only")
    print(f"    3. Sequencing: standard 10x vs CapID 10x (capture-based)")
    print(f"    4. Genome build: GRCh38 vs hg19 (may affect gene name mapping)")

    # ======================================================================
    # PRE-SPECIFIED TESTS
    # ======================================================================

    # Composite definitions — identical to GSE197268 analysis
    composite_defs = {
        "Exhaustion H1 topology": {
            "cols": [
                "exhaustion_H1_mean_persistence",
                "exhaustion_H1_max_persistence",
                "exhaustion_H1_entropy",
            ],
            "hypothesis": "primary",
            "expected_direction": "NR > R",
            "rationale": (
                "Non-responder CAR-T cells cycle through exhaustion states "
                "(higher H1 = more loop topology in exhaustion marker space)"
            ),
        },
        "Cytokine signaling H1 topology": {
            "cols": [
                "cytokine_signaling_H1_mean_persistence",
                "cytokine_signaling_H1_max_persistence",
                "cytokine_signaling_H1_entropy",
            ],
            "hypothesis": "secondary",
            "expected_direction": "NR > R",
            "rationale": (
                "Non-responders show more complex cytokine signaling topology"
            ),
        },
    }

    print(f"\n{'='*85}")
    print("PRE-SPECIFIED COMPOSITE SCORE TESTS")
    print(f"(No multiple comparison correction: one test per biological hypothesis)")
    print(f"{'='*85}")

    validation_results = []

    for comp_name, comp_def in composite_defs.items():
        cols = comp_def["cols"]
        avail = [c for c in cols if c in feature_df.columns
                 and feature_df[c].notna().sum() >= n_r + n_nr - 2]

        print(f"\n  {'-'*80}")
        print(f"  {comp_name} [{comp_def['hypothesis']} hypothesis]")
        print(f"  Rationale: {comp_def['rationale']}")
        print(f"  Expected direction from GSE197268: {comp_def['expected_direction']}")
        print(f"  Components available: {len(avail)}/{len(cols)}")

        if len(avail) < 2:
            print(f"  SKIPPED: insufficient features ({avail})")
            validation_results.append({
                "composite": comp_name,
                "hypothesis": comp_def["hypothesis"],
                "status": "skipped",
                "reason": f"only {len(avail)}/{len(cols)} features available",
            })
            continue

        print(f"  Using: {avail}")

        # Z-score and composite
        vals = feature_df[avail].dropna()
        valid_idx = vals.index

        scaler = StandardScaler()
        z_scores = scaler.fit_transform(vals)
        composite = np.mean(z_scores, axis=1)

        r_comp = composite[feature_df.loc[valid_idx, "label"] == "responder"]
        nr_comp = composite[feature_df.loc[valid_idx, "label"] == "non_responder"]

        if len(r_comp) < 2 or len(nr_comp) < 2:
            print(f"  SKIPPED: insufficient samples (R={len(r_comp)}, NR={len(nr_comp)})")
            validation_results.append({
                "composite": comp_name,
                "hypothesis": comp_def["hypothesis"],
                "status": "skipped",
                "reason": f"R={len(r_comp)}, NR={len(nr_comp)} after dropna",
            })
            continue

        r_mean = np.mean(r_comp)
        nr_mean = np.mean(nr_comp)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(r_comp) - 1) * np.var(r_comp, ddof=1) +
             (len(nr_comp) - 1) * np.var(nr_comp, ddof=1))
            / (len(r_comp) + len(nr_comp) - 2)
        )
        d = (r_mean - nr_mean) / pooled_std if pooled_std > 0 else 0.0

        # Mann-Whitney U
        try:
            _, mw_p = mannwhitneyu(r_comp, nr_comp, alternative="two-sided")
        except ValueError:
            mw_p = 1.0

        # Permutation test (10,000 permutations)
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
        observed_direction = "R < NR" if r_mean < nr_mean else "R > NR"
        direction_consistent = (
            (comp_def["expected_direction"] == "NR > R" and r_mean < nr_mean) or
            (comp_def["expected_direction"] == "R > NR" and r_mean > nr_mean)
        )

        # Significance assessment
        sig_label = ""
        if perm_p < 0.001:
            sig_label = "***"
        elif perm_p < 0.01:
            sig_label = "**"
        elif perm_p < 0.05:
            sig_label = "*"
        elif perm_p < 0.10:
            sig_label = "."

        # Effect size interpretation
        if abs(d) >= 0.8:
            d_label = "large"
        elif abs(d) >= 0.5:
            d_label = "medium"
        elif abs(d) >= 0.2:
            d_label = "small"
        else:
            d_label = "negligible"

        print(f"\n  RESULTS:")
        print(f"    R mean (z):  {r_mean:+.3f}  (n={len(r_comp)})")
        print(f"    NR mean (z): {nr_mean:+.3f}  (n={len(nr_comp)})")
        print(f"    Direction:   {observed_direction}")
        print(f"    Consistent with GSE197268: {'YES' if direction_consistent else 'NO'}")
        print(f"    Cohen's d:   {d:+.3f} ({d_label})")
        print(f"    MW-U p:      {mw_p:.4f}")
        print(f"    Perm p:      {perm_p:.4f} {sig_label}")

        if perm_p < 0.05 and direction_consistent:
            print(f"    >>> VALIDATED: significant (p < 0.05) with consistent direction <<<")
        elif direction_consistent and abs(d) >= 0.5:
            print(f"    >>> TREND: consistent direction with {d_label} effect size <<<")
        elif not direction_consistent:
            print(f"    >>> NOT REPLICATED: opposite direction <<<")

        validation_results.append({
            "composite": comp_name,
            "hypothesis": comp_def["hypothesis"],
            "status": "tested",
            "n_r": len(r_comp),
            "n_nr": len(nr_comp),
            "r_mean": r_mean,
            "nr_mean": nr_mean,
            "direction": observed_direction,
            "direction_consistent": direction_consistent,
            "cohens_d": d,
            "d_label": d_label,
            "mw_p": mw_p,
            "perm_p": perm_p,
            "sig_label": sig_label,
        })

    # --- Individual feature tests (for completeness) ---
    print(f"\n{'='*85}")
    print("INDIVIDUAL FEATURE TESTS (pre-specified from GSE197268)")
    print(f"{'='*85}")

    prespecified_features = [
        "exhaustion_H1_mean_persistence",
        "exhaustion_H1_max_persistence",
        "exhaustion_H1_count",
        "exhaustion_H0_entropy",
        "cytokine_signaling_H1_mean_persistence",
        "cytokine_signaling_H1_max_persistence",
        "cytokine_signaling_H1_count",
    ]

    available_feats = [f for f in prespecified_features if f in feature_df.columns]

    if available_feats:
        print(f"\n  {'Feature':<42} {'R':>8} {'NR':>8} {'d':>7} {'MW p':>8} {'Dir':>7}")
        print(f"  {'-'*82}")

        for col in available_feats:
            r_vals = feature_df.loc[r_mask, col].dropna().values.astype(float)
            nr_vals = feature_df.loc[nr_mask, col].dropna().values.astype(float)
            if len(r_vals) < 2 or len(nr_vals) < 2:
                continue

            r_mean = np.mean(r_vals)
            nr_mean = np.mean(nr_vals)
            ps = np.sqrt(
                ((len(r_vals) - 1) * np.var(r_vals, ddof=1) +
                 (len(nr_vals) - 1) * np.var(nr_vals, ddof=1))
                / (len(r_vals) + len(nr_vals) - 2)
            )
            d = (r_mean - nr_mean) / ps if ps > 0 else 0.0
            try:
                _, p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
            except ValueError:
                p = 1.0
            direction = "R<NR" if r_mean < nr_mean else "R>NR"
            sig = "*" if p < 0.05 else ("." if p < 0.10 else "")
            print(f"  {col:<42} {r_mean:>8.3f} {nr_mean:>8.3f} {d:>+7.3f} {p:>8.4f} {direction:>7} {sig}")

    # --- Summary ---
    print(f"\n{'='*85}")
    print("VALIDATION SUMMARY")
    print(f"{'='*85}")

    for vr in validation_results:
        if vr["status"] == "skipped":
            print(f"  {vr['composite']}: SKIPPED ({vr['reason']})")
        else:
            replicated = vr["direction_consistent"] and vr["perm_p"] < 0.05
            trending = vr["direction_consistent"] and abs(vr["cohens_d"]) >= 0.5
            status = "VALIDATED" if replicated else ("TRENDING" if trending else "NOT REPLICATED")
            print(f"  {vr['composite']}: {status}")
            print(f"    d={vr['cohens_d']:+.3f} ({vr['d_label']}), "
                  f"perm p={vr['perm_p']:.4f}, "
                  f"direction={'consistent' if vr['direction_consistent'] else 'OPPOSITE'}")

    # --- Next-best dataset recommendation ---
    print(f"\n{'='*85}")
    print("RECOMMENDED NEXT VALIDATION DATASET")
    print(f"{'='*85}")
    print("""
  1. GSE186441 — Bai et al., Nature Medicine 2022
     "Single-cell multiomics dissection of basal and antigen-specific
      activation states of CD19-targeted CAR T cells"
     - 10 patients with B-ALL, 10x multiome (RNA + ATAC)
     - Response labels available (CR vs relapse)
     - Advantage: multiomics enables regulatory network TDA
     - Limitation: different disease (ALL vs LBCL), smaller cohort

  2. GSE197268 subgroup analysis (already available)
     - Split axi-cel (n=19) vs tisa-cel (n=13) within primary cohort
     - Tests product-specific topology without new data download
     - Advantage: immediate, no compatibility concerns
     - Limitation: not truly independent

  3. GSE125881 — Fraietta et al. 2018 + related datasets
     - CD19 CAR-T in CLL (different disease context)
     - Tests whether exhaustion topology generalizes across indications
     - Advantage: well-characterized responder/non-responder labels
     - Limitation: CLL is biologically distinct from LBCL

  Recommendation: Start with GSE197268 axi-cel/tisa-cel subgroup analysis
  (no new download needed), then GSE186441 for true external validation.
""")

    return validation_results


def run_validation(download=False, resume=False):
    """Run the full external validation pipeline."""
    output_dir = _project_root / "data" / "results" / DATASET_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "validation_features.csv"

    if resume and features_path.exists():
        print("=" * 85)
        print(f"EXTERNAL VALIDATION: Resume from saved features")
        print("=" * 85)
        feature_df = pd.read_csv(features_path)
        print(f"  Loaded {len(feature_df)} patients, "
              f"{len(feature_df.columns) - 6} topological features")
        compatibility_notes = []  # Notes not saved, will be partial
        run_prespecified_tests(feature_df, compatibility_notes)
        return

    print("=" * 85)
    print(f"EXTERNAL VALIDATION: GSE151511 (Deng et al., Nature Medicine 2020)")
    print(f"Testing pre-specified hypotheses from GSE197268 (primary cohort)")
    print("=" * 85)

    # --- 1. Metadata ---
    print("\n[1/5] Loading metadata...")
    if download:
        meta = download_geo_metadata(DATASET_ID)
        download_geo_supplementary(DATASET_ID)
    else:
        meta = download_geo_metadata(DATASET_ID)

    # --- 2. Response labels ---
    print("\n[2/5] Parsing response labels...")
    response_labels = parse_response_labels(meta, DATASET_ID)

    # --- 3. Load scRNA-seq ---
    print("\n[3/5] Loading scRNA-seq data...")
    adata = load_real_dataset(DATASET_ID)
    print(f"  Raw: {adata.shape[0]} cells × {adata.shape[1]} genes")

    adata = load_cart_dataset(adata)
    print(f"  After QC: {adata.shape[0]} cells × {adata.shape[1]} genes")

    adata = compute_embedding(adata)

    # --- 4. Compute features ---
    feature_df, compatibility_notes = compute_patient_features(
        adata, meta, response_labels
    )

    # Save features
    feature_df.to_csv(features_path, index=False)
    print(f"\n  Saved: {features_path}")
    print(f"  Patients: {len(feature_df)}")
    print(f"  Features: {len(feature_df.columns) - 6}")

    # --- 5. Pre-specified tests ---
    run_prespecified_tests(feature_df, compatibility_notes)

    print(f"\n{'='*85}")
    print("EXTERNAL VALIDATION COMPLETE")
    print(f"  Results saved to: {output_dir}")
    print(f"  Resume later: python run_validation_gse151511.py --resume")
    print(f"{'='*85}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="External validation of TDA-CAR-T on GSE151511"
    )
    parser.add_argument("--download", action="store_true",
                        help="Download data from GEO before analysis")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved validation_features.csv")
    args = parser.parse_args()

    run_validation(download=args.download, resume=args.resume)
