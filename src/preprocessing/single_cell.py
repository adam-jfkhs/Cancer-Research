"""Preprocessing pipeline for single-cell RNA-seq data from CAR-T clinical trials.

Designed to work with public datasets from GEO (e.g., GSE151511, GSE197268)
containing pre/post-infusion CAR-T cell transcriptomes.
"""

import numpy as np
import pandas as pd
from typing import Optional


def load_cart_dataset(adata, min_genes=200, min_cells=3, max_mito_pct=20.0):
    """Standard QC and filtering for CAR-T single-cell data.

    Parameters
    ----------
    adata : anndata.AnnData
        Raw count matrix with cell metadata.
    min_genes : int
        Minimum genes per cell.
    min_cells : int
        Minimum cells per gene.
    max_mito_pct : float
        Maximum mitochondrial gene percentage.

    Returns
    -------
    anndata.AnnData
        Filtered and normalized AnnData object.
    """
    import scanpy as sc

    adata.var_names_make_unique()

    # QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # Filter
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs.pct_counts_mt < max_mito_pct, :].copy()

    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def select_cart_cells(adata, car_marker="CAR", method="annotation"):
    """Isolate CAR-positive T cells from mixed populations.

    Parameters
    ----------
    adata : anndata.AnnData
        Processed single-cell data.
    car_marker : str
        Column in obs or gene name identifying CAR+ cells.
    method : str
        'annotation' uses obs metadata, 'expression' uses transgene detection.

    Returns
    -------
    anndata.AnnData
        Subset containing only CAR+ T cells.
    """
    if method == "annotation":
        if car_marker in adata.obs.columns:
            mask = adata.obs[car_marker].astype(bool)
        else:
            raise KeyError(f"'{car_marker}' not found in adata.obs")
    elif method == "expression":
        if car_marker in adata.var_names:
            mask = np.asarray(adata[:, car_marker].X.todense()).flatten() > 0
        else:
            raise KeyError(f"'{car_marker}' not found in adata.var_names")
    else:
        raise ValueError(f"Unknown method: {method}")

    return adata[mask].copy()


def extract_exhaustion_signature(adata):
    """Score cells on canonical T cell exhaustion markers.

    Uses PDCD1 (PD-1), HAVCR2 (TIM-3), LAG3, TIGIT, TOX, ENTPD1 (CD39),
    CTLA4, and LAYN as exhaustion markers.

    Returns
    -------
    np.ndarray
        Exhaustion score per cell.
    """
    import scanpy as sc

    exhaustion_genes = ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1", "CTLA4", "LAYN"]
    present = [g for g in exhaustion_genes if g in adata.var_names]

    if len(present) < 3:
        raise ValueError(
            f"Only {len(present)} exhaustion markers found. Need at least 3."
        )

    sc.tl.score_genes(adata, gene_list=present, score_name="exhaustion_score")
    return adata.obs["exhaustion_score"].values


def compute_embedding(adata, n_pcs=50, n_neighbors=15, resolution=1.0):
    """Compute PCA, neighbors, UMAP, and Leiden clustering.

    Returns
    -------
    anndata.AnnData
        With .obsm['X_pca'], .obsm['X_umap'], .obs['leiden'].
    """
    import scanpy as sc

    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    sc.tl.pca(adata_hvg, n_comps=n_pcs)
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=resolution)

    # Transfer back
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obs["leiden"] = adata_hvg.obs["leiden"]

    return adata


def prepare_for_tda(adata, representation="pca", n_components=20):
    """Extract point cloud for TDA from processed single-cell data.

    Parameters
    ----------
    adata : anndata.AnnData
        Processed data with embeddings.
    representation : str
        'pca' or 'umap'.
    n_components : int
        Number of PCA components (ignored if representation='umap').

    Returns
    -------
    np.ndarray
        Point cloud (n_cells, n_dims).
    """
    if representation == "pca":
        if "X_pca" not in adata.obsm:
            raise ValueError("Run compute_embedding first.")
        return adata.obsm["X_pca"][:, :n_components]
    elif representation == "umap":
        if "X_umap" not in adata.obsm:
            raise ValueError("Run compute_embedding first.")
        return adata.obsm["X_umap"]
    else:
        raise ValueError(f"Unknown representation: {representation}")
