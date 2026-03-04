"""Persistent homology analysis for CAR-T cell transcriptomic landscapes.

Computes persistence diagrams from single-cell point clouds to capture
topological features (connected components, loops, voids) that characterize
T cell state spaces — including exhaustion trajectories, effector/memory
bifurcations, and response heterogeneity.
"""

import numpy as np
from typing import Optional


def compute_persistence(
    point_cloud,
    maxdim=2,
    thresh=None,
    coeff=2,
    metric="euclidean",
):
    """Compute persistent homology of a point cloud via Vietoris-Rips filtration.

    Parameters
    ----------
    point_cloud : np.ndarray
        Shape (n_points, n_dims). Typically PCA-reduced single-cell expression.
    maxdim : int
        Maximum homology dimension to compute.
    thresh : float or None
        Maximum filtration value. None = automatic.
    coeff : int
        Coefficient field for homology (2 = Z/2Z).
    metric : str
        Distance metric.

    Returns
    -------
    dict
        'diagrams': list of persistence diagrams per dimension,
        'cocycles': cocycle representatives (if available).
    """
    from ripser import ripser

    if thresh is None:
        from scipy.spatial.distance import pdist
        dists = pdist(point_cloud, metric=metric)
        thresh = np.percentile(dists, 5)

    result = ripser(
        point_cloud,
        maxdim=maxdim,
        thresh=thresh,
        coeff=coeff,
        do_cocycles=True,
    )

    return {
        "diagrams": result["dgms"],
        "cocycles": result.get("cocycles", []),
        "num_edges": result.get("num_edges", None),
        "dperm2all": result.get("dperm2all", None),
    }


def persistence_statistics(diagrams):
    """Extract statistical summaries from persistence diagrams.

    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams indexed by dimension.

    Returns
    -------
    dict
        Per-dimension statistics: count of features, max/mean/total persistence,
        entropy, and persistent Betti numbers at selected thresholds.
    """
    stats = {}

    for dim, dgm in enumerate(diagrams):
        # Filter out infinite death values
        finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))

        if len(finite) == 0:
            stats[f"H{dim}"] = {
                "count": 0,
                "max_persistence": 0.0,
                "mean_persistence": 0.0,
                "total_persistence": 0.0,
                "entropy": 0.0,
            }
            continue

        lifetimes = finite[:, 1] - finite[:, 0]
        lifetimes = lifetimes[lifetimes > 0]

        if len(lifetimes) == 0:
            stats[f"H{dim}"] = {
                "count": 0,
                "max_persistence": 0.0,
                "mean_persistence": 0.0,
                "total_persistence": 0.0,
                "entropy": 0.0,
            }
            continue

        total = np.sum(lifetimes)
        probs = lifetimes / total
        entropy = -np.sum(probs * np.log(probs + 1e-16))

        stats[f"H{dim}"] = {
            "count": len(lifetimes),
            "max_persistence": float(np.max(lifetimes)),
            "mean_persistence": float(np.mean(lifetimes)),
            "total_persistence": float(total),
            "entropy": float(entropy),
        }

    return stats


def persistence_landscape(diagram, num_landscapes=5, resolution=1000):
    """Compute persistence landscapes for vectorized TDA features.

    Persistence landscapes are functional summaries of persistence diagrams
    that live in a Banach space — enabling statistical tests, means, and
    machine learning on topological features.

    Parameters
    ----------
    diagram : np.ndarray
        Single persistence diagram (birth, death) pairs.
    num_landscapes : int
        Number of landscape functions to compute.
    resolution : int
        Number of sample points.

    Returns
    -------
    np.ndarray
        Shape (num_landscapes, resolution). Landscape functions.
    """
    finite = diagram[np.isfinite(diagram[:, 1])]
    if len(finite) == 0:
        return np.zeros((num_landscapes, resolution))

    births = finite[:, 0]
    deaths = finite[:, 1]

    t_min = np.min(births)
    t_max = np.max(deaths)
    t_vals = np.linspace(t_min, t_max, resolution)

    # Tent functions for each bar
    n_bars = len(finite)
    tent_values = np.zeros((n_bars, resolution))

    for i in range(n_bars):
        b, d = births[i], deaths[i]
        mid = (b + d) / 2.0
        for j, t in enumerate(t_vals):
            if b <= t <= mid:
                tent_values[i, j] = t - b
            elif mid < t <= d:
                tent_values[i, j] = d - t

    # k-th landscape is k-th largest tent value at each t
    landscapes = np.zeros((num_landscapes, resolution))
    for j in range(resolution):
        sorted_vals = np.sort(tent_values[:, j])[::-1]
        for k in range(min(num_landscapes, len(sorted_vals))):
            landscapes[k, j] = sorted_vals[k]

    return landscapes


def persistence_images(diagram, resolution=(20, 20), sigma=0.1, weight_fn=None):
    """Compute persistence images for ML-compatible topological featurization.

    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram.
    resolution : tuple
        Grid resolution (birth_bins, persistence_bins).
    sigma : float
        Gaussian kernel bandwidth.
    weight_fn : callable or None
        Weighting function f(birth, persistence). Default: linear in persistence.

    Returns
    -------
    np.ndarray
        Persistence image of shape resolution.
    """
    from persim import PersistenceImager

    pimgr = PersistenceImager(
        pixel_size=sigma,
        birth_range=None,
        pers_range=None,
        kernel_params={"sigma": [[sigma, 0], [0, sigma]]},
    )

    finite = diagram[np.isfinite(diagram[:, 1])]
    if len(finite) == 0:
        return np.zeros(resolution)

    img = pimgr.fit_transform([finite])
    return img[0]


def compare_persistence_diagrams(dgm1, dgm2, metric="wasserstein", p=2):
    """Compare two persistence diagrams using standard TDA distances.

    Parameters
    ----------
    dgm1, dgm2 : np.ndarray
        Persistence diagrams.
    metric : str
        'wasserstein' or 'bottleneck'.
    p : int
        Wasserstein-p distance (only for metric='wasserstein').

    Returns
    -------
    float
        Distance between diagrams.
    """
    from persim import wasserstein, bottleneck

    # Filter to finite
    d1 = dgm1[np.isfinite(dgm1[:, 1])]
    d2 = dgm2[np.isfinite(dgm2[:, 1])]

    if metric == "wasserstein":
        return wasserstein(d1, d2, order=p)
    elif metric == "bottleneck":
        return bottleneck(d1, d2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def topological_feature_matrix(point_clouds, labels=None, maxdim=1):
    """Build feature matrix from persistence statistics across multiple samples.

    Useful for downstream classification (responder vs. non-responder).

    Parameters
    ----------
    point_clouds : list of np.ndarray
        Each element is a point cloud from one patient/timepoint.
    labels : list or None
        Optional labels for each sample.
    maxdim : int
        Maximum homology dimension.

    Returns
    -------
    pd.DataFrame
        Rows = samples, columns = topological features.
    """
    import pandas as pd

    rows = []
    for i, pc in enumerate(point_clouds):
        result = compute_persistence(pc, maxdim=maxdim)
        stats = persistence_statistics(result["diagrams"])

        row = {"sample_idx": i}
        if labels is not None:
            row["label"] = labels[i]

        for dim_key, dim_stats in stats.items():
            for stat_name, stat_val in dim_stats.items():
                row[f"{dim_key}_{stat_name}"] = stat_val

        rows.append(row)

    return pd.DataFrame(rows)
