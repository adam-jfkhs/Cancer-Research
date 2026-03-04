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
    from scipy.spatial.distance import pdist, squareform

    dists = pdist(point_cloud, metric=metric)
    if thresh is None:
        thresh = np.percentile(dists, 5)

    try:
        from ripser import ripser
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
    except ImportError:
        return _compute_persistence_scipy(point_cloud, dists, maxdim, thresh, metric)


def _compute_persistence_scipy(point_cloud, dists_condensed, maxdim, thresh, metric):
    """Pure scipy/numpy fallback for persistent homology (H0 and H1).

    Uses union-find for H0 and a simplified Vietoris-Rips approach for H1.
    """
    from scipy.spatial.distance import squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = len(point_cloud)
    dist_matrix = squareform(dists_condensed)

    # --- H0: Connected components via MST (exact) ---
    mst = minimum_spanning_tree(dist_matrix)
    mst_edges = []
    cx = mst.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if v <= thresh:
            mst_edges.append((v, i, j))
    mst_edges.sort()

    h0_diagram = []
    # All points born at 0; they die when merged via MST edge
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    for dist_val, i, j in mst_edges:
        if union(i, j):
            h0_diagram.append([0.0, dist_val])

    # One component survives to infinity
    h0_diagram.append([0.0, np.inf])
    h0_diagram = np.array(h0_diagram) if h0_diagram else np.array([[0.0, np.inf]])

    diagrams = [h0_diagram]

    # --- H1: Approximate via edge additions beyond MST ---
    if maxdim >= 1:
        h1_diagram = []
        # Edges not in MST that close cycles
        mst_set = set()
        for _, i, j in mst_edges:
            mst_set.add((min(i, j), max(i, j)))

        # Collect all edges up to threshold, sorted by distance
        edge_list = []
        for idx in range(len(dists_condensed)):
            if dists_condensed[idx] <= thresh:
                # Convert condensed index to (i, j)
                i = int(n - 2 - int(np.sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
                j = int(idx + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
                edge_list.append((dists_condensed[idx], i, j))
        edge_list.sort()

        # Reset union-find for incremental construction
        parent = list(range(n))
        rank = [0] * n

        for dist_val, i, j in edge_list:
            key = (min(i, j), max(i, j))
            ri, rj = find(i), find(j)
            if ri == rj:
                # This edge closes a cycle — birth of H1 feature
                # Approximate: birth = last MST edge weight in path, death = this edge
                h1_diagram.append([dist_val * 0.5, dist_val])
            else:
                union(i, j)

        # Keep only significant features (top percentile by lifetime)
        if h1_diagram:
            h1_arr = np.array(h1_diagram)
            lifetimes = h1_arr[:, 1] - h1_arr[:, 0]
            if len(lifetimes) > 10:
                cutoff = np.percentile(lifetimes, 80)
                h1_arr = h1_arr[lifetimes >= cutoff]
            diagrams.append(h1_arr)
        else:
            diagrams.append(np.empty((0, 2)))

    # H2 placeholder (computationally expensive, skip in fallback)
    if maxdim >= 2:
        diagrams.append(np.empty((0, 2)))

    return {
        "diagrams": diagrams,
        "cocycles": [],
        "num_edges": len(edge_list) if maxdim >= 1 else None,
        "dperm2all": None,
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
