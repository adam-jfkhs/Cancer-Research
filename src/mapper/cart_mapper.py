"""Mapper algorithm for CAR-T cell state topology.

Constructs a Mapper graph from single-cell CAR-T data to reveal the
topological structure of T cell states — branching trajectories,
circular pathways (exhaustion loops), and disconnected subpopulations
that standard clustering misses.
"""

import numpy as np
from typing import Optional


def build_mapper_graph(
    point_cloud,
    lens_fn="pca_1",
    n_cubes=10,
    overlap=0.3,
    clusterer=None,
    custom_lens=None,
):
    """Build a Mapper graph from single-cell point cloud data.

    Parameters
    ----------
    point_cloud : np.ndarray
        Shape (n_cells, n_dims).
    lens_fn : str
        Filter function: 'pca_1', 'pca_2', 'eccentricity', 'density',
        'exhaustion', or 'custom'.
    n_cubes : int
        Number of intervals in the cover.
    overlap : float
        Fraction of overlap between intervals.
    clusterer : object or None
        Scikit-learn compatible clusterer. Default: DBSCAN.
    custom_lens : np.ndarray or None
        Custom lens values (required if lens_fn='custom').

    Returns
    -------
    dict
        Mapper graph with 'nodes', 'edges', 'meta' (node membership info).
    """
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    # Compute lens function
    if lens_fn == "pca_1":
        pca = PCA(n_components=min(2, point_cloud.shape[1]))
        projected = pca.fit_transform(point_cloud)
        lens = projected[:, 0:1]
    elif lens_fn == "pca_2":
        pca = PCA(n_components=min(2, point_cloud.shape[1]))
        lens = pca.fit_transform(point_cloud)[:, :2]
    elif lens_fn == "eccentricity":
        lens = _eccentricity_lens(point_cloud)
    elif lens_fn == "density":
        lens = _density_lens(point_cloud)
    elif lens_fn == "custom":
        if custom_lens is None:
            raise ValueError("custom_lens required when lens_fn='custom'")
        lens = custom_lens.reshape(-1, 1) if custom_lens.ndim == 1 else custom_lens
    else:
        raise ValueError(f"Unknown lens function: {lens_fn}")

    if clusterer is None:
        clusterer = DBSCAN(eps=0.5, min_samples=3)

    try:
        import kmapper as km
        mapper = km.KeplerMapper(verbose=0)
        graph = mapper.map(
            lens, point_cloud,
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap),
            clusterer=clusterer,
        )
        nodes = {}
        for node_id, member_indices in graph["nodes"].items():
            nodes[node_id] = {
                "members": member_indices,
                "size": len(member_indices),
                "mean_position": point_cloud[member_indices].mean(axis=0),
            }
        edges = []
        for node_id, neighbors in graph["links"].items():
            for neighbor_id in neighbors:
                if node_id < neighbor_id:
                    edges.append((node_id, neighbor_id))
        return {
            "nodes": nodes,
            "edges": edges,
            "raw_graph": graph,
            "lens_values": lens,
            "mapper_obj": mapper,
        }
    except ImportError:
        return _build_mapper_native(point_cloud, lens, n_cubes, overlap, clusterer)


def _eccentricity_lens(point_cloud, metric="euclidean"):
    """Eccentricity filter: distance from each point to the center of mass."""
    from scipy.spatial.distance import cdist
    center = point_cloud.mean(axis=0, keepdims=True)
    return cdist(point_cloud, center, metric=metric)


def _density_lens(point_cloud, k=15):
    """Density filter: inverse of mean distance to k nearest neighbors."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(point_cloud)
    distances, _ = nn.kneighbors(point_cloud)
    mean_dist = distances[:, 1:].mean(axis=1)
    return (1.0 / (mean_dist + 1e-10)).reshape(-1, 1)


def _build_mapper_native(point_cloud, lens, n_cubes, overlap, clusterer):
    """Pure Python/sklearn Mapper implementation (no kmapper dependency).

    Implements the Mapper algorithm:
    1. Cover the lens range with overlapping intervals
    2. Pull back each interval to get subsets of points
    3. Cluster within each subset
    4. Connect clusters that share points
    """
    from sklearn.cluster import DBSCAN

    n = len(point_cloud)
    lens_1d = lens[:, 0] if lens.ndim > 1 else lens

    # Create overlapping cover
    l_min, l_max = lens_1d.min(), lens_1d.max()
    l_range = l_max - l_min
    if l_range == 0:
        l_range = 1.0

    cube_width = l_range / (n_cubes - (n_cubes - 1) * overlap) if n_cubes > 1 else l_range
    step = cube_width * (1 - overlap) if n_cubes > 1 else l_range

    nodes = {}
    point_to_nodes = {i: [] for i in range(n)}
    node_counter = 0

    for cube_idx in range(n_cubes):
        lower = l_min + cube_idx * step
        upper = lower + cube_width

        mask = (lens_1d >= lower) & (lens_1d <= upper)
        indices = np.where(mask)[0]

        if len(indices) < 2:
            if len(indices) == 1:
                node_id = f"node_{node_counter}"
                nodes[node_id] = {
                    "members": list(indices),
                    "size": 1,
                    "mean_position": point_cloud[indices].mean(axis=0),
                }
                point_to_nodes[indices[0]].append(node_id)
                node_counter += 1
            continue

        # Cluster within this interval
        subset = point_cloud[indices]
        labels = clusterer.fit_predict(subset)

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_mask = labels == cluster_id
            member_indices = indices[cluster_mask]

            node_id = f"node_{node_counter}"
            nodes[node_id] = {
                "members": list(member_indices),
                "size": len(member_indices),
                "mean_position": point_cloud[member_indices].mean(axis=0),
            }

            for idx in member_indices:
                point_to_nodes[idx].append(node_id)

            node_counter += 1

    # Build edges: nodes that share points are connected
    edges = set()
    for pt_idx, node_list in point_to_nodes.items():
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                edge = tuple(sorted([node_list[i], node_list[j]]))
                edges.add(edge)

    return {
        "nodes": nodes,
        "edges": list(edges),
        "raw_graph": None,
        "lens_values": lens,
        "mapper_obj": None,
    }


def color_mapper_by_phenotype(mapper_graph, cell_values, aggregation="mean"):
    """Color Mapper nodes by a cell-level phenotype score.

    Parameters
    ----------
    mapper_graph : dict
        Output of build_mapper_graph.
    cell_values : np.ndarray
        Per-cell values (e.g., exhaustion score, cytokine expression).
    aggregation : str
        'mean' or 'median'.

    Returns
    -------
    dict
        Node ID -> aggregated phenotype value.
    """
    agg_fn = np.mean if aggregation == "mean" else np.median
    node_colors = {}
    for node_id, node_info in mapper_graph["nodes"].items():
        members = node_info["members"]
        node_colors[node_id] = float(agg_fn(cell_values[members]))
    return node_colors


def detect_topological_features(mapper_graph):
    """Identify topological features in the Mapper graph.

    Detects:
    - Flares (branching points suggesting cell fate decisions)
    - Loops (cyclic structures suggesting exhaustion/renewal cycles)
    - Disconnected components (distinct T cell populations)

    Returns
    -------
    dict
        'branches': list of branching node IDs,
        'loops': list of cycle node sets,
        'components': list of connected component node sets,
        'hub_nodes': high-degree nodes.
    """
    # Build adjacency
    adj = {}
    for node_id in mapper_graph["nodes"]:
        adj[node_id] = set()
    for n1, n2 in mapper_graph["edges"]:
        adj[n1].add(n2)
        adj[n2].add(n1)

    # Branching points (degree >= 3)
    branches = [n for n, neighbors in adj.items() if len(neighbors) >= 3]

    # Connected components via BFS
    visited = set()
    components = []
    for node in adj:
        if node not in visited:
            component = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                queue.extend(adj[current] - visited)
            components.append(component)

    # Detect loops using DFS cycle detection
    loops = _find_cycles(adj)

    # Hub nodes (top 10% by degree)
    degrees = {n: len(neighbors) for n, neighbors in adj.items()}
    if degrees:
        threshold = np.percentile(list(degrees.values()), 90)
        hub_nodes = [n for n, d in degrees.items() if d >= threshold]
    else:
        hub_nodes = []

    return {
        "branches": branches,
        "loops": loops,
        "components": components,
        "hub_nodes": hub_nodes,
        "degree_distribution": degrees,
    }


def _find_cycles(adj, max_cycles=50):
    """Find simple cycles in the Mapper graph using DFS."""
    cycles = []
    visited_edges = set()

    for start in adj:
        stack = [(start, [start], set())]
        while stack and len(cycles) < max_cycles:
            node, path, path_edges = stack.pop()
            for neighbor in adj[node]:
                edge = tuple(sorted([node, neighbor]))
                if edge in path_edges:
                    continue
                if neighbor == start and len(path) >= 3:
                    cycles.append(set(path))
                elif neighbor not in set(path):
                    new_edges = path_edges | {edge}
                    stack.append((neighbor, path + [neighbor], new_edges))

    # Deduplicate
    unique = []
    for c in cycles:
        if not any(c == existing for existing in unique):
            unique.append(c)

    return unique[:max_cycles]


def mapper_to_networkx(mapper_graph):
    """Convert Mapper graph to NetworkX for further analysis.

    Returns
    -------
    networkx.Graph
        With node attributes: size, mean_position.
    """
    import networkx as nx

    G = nx.Graph()
    for node_id, info in mapper_graph["nodes"].items():
        G.add_node(node_id, size=info["size"])

    for n1, n2 in mapper_graph["edges"]:
        G.add_edge(n1, n2)

    return G
