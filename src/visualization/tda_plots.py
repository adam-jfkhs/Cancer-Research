"""Visualization module for TDA-CAR-T analysis.

Persistence diagrams, landscapes, Mapper graphs, and integrated
metabolic-topological views.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional


def plot_persistence_diagram(diagrams, title="Persistence Diagram", ax=None):
    """Plot persistence diagram with birth vs. death axes.

    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams by dimension.
    title : str
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    labels = ["H0 (components)", "H1 (loops)", "H2 (voids)", "H3"]

    max_val = 0
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            max_val = max(max_val, finite.max())
            ax.scatter(
                finite[:, 0], finite[:, 1],
                c=colors[dim % len(colors)],
                label=labels[dim] if dim < len(labels) else f"H{dim}",
                alpha=0.6,
                s=30,
                edgecolors="black",
                linewidth=0.5,
            )

        # Plot infinite features as triangles at the top
        infinite = dgm[~np.isfinite(dgm[:, 1])]
        if len(infinite) > 0:
            ax.scatter(
                infinite[:, 0],
                [max_val * 1.1] * len(infinite),
                c=colors[dim % len(colors)],
                marker="^",
                alpha=0.8,
                s=60,
            )

    # Diagonal line
    lim = max_val * 1.15
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")

    return fig


def plot_persistence_landscape(landscapes, title="Persistence Landscape", ax=None):
    """Plot persistence landscape functions.

    Parameters
    ----------
    landscapes : np.ndarray
        Shape (num_landscapes, resolution).
    title : str
        Plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(landscapes)))
    x = np.linspace(0, 1, landscapes.shape[1])

    for k, (landscape, color) in enumerate(zip(landscapes, colors)):
        if np.any(landscape > 0):
            ax.fill_between(x, landscape, alpha=0.2, color=color)
            ax.plot(x, landscape, color=color, label=f"$\\lambda_{k+1}$", linewidth=1.5)

    ax.set_xlabel("Filtration parameter", fontsize=12)
    ax.set_ylabel("Landscape value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)

    return fig


def plot_mapper_graph(
    mapper_graph,
    node_color_values=None,
    node_color_label="Value",
    title="Mapper Graph",
    figsize=(12, 10),
    cmap="RdYlBu_r",
):
    """Visualize Mapper graph with optional node coloring.

    Parameters
    ----------
    mapper_graph : dict
        Output of build_mapper_graph.
    node_color_values : dict or None
        Node ID -> color value.
    node_color_label : str
        Colorbar label.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import networkx as nx

    G = nx.Graph()
    for node_id, info in mapper_graph["nodes"].items():
        G.add_node(node_id, size=info["size"])
    for n1, n2 in mapper_graph["edges"]:
        G.add_edge(n1, n2)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(len(G.nodes)))

    # Node sizes proportional to cell count
    sizes = [mapper_graph["nodes"][n]["size"] * 10 for n in G.nodes]

    if node_color_values is not None:
        colors = [node_color_values.get(n, 0) for n in G.nodes]
        scatter = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=sizes,
            node_color=colors, cmap=cmap,
            edgecolors="black", linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label=node_color_label, shrink=0.8)
    else:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=sizes,
            node_color="#2196F3", alpha=0.7,
            edgecolors="black", linewidths=0.5,
        )

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=1)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    return fig


def plot_responder_comparison(
    responder_stats, non_responder_stats, title="Topological Feature Comparison"
):
    """Compare topological features between responders and non-responders.

    Parameters
    ----------
    responder_stats : list of dict
        Persistence statistics for responder patients.
    non_responder_stats : list of dict
        Persistence statistics for non-responder patients.

    Returns
    -------
    matplotlib.figure.Figure
    """
    features = ["H0_count", "H0_max_persistence", "H0_entropy",
                "H1_count", "H1_max_persistence", "H1_entropy"]

    r_values = {f: [] for f in features}
    nr_values = {f: [] for f in features}

    for stats in responder_stats:
        for f in features:
            dim, metric = f.split("_", 1)
            r_values[f].append(stats.get(dim, {}).get(metric, 0))

    for stats in non_responder_stats:
        for f in features:
            dim, metric = f.split("_", 1)
            nr_values[f].append(stats.get(dim, {}).get(metric, 0))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        data = [r_values[feature], nr_values[feature]]
        bp = ax.boxplot(data, labels=["Responder", "Non-Responder"],
                        patch_artist=True)
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#FF5722")
        ax.set_title(feature.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Value")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_metabolic_topology(
    mapper_graph,
    metabolic_node_scores,
    pathways=None,
    figsize=(16, 10),
):
    """Multi-panel Mapper graph colored by different metabolic pathways.

    Parameters
    ----------
    mapper_graph : dict
        Mapper graph.
    metabolic_node_scores : pd.DataFrame
        From integrate_metabolic_topology.
    pathways : list or None
        Which pathways to plot. Default: all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import networkx as nx

    if pathways is None:
        pathways = list(metabolic_node_scores.columns)

    n_panels = len(pathways)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    G = nx.Graph()
    for node_id, info in mapper_graph["nodes"].items():
        G.add_node(node_id, size=info["size"])
    for n1, n2 in mapper_graph["edges"]:
        G.add_edge(n1, n2)

    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(max(len(G.nodes), 1)))
    sizes = [mapper_graph["nodes"][n]["size"] * 10 for n in G.nodes]

    cmaps = ["Reds", "Blues", "Greens", "Purples", "Oranges", "YlOrRd"]

    for i, pathway in enumerate(pathways):
        ax = axes[i]
        colors = [metabolic_node_scores.loc[n, pathway]
                  if n in metabolic_node_scores.index else 0
                  for n in G.nodes]

        scatter = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=sizes,
            node_color=colors, cmap=cmaps[i % len(cmaps)],
            edgecolors="black", linewidths=0.5,
        )
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
        plt.colorbar(scatter, ax=ax, shrink=0.6)
        ax.set_title(pathway.replace("_", " ").title(), fontsize=11)
        ax.axis("off")

    # Hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metabolic Pathway Activity on Mapper Graph",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_timecourse_topology(timecourse_stats, metric="H1_count"):
    """Plot topological feature evolution over treatment timecourse.

    Parameters
    ----------
    timecourse_stats : dict
        Timepoint -> persistence statistics.
    metric : str
        Which topological metric to plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    timepoints = list(timecourse_stats.keys())
    dim, stat = metric.split("_", 1)
    values = [timecourse_stats[tp].get(dim, {}).get(stat, 0) for tp in timepoints]

    ax.plot(range(len(timepoints)), values, "o-", color="#2196F3",
            linewidth=2, markersize=8)
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, rotation=45, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Topological Evolution: {metric}", fontsize=14)

    plt.tight_layout()
    return fig
