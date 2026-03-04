"""Interactive 3D visualizations for TDA-CAR-T analysis.

Generates HTML files with interactive Plotly visualizations:
- 3D point clouds of T cell state spaces (colored by state/exhaustion/pseudotime)
- 3D Mapper graphs with metabolic overlays
- Persistence diagram 3D views
- Animated timecourse topology evolution
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path


def visualize_cell_landscape_3d(
    latent_coords,
    states,
    exhaustion=None,
    pseudotime=None,
    title="CAR-T Cell State Landscape",
    output_path=None,
):
    """Interactive 3D scatter of T cell state space.

    Parameters
    ----------
    latent_coords : np.ndarray
        Shape (n_cells, >=3). First 3 dims used for 3D plot.
    states : list of str
        Cell state labels.
    exhaustion : np.ndarray or None
        Exhaustion scores per cell.
    pseudotime : np.ndarray or None
        Pseudotime values per cell.
    output_path : str or None
        If provided, saves interactive HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    coords = latent_coords[:, :3]

    # Color by state
    unique_states = sorted(set(states))
    state_colors = {
        "naive": "#4CAF50",
        "activated": "#FF9800",
        "effector": "#F44336",
        "memory_precursor": "#2196F3",
        "central_memory": "#3F51B5",
        "effector_memory": "#9C27B0",
        "exhausted_progenitor": "#795548",
        "terminally_exhausted": "#212121",
        "stem_memory": "#00BCD4",
    }

    fig = make_subplots(
        rows=1, cols=2 if exhaustion is not None else 1,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]]
        if exhaustion is not None
        else [[{"type": "scatter3d"}]],
        subplot_titles=(
            ["Cell States", "Exhaustion Score"]
            if exhaustion is not None
            else ["Cell States"]
        ),
        horizontal_spacing=0.02,
    )

    # Panel 1: colored by state
    for state in unique_states:
        mask = np.array([s == state for s in states])
        if not mask.any():
            continue

        display_name = state.replace("_", " ").title()
        color = state_colors.get(state, "#999999")

        # Handle trajectory states
        if state not in state_colors:
            if "exhausted" in state:
                color = "#795548"
            elif "memory" in state:
                color = "#2196F3"
            elif "effector" in state:
                color = "#F44336"
            else:
                color = "#999999"

        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.6),
                name=display_name,
                text=[f"State: {display_name}" for _ in range(mask.sum())],
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # Panel 2: colored by exhaustion
    if exhaustion is not None:
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=exhaustion,
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="Exhaustion", x=1.0),
                    opacity=0.6,
                ),
                name="Exhaustion",
                text=[f"Exhaustion: {e:.2f}" for e in exhaustion],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700,
        width=1400 if exhaustion is not None else 800,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            font=dict(size=10),
        ),
    )

    if exhaustion is not None:
        fig.update_layout(
            scene2=dict(
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
                zaxis_title="Dim 3",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            ),
        )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def visualize_mapper_3d(
    mapper_graph,
    node_color_values=None,
    color_label="Value",
    title="Mapper Graph — T Cell Topology",
    output_path=None,
):
    """Interactive 3D visualization of Mapper graph.

    Nodes positioned by spring layout in 3D, sized by cell count,
    colored by phenotype scores.

    Parameters
    ----------
    mapper_graph : dict
        Output of build_mapper_graph.
    node_color_values : dict or None
        Node ID -> scalar value for coloring.
    color_label : str
        Colorbar label.
    output_path : str or None
        Save path for HTML.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import networkx as nx

    G = nx.Graph()
    for node_id, info in mapper_graph["nodes"].items():
        G.add_node(node_id, size=info["size"])
    for n1, n2 in mapper_graph["edges"]:
        G.add_edge(n1, n2)

    # 3D spring layout
    pos = nx.spring_layout(G, dim=3, seed=42, k=3.0 / np.sqrt(max(len(G.nodes), 1)))

    # Extract coordinates
    node_ids = list(G.nodes)
    node_x = [pos[n][0] for n in node_ids]
    node_y = [pos[n][1] for n in node_ids]
    node_z = [pos[n][2] for n in node_ids]
    node_sizes = [mapper_graph["nodes"][n]["size"] for n in node_ids]

    # Normalize sizes for display
    max_size = max(node_sizes) if node_sizes else 1
    display_sizes = [max(5, s / max_size * 30) for s in node_sizes]

    # Edge traces
    edge_x, edge_y, edge_z = [], [], []
    for n1, n2 in mapper_graph["edges"]:
        if n1 in pos and n2 in pos:
            edge_x.extend([pos[n1][0], pos[n2][0], None])
            edge_y.extend([pos[n1][1], pos[n2][1], None])
            edge_z.extend([pos[n1][2], pos[n2][2], None])

    fig = go.Figure()

    # Edges
    fig.add_trace(
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", width=2),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # Nodes
    if node_color_values is not None:
        colors = [node_color_values.get(n, 0) for n in node_ids]
        hover_text = [
            f"Node: {n}<br>Cells: {mapper_graph['nodes'][n]['size']}<br>"
            f"{color_label}: {node_color_values.get(n, 0):.3f}"
            for n in node_ids
        ]

        fig.add_trace(
            go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode="markers",
                marker=dict(
                    size=display_sizes,
                    color=colors,
                    colorscale="RdYlBu_r",
                    colorbar=dict(title=color_label),
                    line=dict(color="black", width=0.5),
                    opacity=0.85,
                ),
                text=hover_text,
                hoverinfo="text",
                name="Mapper nodes",
            )
        )
    else:
        hover_text = [
            f"Node: {n}<br>Cells: {mapper_graph['nodes'][n]['size']}"
            for n in node_ids
        ]

        fig.add_trace(
            go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode="markers",
                marker=dict(
                    size=display_sizes,
                    color="#2196F3",
                    line=dict(color="black", width=0.5),
                    opacity=0.85,
                ),
                text=hover_text,
                hoverinfo="text",
                name="Mapper nodes",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=800,
        width=900,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            bgcolor="rgba(240,240,240,0.1)",
        ),
        showlegend=False,
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def visualize_responder_comparison_3d(
    responder_data,
    non_responder_data,
    title="Responder vs Non-Responder Topology",
    output_path=None,
):
    """Side-by-side 3D comparison of responder vs non-responder landscapes.

    Parameters
    ----------
    responder_data : dict
        Simulation result for responder.
    non_responder_data : dict
        Simulation result for non-responder.
    output_path : str or None
        Save path.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=["Responder", "Non-Responder"],
        horizontal_spacing=0.02,
    )

    for col, (data, label) in enumerate(
        [(responder_data, "Responder"), (non_responder_data, "Non-Responder")], 1
    ):
        coords = data["latent"][:, :3]
        exhaustion = data["exhaustion"]

        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=exhaustion,
                    colorscale="RdYlBu_r",
                    colorbar=dict(
                        title="Exhaustion",
                        x=1.0 if col == 2 else -0.05,
                        len=0.8,
                    ),
                    opacity=0.5,
                ),
                name=label,
                text=[f"Exhaustion: {e:.2f}" for e in exhaustion],
                hoverinfo="text",
            ),
            row=1, col=col,
        )

    scene_config = dict(
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        zaxis_title="Dim 3",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700,
        width=1400,
        scene=scene_config,
        scene2=scene_config,
        showlegend=False,
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def visualize_persistence_3d(
    diagrams,
    title="Persistence Diagram (3D)",
    output_path=None,
):
    """3D persistence diagram: birth x death x dimension.

    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams by dimension.
    output_path : str or None
        Save path.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    dim_names = ["H0 (components)", "H1 (loops)", "H2 (voids)"]

    fig = go.Figure()

    max_val = 0
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            continue

        max_val = max(max_val, finite.max())
        lifetimes = finite[:, 1] - finite[:, 0]

        fig.add_trace(
            go.Scatter3d(
                x=finite[:, 0],
                y=finite[:, 1],
                z=[dim] * len(finite),
                mode="markers",
                marker=dict(
                    size=lifetimes * 15 + 3,
                    color=colors[dim % len(colors)],
                    opacity=0.7,
                    line=dict(color="black", width=0.5),
                ),
                name=dim_names[dim] if dim < len(dim_names) else f"H{dim}",
                text=[
                    f"Birth: {b:.3f}<br>Death: {d:.3f}<br>"
                    f"Lifetime: {d-b:.3f}<br>Dim: H{dim}"
                    for b, d in finite
                ],
                hoverinfo="text",
            )
        )

    # Diagonal plane
    grid_size = 20
    diag_range = np.linspace(0, max_val * 1.1, grid_size)
    xx, zz = np.meshgrid(diag_range, np.linspace(-0.5, len(diagrams) - 0.5, 5))
    yy = xx.copy()

    fig.add_trace(
        go.Surface(
            x=xx, y=yy, z=zz,
            opacity=0.1,
            colorscale=[[0, "gray"], [1, "gray"]],
            showscale=False,
            hoverinfo="none",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=700,
        width=800,
        scene=dict(
            xaxis_title="Birth",
            yaxis_title="Death",
            zaxis_title="Dimension",
            zaxis=dict(
                tickvals=list(range(len(diagrams))),
                ticktext=[f"H{d}" for d in range(len(diagrams))],
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def visualize_timecourse_3d(
    timecourse_data,
    title="CAR-T Topology Over Time",
    output_path=None,
):
    """Animated 3D visualization of T cell landscape evolution during treatment.

    Parameters
    ----------
    timecourse_data : dict
        Timepoint -> simulation result with 'latent', 'exhaustion'.
    output_path : str or None
        Save path.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    timepoints = list(timecourse_data.keys())

    # Create frames for animation
    frames = []
    for tp in timepoints:
        data = timecourse_data[tp]
        coords = data["latent"][:, :3]
        exhaustion = data["exhaustion"]

        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=exhaustion,
                        colorscale="RdYlBu_r",
                        cmin=0, cmax=1,
                        opacity=0.5,
                    ),
                )
            ],
            name=tp,
        )
        frames.append(frame)

    # Initial state
    init = timecourse_data[timepoints[0]]
    init_coords = init["latent"][:, :3]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=init_coords[:, 0],
                y=init_coords[:, 1],
                z=init_coords[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=init["exhaustion"],
                    colorscale="RdYlBu_r",
                    cmin=0, cmax=1,
                    colorbar=dict(title="Exhaustion"),
                    opacity=0.5,
                ),
            )
        ],
        frames=frames,
    )

    # Animation controls
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=800,
        width=900,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
            xaxis=dict(range=[-4, 8]),
            yaxis=dict(range=[-4, 6]),
            zaxis=dict(range=[-5, 5]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1500, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=500),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[
                    dict(
                        args=[[tp], dict(
                            frame=dict(duration=300, redraw=True),
                            mode="immediate",
                            transition=dict(duration=300),
                        )],
                        label=tp.replace("_", " ").title(),
                        method="animate",
                    )
                    for tp in timepoints
                ],
                x=0.1, len=0.8,
                xanchor="left",
                y=-0.05,
                currentvalue=dict(
                    prefix="Timepoint: ",
                    visible=True,
                    xanchor="center",
                ),
                transition=dict(duration=300),
            )
        ],
    )

    if output_path:
        fig.write_html(str(output_path), include_plotlyjs="cdn")

    return fig


def generate_all_visualizations(output_dir="data/visualizations"):
    """Generate all 3D interactive visualizations from synthetic data.

    Creates HTML files that can be opened in any browser.

    Parameters
    ----------
    output_dir : str
        Directory to save HTML files.

    Returns
    -------
    list of str
        Paths to generated HTML files.
    """
    import sys
    sys.path.insert(0, ".")

    from src.simulation.synthetic_cart import CARTSimulator
    from src.tda.persistent_homology import compute_persistence
    from src.mapper.cart_mapper import build_mapper_graph, color_mapper_by_phenotype

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated = []

    print("Generating synthetic data...")
    sim = CARTSimulator(n_genes=2000, seed=42)
    responder = sim.simulate_population(n_cells=3000, response_type="responder")
    non_responder = sim.simulate_population(n_cells=3000, response_type="non_responder")

    # 1. Cell state landscape
    print("Creating cell state landscape...")
    path = out / "01_cell_landscape_responder.html"
    visualize_cell_landscape_3d(
        responder["latent"], responder["states"],
        exhaustion=responder["exhaustion"],
        title="Responder — CAR-T Cell State Landscape",
        output_path=path,
    )
    generated.append(str(path))

    # 2. Responder vs non-responder comparison
    print("Creating responder comparison...")
    path = out / "02_responder_comparison.html"
    visualize_responder_comparison_3d(
        responder, non_responder,
        output_path=path,
    )
    generated.append(str(path))

    # 3. Persistence diagrams
    print("Computing persistence and creating 3D diagram...")
    r_ph = compute_persistence(responder["latent"], maxdim=2)
    path = out / "03_persistence_diagram_3d.html"
    visualize_persistence_3d(
        r_ph["diagrams"],
        title="Persistence Diagram — Responder T Cell Topology",
        output_path=path,
    )
    generated.append(str(path))

    # 4. Mapper graph (responder)
    print("Building Mapper graph...")
    r_mapper = build_mapper_graph(
        responder["latent"], lens_fn="pca_1", n_cubes=15, overlap=0.3
    )
    exh_colors = color_mapper_by_phenotype(r_mapper, responder["exhaustion"])

    path = out / "04_mapper_graph_responder.html"
    visualize_mapper_3d(
        r_mapper,
        node_color_values=exh_colors,
        color_label="Exhaustion",
        title="Mapper Graph — Responder T Cell Topology",
        output_path=path,
    )
    generated.append(str(path))

    # 5. Mapper graph (non-responder)
    nr_mapper = build_mapper_graph(
        non_responder["latent"], lens_fn="pca_1", n_cubes=15, overlap=0.3
    )
    nr_exh_colors = color_mapper_by_phenotype(nr_mapper, non_responder["exhaustion"])

    path = out / "05_mapper_graph_non_responder.html"
    visualize_mapper_3d(
        nr_mapper,
        node_color_values=nr_exh_colors,
        color_label="Exhaustion",
        title="Mapper Graph — Non-Responder T Cell Topology",
        output_path=path,
    )
    generated.append(str(path))

    # 6. Animated timecourse
    print("Creating timecourse animation...")
    tc = sim.simulate_treatment_timecourse(n_cells_per_timepoint=1500)
    path = out / "06_timecourse_animation.html"
    visualize_timecourse_3d(
        tc,
        title="CAR-T Cell Landscape Evolution During Treatment",
        output_path=path,
    )
    generated.append(str(path))

    print(f"\nGenerated {len(generated)} visualizations in {output_dir}/")
    for p in generated:
        print(f"  {p}")

    return generated


if __name__ == "__main__":
    generate_all_visualizations()
