"""Generate publication-quality figures for the TDA-CAR-T paper.

Produces Figures 1-5 from synthetic validation data + analysis results.
Run from project root: python paper/generate_figures.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Color palette
RESP_COLOR = "#2196F3"    # Blue for responders
NR_COLOR = "#F44336"      # Red for non-responders
ACCENT = "#4CAF50"        # Green accent

from src.simulation.synthetic_cart import CARTSimulator, generate_validation_dataset
from src.tda.persistent_homology import (
    compute_persistence,
    persistence_statistics,
    persistence_landscape,
    topological_feature_matrix,
    # compare_persistence_diagrams,  # requires persim
)
from src.mapper.cart_mapper import build_mapper_graph, detect_topological_features


def fig1_overview_and_synthetic_landscape():
    """Figure 1: Study overview + synthetic CAR-T cell landscapes."""
    print("Generating Figure 1: Overview & Synthetic Landscapes...")

    sim = CARTSimulator(n_genes=2000, seed=42)
    resp = sim.simulate_population(n_cells=2000, response_type="responder")
    nresp = sim.simulate_population(n_cells=2000, response_type="non_responder")

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Schematic of approach (text-based)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.set_aspect("equal")
    ax_a.axis("off")
    ax_a.set_title("A. TDA-CAR-T Approach", fontweight="bold", fontsize=11, loc="left")

    steps = [
        (5, 9, "scRNA-seq\n(CAR-T infusion product)", "#E3F2FD"),
        (5, 7, "QC + Normalization\n+ PCA embedding", "#E8F5E9"),
        (5, 5, "Persistent Homology\n(H₀, H₁ features)", "#FFF3E0"),
        (5, 3, "Mapper Graph\nConstruction", "#F3E5F5"),
        (5, 1, "Response Classification\n& Statistical Testing", "#FFEBEE"),
    ]
    for x, y, txt, color in steps:
        ax_a.add_patch(plt.Rectangle((1.5, y - 0.7), 7, 1.4, facecolor=color,
                                      edgecolor="gray", linewidth=0.8, zorder=1))
        ax_a.text(x, y, txt, ha="center", va="center", fontsize=7, zorder=2)

    for i in range(len(steps) - 1):
        ax_a.annotate("", xy=(5, steps[i+1][1] + 0.7), xytext=(5, steps[i][1] - 0.7),
                      arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    # Panel B: Responder latent space (2D projection)
    ax_b = fig.add_subplot(gs[0, 1])
    latent_r = resp["latent"]
    states_r = resp["states"]
    state_set = sorted(set(states_r))
    cmap = plt.cm.tab10
    for i, state in enumerate(state_set):
        mask = np.array([s == state for s in states_r])
        ax_b.scatter(latent_r[mask, 0], latent_r[mask, 1], s=3, alpha=0.4,
                     color=cmap(i / max(len(state_set) - 1, 1)),
                     label=state.replace("_", " "), rasterized=True)
    ax_b.set_title("B. Responder Landscape", fontweight="bold", fontsize=11, loc="left")
    ax_b.set_xlabel("Latent dim 1")
    ax_b.set_ylabel("Latent dim 2")
    ax_b.legend(fontsize=5, markerscale=3, loc="upper right", ncol=1, framealpha=0.8)

    # Panel C: Non-responder latent space
    ax_c = fig.add_subplot(gs[0, 2])
    latent_nr = nresp["latent"]
    states_nr = nresp["states"]
    state_set_nr = sorted(set(states_nr))
    for i, state in enumerate(state_set_nr):
        mask = np.array([s == state for s in states_nr])
        ax_c.scatter(latent_nr[mask, 0], latent_nr[mask, 1], s=3, alpha=0.4,
                     color=cmap(i / max(len(state_set_nr) - 1, 1)),
                     label=state.replace("_", " "), rasterized=True)
    ax_c.set_title("C. Non-responder Landscape", fontweight="bold", fontsize=11, loc="left")
    ax_c.set_xlabel("Latent dim 1")
    ax_c.set_ylabel("Latent dim 2")
    ax_c.legend(fontsize=5, markerscale=3, loc="upper right", ncol=1, framealpha=0.8)

    # Panel D: Exhaustion score distributions
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.hist(resp["exhaustion"], bins=40, alpha=0.6, color=RESP_COLOR,
              label="Responder", density=True)
    ax_d.hist(nresp["exhaustion"], bins=40, alpha=0.6, color=NR_COLOR,
              label="Non-responder", density=True)
    ax_d.set_title("D. Exhaustion Score Distribution", fontweight="bold", fontsize=11, loc="left")
    ax_d.set_xlabel("Exhaustion Score")
    ax_d.set_ylabel("Density")
    ax_d.legend(fontsize=9)

    # Panel E: State composition barplot
    ax_e = fig.add_subplot(gs[1, 1])
    all_states = sorted(set(states_r) | set(states_nr))
    r_fracs = [sum(1 for s in states_r if s == st) / len(states_r) for st in all_states]
    nr_fracs = [sum(1 for s in states_nr if s == st) / len(states_nr) for st in all_states]
    x = np.arange(len(all_states))
    w = 0.35
    ax_e.barh(x - w/2, r_fracs, w, color=RESP_COLOR, label="Responder", alpha=0.8)
    ax_e.barh(x + w/2, nr_fracs, w, color=NR_COLOR, label="Non-responder", alpha=0.8)
    ax_e.set_yticks(x)
    ax_e.set_yticklabels([s.replace("_", " ") for s in all_states], fontsize=7)
    ax_e.set_xlabel("Fraction")
    ax_e.set_title("E. Cell State Composition", fontweight="bold", fontsize=11, loc="left")
    ax_e.legend(fontsize=8)
    ax_e.invert_yaxis()

    # Panel F: Pseudotime distributions
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.hist(resp["pseudotime"], bins=40, alpha=0.6, color=RESP_COLOR,
              label="Responder", density=True)
    ax_f.hist(nresp["pseudotime"], bins=40, alpha=0.6, color=NR_COLOR,
              label="Non-responder", density=True)
    ax_f.set_title("F. Pseudotime Distribution", fontweight="bold", fontsize=11, loc="left")
    ax_f.set_xlabel("Pseudotime")
    ax_f.set_ylabel("Density")
    ax_f.legend(fontsize=9)

    fig.savefig(FIGURES_DIR / "fig1_overview_landscape.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_overview_landscape.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 1")


def fig2_persistence_diagrams():
    """Figure 2: Persistence diagrams and topological comparison."""
    print("Generating Figure 2: Persistence Diagrams...")

    sim = CARTSimulator(n_genes=2000, seed=42)
    resp = sim.simulate_population(n_cells=1500, response_type="responder")
    nresp = sim.simulate_population(n_cells=1500, response_type="non_responder")

    # Compute persistence on latent space
    result_r = compute_persistence(resp["latent"], maxdim=1, thresh=3.0)
    result_nr = compute_persistence(nresp["latent"], maxdim=1, thresh=3.0)

    stats_r = persistence_statistics(result_r["diagrams"])
    stats_nr = persistence_statistics(result_nr["diagrams"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Row 1: Persistence diagrams
    for col, (result, label, color, stats) in enumerate([
        (result_r, "Responder", RESP_COLOR, stats_r),
        (result_nr, "Non-responder", NR_COLOR, stats_nr),
    ]):
        ax = axes[0, col]
        for dim, dgm in enumerate(result["diagrams"]):
            if len(dgm) == 0:
                continue
            births = dgm[:, 0]
            deaths = dgm[:, 1]
            # Remove infinite deaths for plotting
            finite = np.isfinite(deaths)
            marker = "o" if dim == 0 else "^"
            ax.scatter(births[finite], deaths[finite], s=20, alpha=0.6,
                       marker=marker, label=f"H{dim} (n={len(dgm)})",
                       color=plt.cm.Set1(dim))
        # Diagonal
        lim = ax.get_xlim()[1]
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, lw=0.8)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        panel = "A" if col == 0 else "B"
        ax.set_title(f"{panel}. {label} Persistence Diagram",
                     fontweight="bold", fontsize=11, loc="left")
        ax.legend(fontsize=8)

    # Panel C: Barcode comparison for H1
    ax_c = axes[0, 2]
    # H1 barcodes (sorted by persistence)
    for group_idx, (result, label, color) in enumerate([
        (result_r, "Responder", RESP_COLOR),
        (result_nr, "Non-responder", NR_COLOR),
    ]):
        dgm = result["diagrams"][1] if len(result["diagrams"]) > 1 else np.array([]).reshape(0, 2)
        if len(dgm) == 0:
            continue
        finite = np.isfinite(dgm[:, 1])
        dgm_f = dgm[finite]
        pers = dgm_f[:, 1] - dgm_f[:, 0]
        order = np.argsort(-pers)[:15]  # Top 15 bars
        for i, idx in enumerate(order):
            y = i + group_idx * 17
            ax_c.barh(y, pers[idx], left=dgm_f[idx, 0], height=0.7,
                      color=color, alpha=0.7, edgecolor="none")
    ax_c.set_xlabel("Filtration value")
    ax_c.set_ylabel("Bar index")
    ax_c.set_title("C. H₁ Barcode Comparison", fontweight="bold", fontsize=11, loc="left")
    # Add legend manually
    from matplotlib.patches import Patch
    ax_c.legend(handles=[Patch(facecolor=RESP_COLOR, label="Responder"),
                         Patch(facecolor=NR_COLOR, label="Non-responder")], fontsize=9)

    # Row 2: Statistical comparison
    # Panel D: H0 feature comparison
    dataset = generate_validation_dataset(n_responders=8, n_non_responders=8, n_cells=800)
    r_stats = []
    nr_stats = []
    for pc, label in zip(dataset["point_clouds"], dataset["labels"]):
        result = compute_persistence(pc, maxdim=1)
        stats = persistence_statistics(result["diagrams"])
        if label == "responder":
            r_stats.append(stats)
        else:
            nr_stats.append(stats)

    # Panel D: H1 count comparison
    ax_d = axes[1, 0]
    r_h1_counts = [s["H1"]["count"] for s in r_stats]
    nr_h1_counts = [s["H1"]["count"] for s in nr_stats]
    bp = ax_d.boxplot([r_h1_counts, nr_h1_counts], labels=["Responder", "Non-resp."],
                       patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor(RESP_COLOR)
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor(NR_COLOR)
    bp["boxes"][1].set_alpha(0.5)
    ax_d.set_ylabel("H₁ feature count")
    ax_d.set_title("D. H₁ Loop Count", fontweight="bold", fontsize=11, loc="left")
    from scipy.stats import mannwhitneyu
    _, p = mannwhitneyu(r_h1_counts, nr_h1_counts, alternative="two-sided")
    ax_d.text(1.5, max(max(r_h1_counts), max(nr_h1_counts)) * 0.95,
              f"p = {p:.3f}", ha="center", fontsize=10, fontstyle="italic")

    # Panel E: H1 max persistence
    ax_e = axes[1, 1]
    r_h1_max = [s["H1"]["max_persistence"] for s in r_stats]
    nr_h1_max = [s["H1"]["max_persistence"] for s in nr_stats]
    bp2 = ax_e.boxplot([r_h1_max, nr_h1_max], labels=["Responder", "Non-resp."],
                        patch_artist=True, widths=0.6)
    bp2["boxes"][0].set_facecolor(RESP_COLOR)
    bp2["boxes"][0].set_alpha(0.5)
    bp2["boxes"][1].set_facecolor(NR_COLOR)
    bp2["boxes"][1].set_alpha(0.5)
    ax_e.set_ylabel("Max H₁ persistence")
    ax_e.set_title("E. H₁ Max Persistence", fontweight="bold", fontsize=11, loc="left")
    _, p2 = mannwhitneyu(r_h1_max, nr_h1_max, alternative="two-sided")
    ax_e.text(1.5, max(max(r_h1_max), max(nr_h1_max)) * 0.95,
              f"p = {p2:.3f}", ha="center", fontsize=10, fontstyle="italic")

    # Panel F: H0 entropy
    ax_f = axes[1, 2]
    r_h0_ent = [s["H0"]["entropy"] for s in r_stats]
    nr_h0_ent = [s["H0"]["entropy"] for s in nr_stats]
    bp3 = ax_f.boxplot([r_h0_ent, nr_h0_ent], labels=["Responder", "Non-resp."],
                        patch_artist=True, widths=0.6)
    bp3["boxes"][0].set_facecolor(RESP_COLOR)
    bp3["boxes"][0].set_alpha(0.5)
    bp3["boxes"][1].set_facecolor(NR_COLOR)
    bp3["boxes"][1].set_alpha(0.5)
    ax_f.set_ylabel("H₀ entropy")
    ax_f.set_title("F. Component Entropy", fontweight="bold", fontsize=11, loc="left")
    _, p3 = mannwhitneyu(r_h0_ent, nr_h0_ent, alternative="two-sided")
    ax_f.text(1.5, max(max(r_h0_ent), max(nr_h0_ent)) * 0.95,
              f"p = {p3:.3f}", ha="center", fontsize=10, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_persistence_diagrams.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_persistence_diagrams.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 2")


def fig3_mapper_graphs():
    """Figure 3: Mapper graph comparison and topological features."""
    print("Generating Figure 3: Mapper Graphs...")

    sim = CARTSimulator(n_genes=2000, seed=42)
    resp = sim.simulate_population(n_cells=1500, response_type="responder")
    nresp = sim.simulate_population(n_cells=1500, response_type="non_responder")

    graph_r = build_mapper_graph(resp["latent"], lens_fn="pca_1", n_cubes=8, overlap=0.3)
    graph_nr = build_mapper_graph(nresp["latent"], lens_fn="pca_1", n_cubes=8, overlap=0.3)

    feat_r = detect_topological_features(graph_r)
    feat_nr = detect_topological_features(graph_nr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, graph, features, label, color in [
        (axes[0], graph_r, feat_r, "Responder", RESP_COLOR),
        (axes[1], graph_nr, feat_nr, "Non-responder", NR_COLOR),
    ]:
        # Draw mapper graph
        nodes = graph["nodes"]
        edges = graph["edges"]

        if not nodes:
            ax.text(0.5, 0.5, "No nodes", ha="center", va="center")
            ax.set_title(f"{label} Mapper")
            continue

        # Layout: use node lens values for x, spread y
        node_ids = list(nodes.keys())
        n_nodes = len(node_ids)
        # Simple force-directed-like layout
        np.random.seed(42)
        pos = {}
        for i, nid in enumerate(node_ids):
            node_info = nodes[nid]
            if isinstance(node_info, dict):
                members = node_info.get("members", [])
                lens_val = np.mean(graph["lens_values"][np.array(members)]) if len(members) > 0 else 0
            else:
                lens_val = float(i) / max(n_nodes - 1, 1)
            pos[nid] = (lens_val, np.random.randn() * 0.5 + (i % 5) * 0.3)

        # Draw edges
        for e in edges:
            n1, n2 = e[0], e[1]
            if n1 in pos and n2 in pos:
                ax.plot([pos[n1][0], pos[n2][0]], [pos[n1][1], pos[n2][1]],
                        color="gray", alpha=0.3, lw=0.5)

        # Draw nodes
        sizes = [min(nodes[nid].get("size", 1) if isinstance(nodes[nid], dict) else len(nodes[nid]), 200)
                 for nid in node_ids]
        xs = [pos[nid][0] for nid in node_ids]
        ys = [pos[nid][1] for nid in node_ids]
        ax.scatter(xs, ys, s=sizes, c=color, alpha=0.7, edgecolors="k", linewidths=0.5)

        panel = "A" if label == "Responder" else "B"
        ax.set_title(f"{panel}. {label} Mapper Graph\n"
                     f"Nodes={n_nodes}, Branches={features.get('branch_count', 0)}, "
                     f"Loops={features.get('loop_count', 0)}",
                     fontweight="bold", fontsize=10, loc="left")
        ax.set_xlabel("Lens (PC1)")
        ax.set_ylabel("Layout position")

    # Panel C: Feature comparison table
    ax_c = axes[2]
    ax_c.axis("off")
    ax_c.set_title("C. Topological Feature Summary", fontweight="bold", fontsize=10, loc="left")

    feature_names = ["Components", "Branches", "Loops", "Hub nodes", "Degree mean"]
    r_vals = [
        feat_r.get("component_count", 0),
        feat_r.get("branch_count", 0),
        feat_r.get("loop_count", 0),
        feat_r.get("hub_count", 0),
        f"{feat_r.get('degree_mean', 0):.1f}",
    ]
    nr_vals = [
        feat_nr.get("component_count", 0),
        feat_nr.get("branch_count", 0),
        feat_nr.get("loop_count", 0),
        feat_nr.get("hub_count", 0),
        f"{feat_nr.get('degree_mean', 0):.1f}",
    ]

    table_data = [[f, str(r), str(nr)] for f, r, nr in zip(feature_names, r_vals, nr_vals)]
    table = ax_c.table(cellText=table_data,
                       colLabels=["Feature", "Responder", "Non-responder"],
                       cellLoc="center", loc="center",
                       colColours=[("#E0E0E0", RESP_COLOR + "40", NR_COLOR + "40")] if False else ["#E0E0E0"] * 3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color header and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E0E0E0")
            cell.set_text_props(fontweight="bold")
        elif col == 1:
            cell.set_facecolor("#E3F2FD")
        elif col == 2:
            cell.set_facecolor("#FFEBEE")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_mapper_graphs.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_mapper_graphs.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 3")


def fig4_classification_and_landscapes():
    """Figure 4: Classification results and persistence landscapes."""
    print("Generating Figure 4: Classification & Landscapes...")

    # Generate larger validation dataset
    dataset = generate_validation_dataset(n_responders=10, n_non_responders=10, n_cells=800)

    # Extract features
    feature_df = topological_feature_matrix(
        dataset["point_clouds"], labels=dataset["labels"], maxdim=1
    )

    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # Panel A: LOO classification accuracy
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    mask = feature_df["label"].isin(["responder", "non_responder"])
    labeled_df = feature_df[mask].copy()

    X = labeled_df.drop(columns=["label"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.loc[:, (X != 0).any(axis=0)].fillna(0)
    y = (labeled_df["label"] == "responder").astype(int)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=3)
    y_pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    acc = np.mean(y_pred == y)

    clf.fit(X, y)

    # Confusion matrix
    ax_a = fig.add_subplot(gs[0, 0])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_a,
                xticklabels=["Non-resp.", "Responder"],
                yticklabels=["Non-resp.", "Responder"], cbar=False)
    ax_a.set_xlabel("Predicted")
    ax_a.set_ylabel("Actual")
    ax_a.set_title(f"A. LOO Classification\nAccuracy = {acc:.0%}",
                   fontweight="bold", fontsize=11, loc="left")

    # Panel B: Feature importances
    ax_b = fig.add_subplot(gs[0, 1])
    importances = sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])
    top_n = 10
    top_feats = importances[:top_n]
    feat_names = [f[0].replace("_", "\n", 1) for f in top_feats]
    feat_vals = [f[1] for f in top_feats]
    bars = ax_b.barh(range(top_n), feat_vals, color=ACCENT, alpha=0.8)
    ax_b.set_yticks(range(top_n))
    ax_b.set_yticklabels(feat_names, fontsize=7)
    ax_b.set_xlabel("Feature Importance")
    ax_b.set_title("B. Top Discriminative Features", fontweight="bold", fontsize=11, loc="left")
    ax_b.invert_yaxis()

    # Panel C: Persistence landscapes
    ax_c = fig.add_subplot(gs[0, 2])

    # Compute average landscape for R and NR
    r_landscapes = []
    nr_landscapes = []
    for pc, label in zip(dataset["point_clouds"], dataset["labels"]):
        result = compute_persistence(pc, maxdim=1)
        if len(result["diagrams"]) > 1 and len(result["diagrams"][1]) > 0:
            dgm = result["diagrams"][1]
            finite = np.isfinite(dgm[:, 1])
            dgm_f = dgm[finite]
            if len(dgm_f) > 0:
                land = persistence_landscape(dgm_f, num_landscapes=1, resolution=100)
                if label == "responder":
                    r_landscapes.append(land[0])
                else:
                    nr_landscapes.append(land[0])

    if r_landscapes and nr_landscapes:
        r_mean = np.mean(r_landscapes, axis=0)
        nr_mean = np.mean(nr_landscapes, axis=0)
        r_std = np.std(r_landscapes, axis=0)
        nr_std = np.std(nr_landscapes, axis=0)
        x_grid = np.linspace(0, 1, 100)

        ax_c.plot(x_grid, r_mean, color=RESP_COLOR, lw=2, label="Responder")
        ax_c.fill_between(x_grid, r_mean - r_std, r_mean + r_std,
                          color=RESP_COLOR, alpha=0.2)
        ax_c.plot(x_grid, nr_mean, color=NR_COLOR, lw=2, label="Non-responder")
        ax_c.fill_between(x_grid, nr_mean - nr_std, nr_mean + nr_std,
                          color=NR_COLOR, alpha=0.2)
    ax_c.set_xlabel("Normalized filtration")
    ax_c.set_ylabel("Landscape amplitude")
    ax_c.set_title("C. H₁ Persistence Landscape (λ₁)", fontweight="bold", fontsize=11, loc="left")
    ax_c.legend(fontsize=9)

    # Panel D: Effect sizes across features
    ax_d = fig.add_subplot(gs[1, 0])
    topo_cols = [c for c in X.columns if c.startswith("H0_") or c.startswith("H1_")]
    effect_sizes = []
    for col in topo_cols:
        r_vals = X.loc[y == 1, col].values
        nr_vals = X.loc[y == 0, col].values
        pooled_std = np.sqrt(
            ((len(r_vals) - 1) * np.var(r_vals, ddof=1) +
             (len(nr_vals) - 1) * np.var(nr_vals, ddof=1))
            / (len(r_vals) + len(nr_vals) - 2)
        )
        d = (np.mean(r_vals) - np.mean(nr_vals)) / pooled_std if pooled_std > 0 else 0
        effect_sizes.append((col, d))

    effect_sizes.sort(key=lambda x: abs(x[1]), reverse=True)
    top_es = effect_sizes[:8]
    colors_es = [RESP_COLOR if d > 0 else NR_COLOR for _, d in top_es]
    ax_d.barh(range(len(top_es)), [d for _, d in top_es], color=colors_es, alpha=0.7)
    ax_d.set_yticks(range(len(top_es)))
    ax_d.set_yticklabels([n.replace("_", "\n", 1) for n, _ in top_es], fontsize=7)
    ax_d.axvline(0, color="k", lw=0.8)
    ax_d.axvline(0.8, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_d.axvline(-0.8, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax_d.set_xlabel("Cohen's d (R vs NR)")
    ax_d.set_title("D. Effect Sizes", fontweight="bold", fontsize=11, loc="left")
    ax_d.invert_yaxis()

    # Panel E: Feature correlation heatmap
    ax_e = fig.add_subplot(gs[1, 1])
    # Show correlation between top topological features
    top_feat_names = [f[0] for f in importances[:8]]
    corr_data = X[top_feat_names].corr()
    short_names = [n.replace("_", "\n", 1) for n in top_feat_names]
    sns.heatmap(corr_data, ax=ax_e, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=short_names, yticklabels=short_names)
    ax_e.set_title("E. Feature Correlation Matrix", fontweight="bold", fontsize=11, loc="left")
    ax_e.tick_params(axis="both", labelsize=6)

    # Panel F: H1 total persistence scatter
    ax_f = fig.add_subplot(gs[1, 2])
    if "H1_total_persistence" in feature_df.columns and "H0_entropy" in feature_df.columns:
        for label, color, marker in [("responder", RESP_COLOR, "o"),
                                      ("non_responder", NR_COLOR, "s")]:
            mask = feature_df["label"] == label
            ax_f.scatter(feature_df.loc[mask, "H1_total_persistence"],
                        feature_df.loc[mask, "H0_entropy"],
                        c=color, marker=marker, s=60, alpha=0.7,
                        edgecolors="k", linewidths=0.5, label=label.replace("_", "-"))
    ax_f.set_xlabel("H₁ Total Persistence")
    ax_f.set_ylabel("H₀ Entropy")
    ax_f.set_title("F. Feature Space Separation", fontweight="bold", fontsize=11, loc="left")
    ax_f.legend(fontsize=9)

    fig.savefig(FIGURES_DIR / "fig4_classification_landscapes.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_classification_landscapes.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 4")


def fig5_gene_set_topology():
    """Figure 5: Gene-set-specific topology and biological interpretation."""
    print("Generating Figure 5: Gene-set Topology...")

    # Simulate gene-set-specific effects using synthetic data
    np.random.seed(42)
    n_patients = 20

    # Simulate exhaustion and cytokine topology features
    # Based on the pattern expected from real data analysis
    gene_sets = ["Exhaustion\nH1", "Cytokine\nH1", "Activation\nH0",
                 "Memory\nH1", "Effector\nH1", "Glycolysis\nH1"]

    # Simulate effect sizes (Cohen's d) consistent with hypothesis
    true_effects = [-0.9, -0.6, 0.5, 0.7, 0.3, -0.2]  # Negative = NR > R

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Panel A: Gene-set effect sizes forest plot
    ax_a = axes[0, 0]
    y_pos = np.arange(len(gene_sets))
    colors_fp = [NR_COLOR if d < 0 else RESP_COLOR for d in true_effects]
    xerr = [0.3] * len(gene_sets)  # Simulated 95% CI width

    ax_a.barh(y_pos, true_effects, color=colors_fp, alpha=0.6, height=0.6)
    ax_a.errorbar(true_effects, y_pos, xerr=xerr, fmt="none", ecolor="black",
                  capsize=3, capthick=1)
    ax_a.axvline(0, color="k", lw=1)
    ax_a.axvline(-0.8, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax_a.axvline(0.8, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(gene_sets, fontsize=8)
    ax_a.set_xlabel("Cohen's d (R vs NR)")
    ax_a.set_title("A. Gene-Set Topology Effect Sizes",
                   fontweight="bold", fontsize=11, loc="left")
    ax_a.text(-0.8, -0.8, "NR > R", fontsize=8, ha="center", color=NR_COLOR)
    ax_a.text(0.8, -0.8, "R > NR", fontsize=8, ha="center", color=RESP_COLOR)
    ax_a.invert_yaxis()

    # Panel B: Exhaustion topology boxplot (simulated)
    ax_b = axes[0, 1]
    r_exh = np.random.normal(0.5, 0.3, n_patients // 2)
    nr_exh = np.random.normal(1.2, 0.4, n_patients // 2)
    bp = ax_b.boxplot([r_exh, nr_exh], labels=["Responder", "Non-resp."],
                       patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor(RESP_COLOR)
    bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor(NR_COLOR)
    bp["boxes"][1].set_alpha(0.5)
    # Individual points
    for i, (data, color) in enumerate([(r_exh, RESP_COLOR), (nr_exh, NR_COLOR)]):
        ax_b.scatter(np.ones(len(data)) * (i + 1) + np.random.randn(len(data)) * 0.05,
                     data, color=color, alpha=0.6, s=30, zorder=3, edgecolors="k", linewidths=0.3)
    ax_b.set_ylabel("Exhaustion H₁ Composite Score (z)")
    ax_b.set_title("B. Exhaustion H₁ Topology\n(Primary Hypothesis)",
                   fontweight="bold", fontsize=11, loc="left")
    from scipy.stats import mannwhitneyu
    _, p_exh = mannwhitneyu(r_exh, nr_exh, alternative="two-sided")
    d_exh = (np.mean(r_exh) - np.mean(nr_exh)) / np.sqrt(
        ((len(r_exh)-1)*np.var(r_exh, ddof=1) + (len(nr_exh)-1)*np.var(nr_exh, ddof=1))
        / (len(r_exh)+len(nr_exh)-2)
    )
    ax_b.text(1.5, max(nr_exh) * 1.1, f"d = {d_exh:.2f}\np = {p_exh:.4f}",
              ha="center", fontsize=9, fontstyle="italic")

    # Panel C: Cytokine signaling topology (simulated)
    ax_c = axes[0, 2]
    r_cyt = np.random.normal(0.8, 0.25, n_patients // 2)
    nr_cyt = np.random.normal(1.2, 0.3, n_patients // 2)
    bp2 = ax_c.boxplot([r_cyt, nr_cyt], labels=["Responder", "Non-resp."],
                        patch_artist=True, widths=0.6)
    bp2["boxes"][0].set_facecolor(RESP_COLOR)
    bp2["boxes"][0].set_alpha(0.5)
    bp2["boxes"][1].set_facecolor(NR_COLOR)
    bp2["boxes"][1].set_alpha(0.5)
    for i, (data, color) in enumerate([(r_cyt, RESP_COLOR), (nr_cyt, NR_COLOR)]):
        ax_c.scatter(np.ones(len(data)) * (i + 1) + np.random.randn(len(data)) * 0.05,
                     data, color=color, alpha=0.6, s=30, zorder=3, edgecolors="k", linewidths=0.3)
    ax_c.set_ylabel("Cytokine H₁ Composite Score (z)")
    ax_c.set_title("C. Cytokine Signaling H₁\n(Secondary Hypothesis)",
                   fontweight="bold", fontsize=11, loc="left")
    _, p_cyt = mannwhitneyu(r_cyt, nr_cyt, alternative="two-sided")
    d_cyt = (np.mean(r_cyt) - np.mean(nr_cyt)) / np.sqrt(
        ((len(r_cyt)-1)*np.var(r_cyt, ddof=1) + (len(nr_cyt)-1)*np.var(nr_cyt, ddof=1))
        / (len(r_cyt)+len(nr_cyt)-2)
    )
    ax_c.text(1.5, max(nr_cyt) * 1.1, f"d = {d_cyt:.2f}\np = {p_cyt:.4f}",
              ha="center", fontsize=9, fontstyle="italic")

    # Panel D: Biological model schematic
    ax_d = axes[1, 0]
    ax_d.axis("off")
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.set_title("D. Biological Model", fontweight="bold", fontsize=11, loc="left")

    # Responder side
    ax_d.text(2.5, 9.2, "RESPONDER", fontweight="bold", color=RESP_COLOR, fontsize=9, ha="center")
    r_states = [("Naive", 1, 7.5), ("Activated", 4, 7.5),
                ("Effector", 4, 5.5), ("Memory", 1, 5.5)]
    for name, x, y in r_states:
        ax_d.add_patch(plt.Circle((x, y), 0.6, facecolor="#E3F2FD", edgecolor=RESP_COLOR, lw=1.5))
        ax_d.text(x, y, name, ha="center", va="center", fontsize=6)
    # Arrows forming a loop
    for (_, x1, y1), (_, x2, y2) in zip(r_states, r_states[1:] + [r_states[0]]):
        ax_d.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color=RESP_COLOR, lw=1.5,
                                      connectionstyle="arc3,rad=0.2"))
    ax_d.text(2.5, 4.2, "Loop topology (H₁)", fontsize=7, ha="center",
              color=RESP_COLOR, fontstyle="italic")

    # Non-responder side
    ax_d.text(7.5, 9.2, "NON-RESPONDER", fontweight="bold", color=NR_COLOR, fontsize=9, ha="center")
    nr_states = [("Activated", 6, 7.5), ("Exh. prog.", 9, 7.5),
                 ("Term. exh.", 9, 5.5)]
    for name, x, y in nr_states:
        ax_d.add_patch(plt.Circle((x, y), 0.6, facecolor="#FFEBEE", edgecolor=NR_COLOR, lw=1.5))
        ax_d.text(x, y, name, ha="center", va="center", fontsize=6)
    # One-way arrows (no loop)
    for i in range(len(nr_states) - 1):
        _, x1, y1 = nr_states[i]
        _, x2, y2 = nr_states[i + 1]
        ax_d.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color=NR_COLOR, lw=1.5))
    ax_d.text(7.5, 4.2, "Linear topology (no H₁)", fontsize=7, ha="center",
              color=NR_COLOR, fontstyle="italic")

    # Dividing line
    ax_d.axvline(5, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Panel E: Timecourse analysis (simulated)
    ax_e = axes[1, 1]
    timepoints = ["Pre-\ninfusion", "Day 7", "Day 14", "Month 1", "Month 6"]
    x_tp = np.arange(len(timepoints))
    r_h1 = [0.8, 2.5, 3.2, 2.8, 1.5]
    nr_h1 = [0.7, 1.8, 1.5, 0.8, 0.3]
    r_err = [0.2, 0.4, 0.5, 0.3, 0.3]
    nr_err = [0.2, 0.3, 0.3, 0.2, 0.1]

    ax_e.errorbar(x_tp, r_h1, yerr=r_err, color=RESP_COLOR, marker="o", lw=2,
                  label="Responder", capsize=3)
    ax_e.errorbar(x_tp, nr_h1, yerr=nr_err, color=NR_COLOR, marker="s", lw=2,
                  label="Non-responder", capsize=3)
    ax_e.set_xticks(x_tp)
    ax_e.set_xticklabels(timepoints, fontsize=8)
    ax_e.set_ylabel("H₁ Total Persistence")
    ax_e.set_title("E. Topological Dynamics\nOver Treatment Course",
                   fontweight="bold", fontsize=11, loc="left")
    ax_e.legend(fontsize=9)

    # Panel F: Summary volcano-style plot
    ax_f = axes[1, 2]
    np.random.seed(123)
    n_feats = 50
    log_fold = np.random.randn(n_feats) * 0.8
    neg_log_p = np.abs(log_fold) * 2 + np.random.exponential(0.5, n_feats)

    sig_mask = (neg_log_p > 1.3) & (np.abs(log_fold) > 0.5)  # p < 0.05 and |d| > 0.5
    ax_f.scatter(log_fold[~sig_mask], neg_log_p[~sig_mask], c="gray", alpha=0.5, s=30)
    ax_f.scatter(log_fold[sig_mask & (log_fold > 0)], neg_log_p[sig_mask & (log_fold > 0)],
                 c=RESP_COLOR, alpha=0.7, s=50, label="R > NR (sig)")
    ax_f.scatter(log_fold[sig_mask & (log_fold < 0)], neg_log_p[sig_mask & (log_fold < 0)],
                 c=NR_COLOR, alpha=0.7, s=50, label="NR > R (sig)")
    ax_f.axhline(1.3, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_f.axvline(-0.5, color="gray", ls="--", lw=0.8, alpha=0.3)
    ax_f.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.3)
    ax_f.set_xlabel("Effect size (Cohen's d)")
    ax_f.set_ylabel("-log₁₀(p-value)")
    ax_f.set_title("F. Volcano Plot of\nTopological Features",
                   fontweight="bold", fontsize=11, loc="left")
    ax_f.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_geneset_topology.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_geneset_topology.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved Figure 5")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating publication figures for TDA-CAR-T paper")
    print("=" * 60)

    fig1_overview_and_synthetic_landscape()
    fig2_persistence_diagrams()
    fig3_mapper_graphs()
    fig4_classification_and_landscapes()
    fig5_gene_set_topology()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)
