"""Metabolic flux analysis integration with topological features.

Links T cell metabolic state (glycolysis, oxidative phosphorylation,
fatty acid oxidation) to topological structure of the CAR-T response
landscape. Uses gene expression as proxy for metabolic flux when
full FBA models are not available.
"""

import numpy as np
import pandas as pd
from typing import Optional


# Canonical metabolic gene sets for T cell states
METABOLIC_SIGNATURES = {
    "glycolysis": [
        "HK1", "HK2", "GPI", "PFKFB3", "PFKP", "ALDOA", "TPI1",
        "GAPDH", "PGK1", "PGAM1", "ENO1", "PKM", "LDHA", "SLC2A1",
        "SLC2A3",
    ],
    "oxidative_phosphorylation": [
        "NDUFA1", "NDUFB1", "SDHA", "SDHB", "UQCRC1", "UQCRC2",
        "COX5A", "COX5B", "ATP5F1A", "ATP5F1B", "ATP5MC1", "CS",
        "IDH2", "OGDH", "DLST",
    ],
    "fatty_acid_oxidation": [
        "CPT1A", "CPT2", "ACADM", "ACADL", "HADHA", "HADHB",
        "ACADVL", "ECHS1", "ACOX1",
    ],
    "glutaminolysis": [
        "GLS", "GLS2", "GLUD1", "GOT1", "GOT2", "SLC1A5", "SLC38A1",
    ],
    "pentose_phosphate": [
        "G6PD", "PGD", "TALDO1", "TKT", "RPIA",
    ],
}

# T cell state metabolic profiles from literature
TCELL_METABOLIC_PROFILES = {
    "naive": {"glycolysis": "low", "oxphos": "low", "fao": "low"},
    "effector": {"glycolysis": "high", "oxphos": "medium", "fao": "low"},
    "memory": {"glycolysis": "low", "oxphos": "high", "fao": "high"},
    "exhausted": {"glycolysis": "medium", "oxphos": "low", "fao": "low"},
    "stem_memory": {"glycolysis": "low", "oxphos": "medium", "fao": "high"},
}


def score_metabolic_pathways(adata, pathways=None):
    """Score cells on metabolic pathway activity.

    Parameters
    ----------
    adata : anndata.AnnData
        Processed single-cell data.
    pathways : dict or None
        Gene sets per pathway. Default: METABOLIC_SIGNATURES.

    Returns
    -------
    pd.DataFrame
        Cells x pathways score matrix.
    """
    import scanpy as sc

    if pathways is None:
        pathways = METABOLIC_SIGNATURES

    scores = {}
    for pathway_name, genes in pathways.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 3:
            scores[pathway_name] = np.zeros(adata.n_obs)
            continue

        sc.tl.score_genes(adata, gene_list=present, score_name=f"_{pathway_name}")
        scores[pathway_name] = adata.obs[f"_{pathway_name}"].values.copy()
        del adata.obs[f"_{pathway_name}"]

    return pd.DataFrame(scores, index=adata.obs_names)


def metabolic_state_classification(metabolic_scores):
    """Classify cells into metabolic states based on pathway scores.

    Uses a simple rule-based classifier based on relative pathway activity.

    Parameters
    ----------
    metabolic_scores : pd.DataFrame
        Output of score_metabolic_pathways.

    Returns
    -------
    pd.Series
        Metabolic state labels per cell.
    """
    states = []

    glyc = metabolic_scores.get("glycolysis", pd.Series(0, index=metabolic_scores.index))
    oxphos = metabolic_scores.get("oxidative_phosphorylation", pd.Series(0, index=metabolic_scores.index))
    fao = metabolic_scores.get("fatty_acid_oxidation", pd.Series(0, index=metabolic_scores.index))

    glyc_med = glyc.median()
    oxphos_med = oxphos.median()
    fao_med = fao.median()

    for i in range(len(metabolic_scores)):
        g = glyc.iloc[i]
        o = oxphos.iloc[i]
        f = fao.iloc[i]

        if g > glyc_med and o <= oxphos_med:
            states.append("effector_like")
        elif o > oxphos_med and f > fao_med and g <= glyc_med:
            states.append("memory_like")
        elif g <= glyc_med and o <= oxphos_med and f <= fao_med:
            states.append("quiescent")
        elif g > glyc_med and o > oxphos_med:
            states.append("activated")
        else:
            states.append("transitional")

    return pd.Series(states, index=metabolic_scores.index, name="metabolic_state")


def integrate_metabolic_topology(mapper_graph, metabolic_scores, cell_indices=None):
    """Map metabolic pathway scores onto Mapper graph nodes.

    Parameters
    ----------
    mapper_graph : dict
        From build_mapper_graph.
    metabolic_scores : pd.DataFrame
        From score_metabolic_pathways.
    cell_indices : np.ndarray or None
        If metabolic_scores doesn't align with mapper_graph cell indices.

    Returns
    -------
    pd.DataFrame
        Nodes x pathways: mean metabolic score per Mapper node.
    """
    node_metabolic = {}

    for node_id, node_info in mapper_graph["nodes"].items():
        members = node_info["members"]

        if cell_indices is not None:
            member_labels = cell_indices[members]
            node_scores = metabolic_scores.loc[member_labels].mean()
        else:
            node_scores = metabolic_scores.iloc[members].mean()

        node_metabolic[node_id] = node_scores

    return pd.DataFrame(node_metabolic).T


def metabolic_gradient_on_graph(mapper_graph, metabolic_node_scores, pathway):
    """Compute metabolic gradient along Mapper graph edges.

    Identifies regions of metabolic transition, which may correspond
    to T cell state transitions.

    Parameters
    ----------
    mapper_graph : dict
        Mapper graph.
    metabolic_node_scores : pd.DataFrame
        From integrate_metabolic_topology.
    pathway : str
        Which pathway to compute gradient for.

    Returns
    -------
    dict
        Edge -> gradient magnitude.
    """
    gradients = {}
    for n1, n2 in mapper_graph["edges"]:
        if n1 in metabolic_node_scores.index and n2 in metabolic_node_scores.index:
            v1 = metabolic_node_scores.loc[n1, pathway]
            v2 = metabolic_node_scores.loc[n2, pathway]
            gradients[(n1, n2)] = abs(v2 - v1)

    return gradients


def compute_metabolic_flux_proxy(adata, model_name="generic_human"):
    """Estimate metabolic fluxes from gene expression using COBRA.

    This is a simplified flux estimation — full FBA requires a
    genome-scale metabolic model and careful constraint setting.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell data.
    model_name : str
        COBRA model to use.

    Returns
    -------
    pd.DataFrame
        Estimated reaction fluxes per cell cluster.
    """
    try:
        import cobra
        from cobra.io import load_model
    except ImportError:
        raise ImportError("COBRApy required for flux estimation. pip install cobra")

    model = load_model(model_name)

    # Average expression per cluster for flux estimation
    if "leiden" not in adata.obs.columns:
        raise ValueError("Run clustering first (compute_embedding).")

    cluster_fluxes = {}
    for cluster_id in adata.obs["leiden"].unique():
        mask = adata.obs["leiden"] == cluster_id
        cluster_expr = np.asarray(adata[mask].X.mean(axis=0)).flatten()

        # Map gene expression to reaction bounds
        gene_expr = pd.Series(cluster_expr, index=adata.var_names)

        for rxn in model.reactions:
            if rxn.gene_reaction_rule:
                genes_in_rxn = [g.id for g in rxn.genes]
                expr_vals = [gene_expr.get(g, 0) for g in genes_in_rxn]
                if any(v > 0 for v in expr_vals):
                    scale = np.mean([v for v in expr_vals if v > 0])
                    rxn.upper_bound = min(rxn.upper_bound, scale * 100)

        solution = model.optimize()
        if solution.status == "optimal":
            cluster_fluxes[cluster_id] = solution.fluxes

    return pd.DataFrame(cluster_fluxes)
