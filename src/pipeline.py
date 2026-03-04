"""Main analysis pipeline for TDA-CAR-T.

Orchestrates the full workflow: data loading, preprocessing, TDA computation,
Mapper construction, metabolic integration, and statistical comparison.
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.tda.persistent_homology import (
    compute_persistence,
    persistence_statistics,
    persistence_landscape,
    compare_persistence_diagrams,
    topological_feature_matrix,
)
from src.mapper.cart_mapper import (
    build_mapper_graph,
    color_mapper_by_phenotype,
    detect_topological_features,
    integrate_metabolic_topology,
)
from src.metabolic.flux_analysis import (
    score_metabolic_pathways,
    metabolic_state_classification,
    integrate_metabolic_topology as metabolic_topo_integration,
)
from src.simulation.synthetic_cart import CARTSimulator, generate_validation_dataset


def load_config(config_path="configs/default_config.yaml"):
    """Load analysis configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_synthetic_validation(config=None):
    """Run full pipeline on synthetic data to validate methods.

    This is the primary entry point for computational validation
    without wet lab data.

    Returns
    -------
    dict
        Complete analysis results including topological features,
        classification accuracy, and ground truth comparison.
    """
    if config is None:
        config = load_config()

    sim_cfg = config.get("simulation", {})
    tda_cfg = config.get("tda", {})
    mapper_cfg = config.get("mapper", {})

    print("Generating synthetic CAR-T data...")
    dataset = generate_validation_dataset(
        n_responders=sim_cfg.get("n_responders", 5),
        n_non_responders=sim_cfg.get("n_non_responders", 5),
        n_cells=sim_cfg.get("n_cells", 3000),
    )

    print("Computing persistent homology per patient...")
    all_stats = []
    all_diagrams = []
    for i, pc in enumerate(dataset["point_clouds"]):
        result = compute_persistence(
            pc,
            maxdim=tda_cfg.get("maxdim", 2),
        )
        stats = persistence_statistics(result["diagrams"])
        all_stats.append(stats)
        all_diagrams.append(result["diagrams"])

    # Build topological feature matrix
    print("Building topological feature matrix...")
    feature_df = topological_feature_matrix(
        dataset["point_clouds"],
        labels=dataset["labels"],
        maxdim=tda_cfg.get("maxdim", 1),
    )

    # Classify responders vs non-responders using topological features
    print("Classifying response using topological features...")
    classification = _classify_response(feature_df)

    # Build Mapper graphs for representative patients
    print("Constructing Mapper graphs...")
    mapper_results = {}
    for label in ["responder", "non_responder"]:
        idx = dataset["labels"].index(label)
        pc = dataset["point_clouds"][idx]
        mapper_graph = build_mapper_graph(
            pc,
            lens_fn=mapper_cfg.get("lens_fn", "pca_1"),
            n_cubes=mapper_cfg.get("n_cubes", 15),
            overlap=mapper_cfg.get("overlap", 0.3),
        )
        topo_features = detect_topological_features(mapper_graph)
        mapper_results[label] = {
            "graph": mapper_graph,
            "features": topo_features,
        }

    # Compare with ground truth
    print("Validating against ground truth...")
    validation = _validate_topology(
        all_stats, dataset["labels"], dataset["expected_topology"]
    )

    return {
        "feature_matrix": feature_df,
        "persistence_stats": all_stats,
        "diagrams": all_diagrams,
        "classification": classification,
        "mapper_results": mapper_results,
        "validation": validation,
        "dataset": dataset,
    }


def run_timecourse_analysis(config=None):
    """Analyze topological evolution over CAR-T treatment timecourse.

    Returns
    -------
    dict
        Timecourse topology results.
    """
    if config is None:
        config = load_config()

    sim = CARTSimulator(seed=config.get("simulation", {}).get("seed", 42))
    timecourse = sim.simulate_treatment_timecourse()

    tda_cfg = config.get("tda", {})
    timecourse_stats = {}

    for tp, data in timecourse.items():
        result = compute_persistence(
            data["latent"],
            maxdim=tda_cfg.get("maxdim", 2),
        )
        timecourse_stats[tp] = persistence_statistics(result["diagrams"])

    return {
        "timecourse_data": timecourse,
        "timecourse_stats": timecourse_stats,
    }


def _classify_response(feature_df):
    """Simple classification of responder vs non-responder using TDA features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    X = feature_df.drop(columns=["sample_idx", "label"], errors="ignore")
    # Remove columns that are all NaN or all zero
    X = X.loc[:, (X != 0).any(axis=0)]
    X = X.fillna(0)

    y = (feature_df["label"] == "responder").astype(int)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo, scoring="accuracy")

    # Fit final model for feature importances
    clf.fit(X, y)

    return {
        "accuracy": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "feature_importances": dict(zip(X.columns, clf.feature_importances_)),
        "n_samples": len(y),
    }


def _validate_topology(stats_list, labels, expected):
    """Compare observed topology against expected ground truth."""
    responder_h1 = []
    non_responder_h1 = []

    for stats, label in zip(stats_list, labels):
        h1_count = stats.get("H1", {}).get("count", 0)
        if label == "responder":
            responder_h1.append(h1_count)
        else:
            non_responder_h1.append(h1_count)

    responders_have_more_loops = (
        np.mean(responder_h1) > np.mean(non_responder_h1)
        if responder_h1 and non_responder_h1
        else None
    )

    return {
        "responder_H1_mean": float(np.mean(responder_h1)) if responder_h1 else 0,
        "non_responder_H1_mean": float(np.mean(non_responder_h1)) if non_responder_h1 else 0,
        "responders_have_more_loops": responders_have_more_loops,
        "matches_expected": responders_have_more_loops == expected.get("responders_have_loops"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TDA-CAR-T: Topological Analysis of CAR-T Cell Response")
    print("=" * 60)

    results = run_synthetic_validation()

    print(f"\nClassification accuracy: {results['classification']['accuracy']:.2f} "
          f"(+/- {results['classification']['accuracy_std']:.2f})")
    print(f"Validation: topology matches expected = {results['validation']['matches_expected']}")
    print(f"  Responder H1 (loops): {results['validation']['responder_H1_mean']:.1f}")
    print(f"  Non-responder H1 (loops): {results['validation']['non_responder_H1_mean']:.1f}")

    print("\nTop topological features for response prediction:")
    importances = results["classification"]["feature_importances"]
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
        print(f"  {feat}: {imp:.3f}")
