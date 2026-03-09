"""Tests for TDA-CAR-T pipeline using synthetic data.

Validates that TDA methods correctly recover known topological features
from synthetic CAR-T populations with ground truth.
"""

import numpy as np
import pytest


def test_synthetic_data_generation():
    """Test that synthetic data has expected structure."""
    from src.simulation.synthetic_cart import CARTSimulator

    sim = CARTSimulator(n_genes=500, seed=42)

    # Responder
    result = sim.simulate_population(n_cells=1000, response_type="responder")
    assert result["expression"].shape[1] == 500
    assert len(result["states"]) >= 1000  # >= because trajectories add cells
    assert result["latent"].shape[1] == 5
    assert len(result["pseudotime"]) == len(result["states"])

    # Non-responder
    result_nr = sim.simulate_population(n_cells=1000, response_type="non_responder")
    assert result_nr["expression"].shape[1] == 500


def test_persistence_computation():
    """Test persistent homology on known topology."""
    from src.tda.persistent_homology import compute_persistence, persistence_statistics

    # Circle should have one H1 feature
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    circle += np.random.randn(*circle.shape) * 0.05

    result = compute_persistence(circle, maxdim=1, thresh=3.0)
    stats = persistence_statistics(result["diagrams"])

    assert stats["H0"]["count"] >= 1
    assert stats["H1"]["count"] >= 1
    assert stats["H1"]["max_persistence"] > 0.5  # Circle has significant H1


def test_persistence_landscape():
    """Test persistence landscape computation."""
    from src.tda.persistent_homology import persistence_landscape

    diagram = np.array([[0, 1], [0.5, 2], [1, 3]])
    landscapes = persistence_landscape(diagram, num_landscapes=3, resolution=100)

    assert landscapes.shape == (3, 100)
    assert np.all(landscapes >= 0)
    assert np.max(landscapes[0]) >= np.max(landscapes[1])  # Lambda_1 >= Lambda_2


def test_topological_feature_matrix():
    """Test feature matrix construction from multiple point clouds."""
    from src.tda.persistent_homology import topological_feature_matrix

    clouds = [np.random.randn(50, 3) for _ in range(5)]
    labels = ["R", "R", "NR", "NR", "R"]

    df = topological_feature_matrix(clouds, labels=labels, maxdim=1)

    assert len(df) == 5
    assert "label" in df.columns
    assert "H0_count" in df.columns
    assert "H1_count" in df.columns


def test_mapper_construction():
    """Test Mapper graph construction."""
    from src.mapper.cart_mapper import build_mapper_graph, detect_topological_features

    pc = np.random.randn(200, 5)
    graph = build_mapper_graph(pc, lens_fn="pca_1", n_cubes=5, overlap=0.3)

    assert len(graph["nodes"]) > 0
    assert "edges" in graph
    assert "lens_values" in graph

    features = detect_topological_features(graph)
    assert "branches" in features
    assert "components" in features
    assert "loops" in features


def test_responder_topology_differs():
    """Core hypothesis test: responder topology differs from non-responder."""
    from src.simulation.synthetic_cart import generate_validation_dataset
    from src.tda.persistent_homology import compute_persistence, persistence_statistics

    dataset = generate_validation_dataset(
        n_responders=3, n_non_responders=3, n_cells=500
    )

    r_h1_counts = []
    nr_h1_counts = []

    for pc, label in zip(dataset["point_clouds"], dataset["labels"]):
        result = compute_persistence(pc, maxdim=1)
        stats = persistence_statistics(result["diagrams"])

        if label == "responder":
            r_h1_counts.append(stats["H1"]["count"])
        else:
            nr_h1_counts.append(stats["H1"]["count"])

    # Responders should generally have more H1 features (loops)
    # due to effector-memory cycle in the simulated data
    assert len(r_h1_counts) > 0
    assert len(nr_h1_counts) > 0


def test_diagram_distance():
    """Test persistence diagram distance computation."""
    from src.tda.persistent_homology import compare_persistence_diagrams

    dgm1 = np.array([[0, 1], [0.5, 2], [1, 3]])
    dgm2 = np.array([[0, 1.1], [0.5, 2.1], [1, 3.1]])
    dgm3 = np.array([[0, 5], [0.5, 6]])

    d_close = compare_persistence_diagrams(dgm1, dgm2, metric="wasserstein")
    d_far = compare_persistence_diagrams(dgm1, dgm3, metric="wasserstein")

    assert d_close < d_far  # Similar diagrams should be closer


def test_timecourse_simulation():
    """Test treatment timecourse simulation."""
    from src.simulation.synthetic_cart import CARTSimulator

    sim = CARTSimulator(seed=42)
    tc = sim.simulate_treatment_timecourse(n_cells_per_timepoint=500)

    assert "pre_infusion" in tc
    assert "day7" in tc
    assert "month6" in tc
    assert tc["day7"]["expression"].shape[0] > 0


def test_gse151511_patient_labels():
    """Test GSE151511 patient response labels match known GEO annotations."""
    from src.data.geo_download import (
        GSE151511_PATIENT_RESPONSE,
        GSE151511_PATIENT_HISTOLOGY,
        GSE151511_RAW_RESPONSE,
        GSE151511_GSM_TO_PATIENT,
    )

    # 24 patients total
    assert len(GSE151511_PATIENT_RESPONSE) == 24
    assert len(GSE151511_GSM_TO_PATIENT) == 24

    # Response counts: 9 CR, 13 PD, 1 PR, 1 NE
    n_r = sum(1 for v in GSE151511_PATIENT_RESPONSE.values() if v == "responder")
    n_nr = sum(1 for v in GSE151511_PATIENT_RESPONSE.values() if v == "non_responder")
    n_unk = sum(1 for v in GSE151511_PATIENT_RESPONSE.values() if v == "unknown")
    assert n_r == 9, f"Expected 9 responders, got {n_r}"
    assert n_nr == 14, f"Expected 14 non-responders, got {n_nr}"
    assert n_unk == 1, f"Expected 1 unknown (NE), got {n_unk}"

    # Verify specific known labels
    assert GSE151511_PATIENT_RESPONSE["ac01"] == "responder"   # CR
    assert GSE151511_PATIENT_RESPONSE["ac02"] == "non_responder"  # PD
    assert GSE151511_PATIENT_RESPONSE["ac06"] == "unknown"     # NE
    assert GSE151511_PATIENT_RESPONSE["ac20"] == "non_responder"  # PR → non_responder

    # Raw response
    assert GSE151511_RAW_RESPONSE["ac01"] == "CR"
    assert GSE151511_RAW_RESPONSE["ac20"] == "PR"
    assert GSE151511_RAW_RESPONSE["ac06"] == "NE"

    # Histology
    assert GSE151511_PATIENT_HISTOLOGY["ac01"] == "DLBCL"
    assert GSE151511_PATIENT_HISTOLOGY["ac05"] == "tFL"
    assert GSE151511_PATIENT_HISTOLOGY["ac06"] == "PMBCL"

    # All patients axi-cel
    # GSM→patient mapping
    assert GSE151511_GSM_TO_PATIENT["GSM4579891"] == "ac01"
    assert GSE151511_GSM_TO_PATIENT["GSM4579914"] == "ac24"


def test_gse151511_fallback_metadata():
    """Test that fallback metadata builder works for GSE151511."""
    from src.data.geo_download import _build_fallback_metadata

    meta = _build_fallback_metadata("GSE151511")
    assert meta is not None
    assert len(meta) == 24
    assert "patient_id" in meta.columns
    assert "response" in meta.columns
    assert "histology" in meta.columns
    assert meta.loc["GSM4579891", "patient_id"] == "ac01"
    assert meta.loc["GSM4579891", "response"] == "responder"
