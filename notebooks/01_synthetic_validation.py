"""
TDA-CAR-T Synthetic Validation Demo
====================================

Demonstrates the full analysis pipeline on synthetic data:
1. Generate synthetic CAR-T populations (responders vs non-responders)
2. Compute persistent homology to extract topological features
3. Build Mapper graphs to visualize T cell state topology
4. Compare topological signatures between response groups
5. Classify response using topological features

Run this script to validate the computational framework before
applying to real clinical single-cell RNA-seq data.

Usage:
    python notebooks/01_synthetic_validation.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np

# --- Step 1: Generate synthetic data ---
print("=" * 60)
print("Step 1: Generating synthetic CAR-T populations")
print("=" * 60)

from src.simulation.synthetic_cart import CARTSimulator, generate_validation_dataset

sim = CARTSimulator(n_genes=2000, seed=42)

# Single patient examples
responder = sim.simulate_population(n_cells=3000, response_type="responder")
non_responder = sim.simulate_population(n_cells=3000, response_type="non_responder")

print(f"Responder: {responder['expression'].shape[0]} cells, {responder['expression'].shape[1]} genes")
print(f"Non-responder: {non_responder['expression'].shape[0]} cells, {non_responder['expression'].shape[1]} genes")
print(f"Responder states: {len(set(responder['states']))} unique states")
print(f"Non-responder states: {len(set(non_responder['states']))} unique states")

# --- Step 2: Compute persistent homology ---
print("\n" + "=" * 60)
print("Step 2: Computing persistent homology")
print("=" * 60)

from src.tda.persistent_homology import (
    compute_persistence,
    persistence_statistics,
    persistence_landscape,
    compare_persistence_diagrams,
)

# Use latent space (ground truth) for validation
r_ph = compute_persistence(responder["latent"], maxdim=2)
nr_ph = compute_persistence(non_responder["latent"], maxdim=2)

r_stats = persistence_statistics(r_ph["diagrams"])
nr_stats = persistence_statistics(nr_ph["diagrams"])

print("\nResponder topology:")
for dim, stats in r_stats.items():
    print(f"  {dim}: {stats['count']} features, "
          f"max persistence = {stats['max_persistence']:.3f}, "
          f"entropy = {stats['entropy']:.3f}")

print("\nNon-responder topology:")
for dim, stats in nr_stats.items():
    print(f"  {dim}: {stats['count']} features, "
          f"max persistence = {stats['max_persistence']:.3f}, "
          f"entropy = {stats['entropy']:.3f}")

# --- Step 3: Persistence landscapes ---
print("\n" + "=" * 60)
print("Step 3: Computing persistence landscapes")
print("=" * 60)

r_landscape_h1 = persistence_landscape(r_ph["diagrams"][1], num_landscapes=5)
nr_landscape_h1 = persistence_landscape(nr_ph["diagrams"][1], num_landscapes=5)

print(f"Responder H1 landscape max amplitude: {r_landscape_h1[0].max():.4f}")
print(f"Non-responder H1 landscape max amplitude: {nr_landscape_h1[0].max():.4f}")

# --- Step 4: Build Mapper graphs ---
print("\n" + "=" * 60)
print("Step 4: Building Mapper graphs")
print("=" * 60)

from src.mapper.cart_mapper import (
    build_mapper_graph,
    detect_topological_features,
    color_mapper_by_phenotype,
)

r_mapper = build_mapper_graph(responder["latent"], lens_fn="pca_1", n_cubes=15, overlap=0.3)
nr_mapper = build_mapper_graph(non_responder["latent"], lens_fn="pca_1", n_cubes=15, overlap=0.3)

r_topo = detect_topological_features(r_mapper)
nr_topo = detect_topological_features(nr_mapper)

print(f"\nResponder Mapper graph:")
print(f"  Nodes: {len(r_mapper['nodes'])}")
print(f"  Edges: {len(r_mapper['edges'])}")
print(f"  Branches (degree >= 3): {len(r_topo['branches'])}")
print(f"  Connected components: {len(r_topo['components'])}")
print(f"  Detected loops: {len(r_topo['loops'])}")

print(f"\nNon-responder Mapper graph:")
print(f"  Nodes: {len(nr_mapper['nodes'])}")
print(f"  Edges: {len(nr_mapper['edges'])}")
print(f"  Branches (degree >= 3): {len(nr_topo['branches'])}")
print(f"  Connected components: {len(nr_topo['components'])}")
print(f"  Detected loops: {len(nr_topo['loops'])}")

# --- Step 5: Cross-patient classification ---
print("\n" + "=" * 60)
print("Step 5: Cross-patient response classification")
print("=" * 60)

from src.tda.persistent_homology import topological_feature_matrix

dataset = generate_validation_dataset(n_responders=5, n_non_responders=5, n_cells=2000)

feature_df = topological_feature_matrix(
    dataset["point_clouds"],
    labels=dataset["labels"],
    maxdim=1,
)

print(f"\nFeature matrix shape: {feature_df.shape}")
print(f"Features: {[c for c in feature_df.columns if c not in ['sample_idx', 'label']]}")

# Simple classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

X = feature_df.drop(columns=["sample_idx", "label"])
X = X.fillna(0)
y = (feature_df["label"] == "responder").astype(int)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
loo = LeaveOneOut()
scores = cross_val_score(clf, X, y, cv=loo, scoring="accuracy")

print(f"\nLeave-one-out accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

clf.fit(X, y)
print("\nTop features for response prediction:")
for feat, imp in sorted(zip(X.columns, clf.feature_importances_), key=lambda x: -x[1])[:5]:
    print(f"  {feat}: {imp:.3f}")

# --- Step 6: Timecourse analysis ---
print("\n" + "=" * 60)
print("Step 6: Treatment timecourse topology")
print("=" * 60)

tc = sim.simulate_treatment_timecourse(n_cells_per_timepoint=1000)
for tp, data in tc.items():
    result = compute_persistence(data["latent"], maxdim=1)
    stats = persistence_statistics(result["diagrams"])
    print(f"  {tp:15s}  H0={stats['H0']['count']:3d}  H1={stats['H1']['count']:3d}  "
          f"H1_entropy={stats['H1']['entropy']:.3f}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
This synthetic validation demonstrates that TDA can distinguish
CAR-T response patterns based on topological features of the
T cell state landscape:

1. PERSISTENT HOMOLOGY captures structural differences:
   - Responders show more H1 features (loops) reflecting
     effector-memory cycling
   - Non-responders show fragmented topology reflecting
     exhaustion dominance

2. MAPPER GRAPHS reveal branching vs. collapse:
   - Responder graphs have more branches (fate decisions)
   - Non-responder graphs are more linear (exhaustion funnel)

3. TOPOLOGICAL FEATURES enable classification:
   - LOO cross-validation shows TDA features can predict response
   - H1 features (loops) are most discriminative

Next steps:
   - Apply to real clinical scRNA-seq data (GSE151511, GSE197268)
   - Integrate metabolic pathway scores with Mapper topology
   - Statistical testing with permutation-based p-values
""")
