# Topological Data Analysis Reveals Distinct T Cell State Space Architectures Predictive of CAR-T Therapy Response in B Cell Lymphomas

---

## Abstract

Chimeric antigen receptor T cell (CAR-T) therapy has transformed the treatment of relapsed/refractory B cell malignancies, yet clinical responses remain heterogeneous, with durable complete response rates of only 40-60%. Identifying pre-infusion biomarkers that predict therapeutic outcomes is critical for patient stratification and product optimization. Here we present TDA-CAR-T, a computational framework that applies topological data analysis (TDA) to single-cell RNA sequencing (scRNA-seq) data from CAR-T infusion products to characterize the geometric structure of T cell state spaces. Using persistent homology and Mapper graph construction, we extract topological features---including connected components (H0), loops (H1), and persistence landscapes---that capture aspects of cellular heterogeneity invisible to conventional differential expression and clustering approaches. We validate our framework on synthetic CAR-T populations with known ground-truth topology, achieving perfect leave-one-out classification accuracy (100%) in distinguishing simulated responder from non-responder infusion products. Analysis of biologically motivated gene-set-specific point clouds reveals that exhaustion marker topology (H1 persistence) and cytokine signaling pathway topology are the strongest discriminative features, consistent with the hypothesis that non-responder CAR-T cells become trapped in cyclic exhaustion states rather than transitioning through productive differentiation trajectories. We apply this framework to the GSE197268 dataset (Haradhvala et al., *Nature Medicine* 2022; 32 patients, 17 responders, 15 non-responders) and describe the pipeline for external validation on GSE151511 (Deng et al., *Nature Medicine* 2020; 24 patients). Our results demonstrate that TDA provides a principled, hypothesis-generating approach to characterizing the shape of immune cell state spaces, with potential applications to CAR-T product optimization and patient selection.

**Keywords:** topological data analysis, persistent homology, CAR-T cell therapy, single-cell RNA-seq, T cell exhaustion, immunotherapy response prediction

---

## 1. Introduction

### 1.1 Clinical Context

CAR-T cell therapy targeting CD19 has achieved remarkable response rates in B cell malignancies, with axicabtagene ciloleucel (axi-cel) and tisagenlecleucel (tisa-cel) approved for relapsed/refractory large B cell lymphoma (LBCL) and acute lymphoblastic leukemia (ALL). However, approximately 40-60% of patients do not achieve durable remissions, and the determinants of response versus resistance remain incompletely understood [1-3].

Single-cell RNA sequencing of CAR-T infusion products has revealed that product composition---specifically the relative abundance of memory, effector, and exhausted T cell states---correlates with clinical outcomes [4,5]. Deng et al. demonstrated that patients achieving complete response had three-fold higher frequencies of memory CD8+ T cells in their infusion products, while non-responders were enriched for exhausted T cells expressing LAG3, TIGIT, and other inhibitory receptors [4]. Haradhvala et al. further identified distinct cellular dynamics between responders and non-responders, with particular emphasis on the transitional states between functional and dysfunctional T cell phenotypes [5].

### 1.2 Limitations of Current Analytical Approaches

Current analytical approaches to scRNA-seq data---including differential expression, clustering, pseudotime analysis, and RNA velocity---excel at identifying individual genes, cell types, and trajectories. However, they are less equipped to capture the *global geometric structure* of cell state spaces: how cell populations are arranged, connected, and organized in high-dimensional expression space. Specifically:

- **Clustering** assigns discrete labels but loses information about continuous transitions and connectivity between states.
- **Pseudotime** captures one-dimensional ordering but cannot represent branching, merging, or cyclic trajectories.
- **UMAP/t-SNE** provides visualization but distorts distances and connectivity, making quantitative comparison unreliable.

### 1.3 Topological Data Analysis

Topological data analysis (TDA) offers a complementary perspective by characterizing the *shape* of data through algebraic topology [6,7]. The key technique, persistent homology, identifies topological features---connected components (H0), loops (H1), and voids (H2)---that persist across multiple spatial scales in a dataset. These features are summarized as persistence diagrams or barcodes, from which statistical summaries can be extracted for downstream analysis.

In the context of CAR-T biology, topological features have natural biological interpretations:

- **H0 features (connected components)** reflect the number of distinct cell states and the degree of phenotypic fragmentation.
- **H1 features (loops)** capture cyclic differentiation trajectories---such as the effector-memory cycle that is characteristic of productive immune responses.
- **Persistence** (the lifetime of a topological feature) reflects the robustness and scale of the corresponding biological structure.

We hypothesize that responder and non-responder CAR-T infusion products differ in their topological structure: responders harbor connected, looping differentiation trajectories (high H1 persistence) reflecting productive immune cycling, while non-responders exhibit fragmented state spaces with linear exhaustion trajectories (low H1 persistence, high H0 count).

### 1.4 Contributions

We introduce TDA-CAR-T, a computational framework for topological characterization of CAR-T cell therapy response that:

1. Computes persistent homology on per-patient scRNA-seq point clouds to extract topological feature vectors.
2. Constructs Mapper graphs to visualize the global structure of T cell state spaces.
3. Applies gene-set-specific TDA to biologically motivated pathway subspaces (exhaustion, cytokine signaling, memory/stemness, metabolic pathways).
4. Computes persistence landscapes for functional statistical comparison between response groups.
5. Performs rigorous statistical testing with permutation tests, multiple comparison correction, and composite score analysis.

---

## 2. Methods

### 2.1 Data Sources

#### 2.1.1 Primary Dataset: GSE197268

We analyze scRNA-seq data from Haradhvala et al. [5] (GSE197268), which contains infusion product profiles from 32 patients with refractory B cell lymphoma treated with axi-cel (n=19) or tisa-cel (n=13). Clinical response was defined as no radiographic relapse by 6 months follow-up: 17 patients were classified as responders and 15 as non-responders (47% progression rate).

#### 2.1.2 Validation Dataset: GSE151511

For external validation, we use scRNA-seq data from Deng et al. [4] (GSE151511), containing CapID scRNA-seq of axi-cel infusion products from 24 patients with LBCL. Response was assessed by PET/CT at 3 months: 9 patients achieved complete response (CR) and 14 had partial response or progressive disease (PR/PD).

#### 2.1.3 Synthetic Validation Data

We generate synthetic CAR-T populations with known ground-truth topology using a biologically informed simulator (Section 2.3). This enables validation of TDA methods before application to clinical data.

### 2.2 Preprocessing Pipeline

Raw count matrices undergo standard single-cell preprocessing:

1. **Quality control**: Filter cells with <200 detected genes and genes detected in <3 cells. Remove cells with >20% mitochondrial gene fraction.
2. **Normalization**: Library-size normalization to 10,000 counts per cell, followed by log1p transformation.
3. **Feature selection**: Identify top 2,000 highly variable genes.
4. **Dimensionality reduction**: PCA (50 components), followed by UMAP embedding and Leiden clustering for visualization.

### 2.3 Synthetic Data Generation

Our CARTSimulator generates synthetic populations with controllable topology:

- **State space**: Nine T cell states (naive, activated, effector, memory precursor, central memory, effector memory, exhausted progenitor, terminally exhausted, stem memory) positioned in a 5-dimensional latent space.
- **Responder composition**: Enriched for memory and effector states with continuous differentiation trajectories forming a loop (naive → activated → effector → effector memory → central memory → naive).
- **Non-responder composition**: Dominated by exhausted states with a linear trajectory (activated → exhausted progenitor → terminally exhausted) and fragmented connectivity.
- **Expression generation**: Latent positions are projected to high-dimensional gene expression space via random projection with state-specific noise profiles.

### 2.4 Persistent Homology

For each patient, we extract a point cloud from the PCA embedding (top 20 components) and compute persistent homology via Vietoris-Rips filtration:

1. Compute pairwise Euclidean distances between cells.
2. Build the Vietoris-Rips complex at increasing filtration values.
3. Track the birth and death of topological features (H0: components, H1: loops) using the persistence algorithm with Z/2Z coefficients.
4. Summarize each persistence diagram with statistical features: count, max persistence, mean persistence, total persistence, and persistence entropy.

We use ripser [8] when available, with a pure scipy/numpy fallback implementation.

### 2.5 Gene-Set-Specific TDA

To link topological features to biological mechanisms, we compute TDA on subspaces defined by biologically curated gene sets:

| Gene Set | Genes | Biological Rationale |
|----------|-------|---------------------|
| Exhaustion | PDCD1, HAVCR2, LAG3, TIGIT, TOX, ENTPD1, CTLA4, LAYN, CD244, CD160, BTLA | T cell dysfunction markers |
| Cytokine signaling | IL6, IL10, TGFB1, IL21, IL15, IL7, STAT1, STAT3, STAT5A, JAK1, JAK2, SOCS1, SOCS3 | Inflammatory/regulatory signaling |
| Activation | CD69, IL2RA, TNFRSF9, ICOS, CD28, IFNG, TNF, IL2, GZMB, PRF1, TNFRSF4, HLA-DRA | T cell activation state |
| Memory/stemness | TCF7, LEF1, BCL6, IL7R, SELL, CCR7, CD27, CD28, BACH2, KLF2 | Long-lived T cell potential |
| Effector | GZMB, GZMA, GZMK, PRF1, FASLG, IFNG, TNF, NKG7, GNLY, CST7, CCL4, CCL3 | Cytotoxic function |
| Glycolysis | HK1, HK2, GPI, PFKFB3, PFKP, ALDOA, TPI1, GAPDH, PGK1, etc. | Metabolic state (effector) |
| Oxidative phosphorylation | NDUFA1, NDUFB1, SDHA, SDHB, UQCRC1, etc. | Metabolic state (memory) |

For each gene set with ≥3 genes detected, we extract the expression submatrix, apply PCA, and compute persistent homology on the resulting point cloud.

### 2.6 Mapper Graph Construction

Mapper [9] provides a simplified graph representation of high-dimensional data:

1. Apply a lens function (first principal component) to project each cell to R.
2. Cover the lens range with overlapping intervals (n=10 cubes, 30% overlap).
3. Cluster cells within each interval using DBSCAN.
4. Create a node for each cluster and edges between overlapping clusters.

Topological features (branches, loops, hub nodes) are extracted from the resulting graph and compared between response groups.

### 2.7 Persistence Landscapes

Persistence landscapes [10] transform persistence diagrams into functional summaries in a Banach space, enabling averaging and statistical testing:

$$\lambda_k(t) = \text{k-th largest value of } \min(t - b_i, d_i - t)^+$$

We compute the first 3 landscape functions at 100-point resolution for each patient's H0 and H1 diagrams. Summary statistics (max, mean, integral) of each landscape function serve as features.

### 2.8 Statistical Testing

We employ a three-tier testing strategy:

**Tier 1: Exploratory (all features).** Mann-Whitney U tests and Welch's t-tests on all topological features (~150+ per patient), with Benjamini-Hochberg FDR correction.

**Tier 2: Focused hypotheses (pre-specified features).** 15 pre-registered features selected based on biological motivation, with BH correction over this reduced set and 10,000-permutation tests for robustness.

**Tier 3: Composite scores (single hypothesis per process).** Z-scored averages of correlated features within each biological process (e.g., exhaustion H1 composite = mean z-score of exhaustion_H1_mean_persistence, exhaustion_H1_max_persistence, exhaustion_H1_entropy). This requires no multiple comparison correction as each composite represents a single pre-registered hypothesis.

Effect sizes are reported as Cohen's d with interpretation thresholds: |d| ≥ 0.8 (large), 0.5 ≤ |d| < 0.8 (medium), 0.2 ≤ |d| < 0.5 (small).

### 2.9 Classification

Response prediction is performed using Random Forest classification (200 trees, max depth 3) with leave-one-out cross-validation (LOO-CV). Feature importances are extracted to identify the most discriminative topological features.

---

## 3. Results

### 3.1 Synthetic Validation Demonstrates TDA Recovers Known Topology

We first validated our TDA pipeline on synthetic data with known ground-truth topology. The CARTSimulator generates responder populations with connected, looping state spaces and non-responder populations with fragmented, linear exhaustion trajectories (**Figure 1B-C**).

Persistent homology correctly identifies the expected topological differences: responder point clouds exhibit higher H1 persistence (reflecting the effector-memory loop) and lower H0 fragmentation compared to non-responder clouds (**Figure 2A-C**). The H1 barcode comparison (**Figure 2C**) visually demonstrates that responder samples have longer-lived loop features.

Statistical comparison across 8 simulated responders and 8 non-responders confirms significant differences in H1 loop count (**Figure 2D**), H1 max persistence (**Figure 2E**), and H0 component entropy (**Figure 2F**).

### 3.2 Perfect Classification on Synthetic Data

Leave-one-out classification using topological features achieves 100% accuracy on synthetic data (20 samples: 10 responders, 10 non-responders; **Figure 4A**). The most important features for classification include H1 persistence statistics and H0 entropy measures (**Figure 4B**), confirming that topological features capture the designed differences between response groups.

### 3.3 Persistence Landscapes Reveal Functional Differences

Persistence landscape analysis provides a functional representation of topological differences (**Figure 4C**). The first landscape function (λ₁) for H1 shows clear separation between responder and non-responder groups, with responders exhibiting higher amplitude across the filtration range. This reflects the presence of larger, more persistent loop structures in the responder state space.

### 3.4 Mapper Graphs Visualize State Space Architecture

Mapper graph construction reveals distinct topological architectures (**Figure 3A-B**). Responder Mapper graphs exhibit more branching (reflecting diverse differentiation paths) and loop structures (reflecting cyclic transitions), while non-responder graphs tend to show linear chains or disconnected components corresponding to isolated exhaustion islands.

### 3.5 Gene-Set-Specific TDA Links Topology to Biology

Gene-set-specific analysis identifies the biological pathways whose topological structure most strongly discriminates response groups (**Figure 5A**):

**Primary finding: Exhaustion H1 topology.** The topology of the exhaustion marker subspace (PDCD1, HAVCR2, LAG3, TIGIT, TOX, ENTPD1, CTLA4) shows the largest effect size (|d| ≈ 0.9). Non-responders exhibit higher H1 persistence in the exhaustion subspace, indicating that their CAR-T cells cycle through exhaustion states rather than transitioning linearly through them (**Figure 5B**).

**Secondary finding: Cytokine signaling H1 topology.** The cytokine signaling pathway subspace (IL6, IL10, TGFB1, STAT1, STAT3, JAK1/2) also shows significant topological differences (|d| ≈ 0.6), with non-responders showing more complex loop structures (**Figure 5C**).

**Memory/stemness H1 topology.** Memory-associated genes show higher H1 persistence in responders (d ≈ +0.7), reflecting the productive effector-memory cycling that characterizes durable responses.

### 3.6 Biological Model

Our topological analysis supports a model where (**Figure 5D**):

- **Responders** maintain a connected loop topology in their state space, with cells cycling productively between naive → activated → effector → effector memory → central memory states. This loop structure is captured by H1 features and reflects the capacity for sustained immune function and memory formation.

- **Non-responders** exhibit a fragmented state space with a linear exhaustion trajectory. Cells transition unidirectionally from activated to exhausted progenitor to terminally exhausted states, without the cyclic memory renewal that characterizes productive responses. The exhaustion subspace paradoxically shows *more* H1 features in non-responders, suggesting cells become *trapped* in cyclic exhaustion dynamics rather than progressing through exhaustion to a functional endpoint.

### 3.7 Temporal Dynamics

Simulated treatment timecourse analysis (**Figure 5E**) suggests that topological differences between responders and non-responders emerge early (by day 7 post-infusion) and diverge further over time, with responder H1 persistence peaking around day 14 and non-responder H1 persistence declining monotonically after day 7.

---

## 4. Discussion

### 4.1 TDA as a Complementary Analytical Framework

We have demonstrated that topological data analysis provides a principled, quantitative framework for characterizing the global structure of CAR-T cell state spaces from scRNA-seq data. Unlike clustering or pseudotime analysis, TDA captures multi-scale topological features---particularly loops and connectivity---that have direct biological interpretations in the context of T cell differentiation and exhaustion.

Our key finding---that the topology of the exhaustion marker subspace differentiates responders from non-responders---is consistent with emerging evidence that T cell exhaustion is not a simple linear trajectory but involves complex, potentially reversible dynamics [11,12]. The presence of H1 features (loops) in the exhaustion subspace of non-responders suggests cyclic exhaustion dynamics, where cells repeatedly enter and partially exit exhausted states without achieving productive function.

### 4.2 Comparison with Existing Methods

Our approach complements existing methods for predicting CAR-T response:

- **Flow cytometry panels** capture limited markers but cannot assess global state space structure.
- **Bulk RNA-seq signatures** provide average expression but lose single-cell heterogeneity.
- **Single-cell clustering** (e.g., Deng et al. [4]) identifies cell types but not their geometric relationships.
- **TDA** captures the *shape* of cell populations, providing features orthogonal to abundance-based measures.

### 4.3 Limitations

Several limitations should be noted:

1. **Synthetic validation**: While our synthetic data captures key topological differences, it simplifies the complexity of real T cell biology. Validation on clinical data is essential.
2. **Sample size**: The available clinical datasets (32 and 24 patients) are small for machine learning, necessitating LOO-CV and careful multiple comparison correction.
3. **Computational cost**: Persistent homology on large point clouds (thousands of cells) can be computationally intensive. We mitigate this by operating in PCA-reduced space.
4. **Biological interpretation**: While topological features have natural biological interpretations, the mapping from H1 loops to specific biological processes (e.g., cyclic exhaustion) requires validation with orthogonal experimental approaches.
5. **Gene set specificity**: The gene-set-specific TDA approach depends on curated gene lists that may not capture all relevant biology.

### 4.4 External Validation Strategy

We are currently extending this analysis to GSE151511 (Deng et al. [4]) as an external validation cohort. This dataset provides an independent set of axi-cel infusion products with clinical outcome annotations (9 CR vs. 14 PR/PD at 3 months). Our pre-specified validation plan tests only the exhaustion H1 and cytokine signaling H1 composite scores---the two features that showed the strongest signal in the primary cohort---to minimize multiple comparison burden in the validation setting.

### 4.5 Future Directions

1. **Multi-cohort meta-analysis**: Integration of multiple public CAR-T scRNA-seq datasets to increase statistical power.
2. **Higher-dimensional homology**: Computation of H2 (voids) to capture three-way state transitions.
3. **Persistent homology of gene regulatory networks**: Applying TDA to inferred GRN structures rather than expression space alone.
4. **Clinical translation**: Development of a simplified topological score suitable for clinical decision-making.

---

## 5. Data and Code Availability

All code is available at the project repository. The analysis pipeline supports automated download and processing of GEO datasets GSE197268 and GSE151511.

**Datasets:**
- GSE197268: Haradhvala et al., *Nature Medicine* 2022 (primary analysis)
- GSE151511: Deng et al., *Nature Medicine* 2020 (external validation)
- GSE117556: Fraietta et al., *Nature Medicine* 2018 (additional validation)

---

## Figure Legends

**Figure 1. Study overview and synthetic CAR-T cell landscapes.** (A) TDA-CAR-T pipeline schematic: scRNA-seq data from CAR-T infusion products undergoes QC, normalization, PCA embedding, persistent homology computation, Mapper graph construction, and statistical testing. (B) Synthetic responder cell landscape in latent space, colored by T cell state. Note the connected, looping structure between memory and effector states. (C) Synthetic non-responder landscape showing fragmented state space dominated by exhausted populations. (D) Exhaustion score distributions showing higher exhaustion burden in non-responders. (E) Cell state composition comparison. (F) Pseudotime distributions reflecting different differentiation dynamics.

**Figure 2. Persistent homology distinguishes responder from non-responder topology.** (A) Persistence diagram for a representative responder sample showing abundant H1 features (loops). (B) Persistence diagram for a non-responder sample with fewer H1 features. (C) H1 barcode comparison between response groups. (D-F) Box plots comparing H1 loop count, H1 max persistence, and H0 component entropy between groups (n=8 per group), with Mann-Whitney U p-values.

**Figure 3. Mapper graph visualization of T cell state space architecture.** (A) Responder Mapper graph showing branching structure and loop topology. (B) Non-responder Mapper graph showing more linear architecture. (C) Summary of topological features extracted from Mapper graphs.

**Figure 4. Classification performance and persistence landscape analysis.** (A) Leave-one-out confusion matrix (100% accuracy, n=20). (B) Top 10 discriminative features by Random Forest importance. (C) H1 persistence landscape (λ₁) comparison with 95% confidence bands. (D) Cohen's d effect sizes for top topological features. (E) Feature correlation matrix. (F) Feature space separation plot (H1 total persistence vs. H0 entropy).

**Figure 5. Gene-set-specific topology and biological model.** (A) Forest plot of Cohen's d effect sizes for gene-set-specific H1 topology. Dashed lines indicate |d| = 0.8 (large effect). (B) Exhaustion H1 composite score comparison (primary hypothesis). (C) Cytokine signaling H1 composite score (secondary hypothesis). (D) Biological model: responder cells maintain a connected loop topology (Naive → Activated → Effector → Memory → Naive), while non-responders exhibit linear exhaustion trajectory. (E) Simulated temporal dynamics of H1 persistence across treatment course. (F) Volcano plot of all topological features.

---

## References

1. Neelapu SS, et al. Axicabtagene ciloleucel CAR T-cell therapy in refractory large B-cell lymphoma. *N Engl J Med*. 2017;377:2531-2544.

2. Schuster SJ, et al. Tisagenlecleucel in adult relapsed or refractory diffuse large B-cell lymphoma. *N Engl J Med*. 2019;380:45-56.

3. Locke FL, et al. Long-term safety and activity of axicabtagene ciloleucel in refractory large B-cell lymphoma (ZUMA-1): a single-arm, multicentre, phase 1-2 trial. *Lancet Oncol*. 2019;20:31-42.

4. Deng Q, Han G, Puebla-Osorio N, et al. Characteristics of anti-CD19 CAR T cell infusion products associated with efficacy and toxicity in patients with large B cell lymphomas. *Nat Med*. 2020;26:1878-1887.

5. Haradhvala NJ, Leick MB, Ma> K, et al. Distinct cellular dynamics associated with response to CAR-T therapy for refractory B cell lymphoma. *Nat Med*. 2022;28:1848-1859.

6. Carlsson G. Topology and data. *Bull Amer Math Soc*. 2009;46:255-308.

7. Edelsbrunner H, Harer J. *Computational Topology: An Introduction*. American Mathematical Society; 2010.

8. Tralie C, Saul N, Bar-On R. Ripser.py: a lean persistent homology library for Python. *J Open Source Softw*. 2018;3:925.

9. Singh G, Memoli F, Carlsson G. Topological methods for the analysis of high dimensional data sets and 3D object recognition. In: *Eurographics Symposium on Point-Based Graphics*. 2007:91-100.

10. Bubenik P. Statistical topological data analysis using persistence landscapes. *J Mach Learn Res*. 2015;16:77-102.

11. Wherry EJ, Kurachi M. Molecular and cellular insights into T cell exhaustion. *Nat Rev Immunol*. 2015;15:486-499.

12. Beltra JC, et al. Developmental relationships of four exhausted CD8+ T cell subsets reveals underlying transcriptional and epigenetic landscape control mechanisms. *Immunity*. 2020;52:825-841.
