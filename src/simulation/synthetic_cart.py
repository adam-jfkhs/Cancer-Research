"""Synthetic CAR-T data generator for method validation.

Generates synthetic single-cell data with known topological ground truth
to validate TDA methods before applying to real clinical data.
Simulates T cell differentiation trajectories, exhaustion dynamics,
and response heterogeneity with controllable parameters.
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Optional


class CARTSimulator:
    """Simulate synthetic CAR-T cell populations with known topology.

    Models T cell states as points in gene expression space with:
    - Branching trajectories (effector vs. memory fate)
    - Exhaustion gradients
    - Metabolic state transitions
    - Response vs. non-response population structure
    """

    def __init__(self, n_genes=2000, seed=42):
        self.n_genes = n_genes
        self.rng = np.random.RandomState(seed)

        # Define canonical T cell state centers in a reduced space
        self.state_centers = {
            "naive": np.array([0, 0, 0, 0, 0]),
            "activated": np.array([2, 1, 0, 0, 0]),
            "effector": np.array([4, 2, -1, 0, 0]),
            "memory_precursor": np.array([3, 0, 1, 1, 0]),
            "central_memory": np.array([2, -1, 2, 2, 0]),
            "effector_memory": np.array([3, 1, 1, 0, -1]),
            "exhausted_progenitor": np.array([3, 2, -1, -1, 1]),
            "terminally_exhausted": np.array([4, 3, -2, -2, 2]),
            "stem_memory": np.array([1, -1, 2, 3, 0]),
        }

    def simulate_population(
        self,
        n_cells=5000,
        response_type="responder",
        noise_level=0.3,
        include_trajectory=True,
    ):
        """Generate a synthetic CAR-T cell population.

        Parameters
        ----------
        n_cells : int
            Number of cells.
        response_type : str
            'responder' or 'non_responder'. Controls population composition.
        noise_level : float
            Expression noise level.
        include_trajectory : bool
            If True, add cells along differentiation trajectories.

        Returns
        -------
        dict
            'expression': np.ndarray (n_cells, n_genes),
            'latent': np.ndarray (n_cells, 5) - true latent positions,
            'states': list of str - ground truth state labels,
            'pseudotime': np.ndarray - differentiation pseudotime,
            'exhaustion': np.ndarray - exhaustion score,
            'topology_ground_truth': dict of expected topological features.
        """
        # Population composition depends on response type
        if response_type == "responder":
            composition = {
                "naive": 0.05,
                "activated": 0.15,
                "effector": 0.30,
                "memory_precursor": 0.15,
                "central_memory": 0.10,
                "effector_memory": 0.10,
                "exhausted_progenitor": 0.08,
                "terminally_exhausted": 0.02,
                "stem_memory": 0.05,
            }
            expected_topology = {
                "H0_components": 1,  # Connected population
                "H1_loops": 1,      # Effector-memory-exhaustion cycle
                "branching": True,   # Effector vs memory fate
            }
        else:
            composition = {
                "naive": 0.02,
                "activated": 0.05,
                "effector": 0.10,
                "memory_precursor": 0.05,
                "central_memory": 0.03,
                "effector_memory": 0.05,
                "exhausted_progenitor": 0.25,
                "terminally_exhausted": 0.40,
                "stem_memory": 0.05,
            }
            expected_topology = {
                "H0_components": 2,  # Disconnected: exhausted vs rest
                "H1_loops": 0,      # No renewal cycle
                "branching": False,  # Collapse toward exhaustion
            }

        latent_points = []
        state_labels = []
        pseudotimes = []
        exhaustion_scores = []

        for state, fraction in composition.items():
            n_state = int(n_cells * fraction)
            center = self.state_centers[state]

            points = self.rng.multivariate_normal(
                center,
                np.eye(5) * noise_level,
                size=n_state,
            )

            latent_points.append(points)
            state_labels.extend([state] * n_state)

            # Pseudotime based on distance from naive
            pt = np.linalg.norm(points - self.state_centers["naive"], axis=1)
            pseudotimes.append(pt)

            # Exhaustion score
            exh = np.linalg.norm(
                points - self.state_centers["terminally_exhausted"], axis=1
            )
            exhaustion_scores.append(1.0 / (1.0 + exh))

        if include_trajectory:
            traj_points, traj_states, traj_pt = self._generate_trajectories(
                n_cells // 5, noise_level
            )
            latent_points.append(traj_points)
            state_labels.extend(traj_states)
            pseudotimes.append(traj_pt)
            exh = np.linalg.norm(
                traj_points - self.state_centers["terminally_exhausted"], axis=1
            )
            exhaustion_scores.append(1.0 / (1.0 + exh))

        latent = np.vstack(latent_points)
        pseudotime = np.concatenate(pseudotimes)
        exhaustion = np.concatenate(exhaustion_scores)

        # Project latent space to high-dimensional gene expression
        expression = self._latent_to_expression(latent, noise_level)

        return {
            "expression": expression,
            "latent": latent,
            "states": state_labels,
            "pseudotime": pseudotime,
            "exhaustion": exhaustion,
            "topology_ground_truth": expected_topology,
        }

    def _generate_trajectories(self, n_points, noise_level):
        """Generate cells along differentiation trajectories.

        Creates smooth paths between T cell states to simulate
        continuous differentiation.
        """
        trajectories = [
            ("naive", "activated", "effector"),
            ("naive", "activated", "memory_precursor", "central_memory"),
            ("activated", "exhausted_progenitor", "terminally_exhausted"),
            ("memory_precursor", "effector_memory"),
        ]

        points = []
        states = []
        pseudotimes = []

        per_traj = n_points // len(trajectories)

        for traj in trajectories:
            for i in range(len(traj) - 1):
                start = self.state_centers[traj[i]]
                end = self.state_centers[traj[i + 1]]

                t_vals = self.rng.uniform(0, 1, per_traj // (len(traj) - 1))
                for t in t_vals:
                    point = start * (1 - t) + end * t
                    point += self.rng.normal(0, noise_level * 0.5, 5)
                    points.append(point)

                    if t < 0.3:
                        states.append(traj[i])
                    elif t > 0.7:
                        states.append(traj[i + 1])
                    else:
                        states.append(f"{traj[i]}_to_{traj[i+1]}")

                    pseudotimes.append(t + i)

        return np.array(points), states, np.array(pseudotimes)

    def _latent_to_expression(self, latent, noise_level):
        """Project 5D latent space to high-dimensional gene expression.

        Uses a random projection matrix (simulating gene regulatory
        network structure) plus noise.
        """
        projection = self.rng.randn(5, self.n_genes) * 0.5

        # Add structured gene programs
        # Effector genes respond to latent dims 0-1
        projection[0, :200] *= 3
        projection[1, :200] *= 2

        # Memory genes respond to latent dims 2-3
        projection[2, 200:400] *= 3
        projection[3, 200:400] *= 2

        # Exhaustion genes respond to latent dim 4
        projection[4, 400:600] *= 3

        # Metabolic genes
        projection[0, 600:800] *= 2  # Glycolysis ~ effector state
        projection[2, 800:1000] *= 2  # OXPHOS ~ memory state

        expression = latent @ projection
        expression += self.rng.randn(*expression.shape) * noise_level

        # Ensure non-negative (like real count data)
        expression = np.maximum(expression, 0)

        return expression

    def simulate_treatment_timecourse(
        self, n_cells_per_timepoint=2000, timepoints=None
    ):
        """Simulate CAR-T population dynamics over a treatment timecourse.

        Parameters
        ----------
        n_cells_per_timepoint : int
            Cells per timepoint.
        timepoints : list or None
            Timepoint labels. Default: ['pre_infusion', 'day7', 'day14',
            'day28', 'month3', 'month6'].

        Returns
        -------
        dict
            Timepoint -> simulation result.
        """
        if timepoints is None:
            timepoints = [
                "pre_infusion", "day7", "day14", "day28", "month3", "month6"
            ]

        # Evolving composition over time
        compositions_responder = {
            "pre_infusion": {"naive": 0.6, "stem_memory": 0.2, "central_memory": 0.2},
            "day7": {
                "activated": 0.4, "effector": 0.3, "naive": 0.1,
                "memory_precursor": 0.1, "exhausted_progenitor": 0.1,
            },
            "day14": {
                "effector": 0.35, "effector_memory": 0.2,
                "memory_precursor": 0.15, "exhausted_progenitor": 0.15,
                "activated": 0.1, "terminally_exhausted": 0.05,
            },
            "day28": {
                "effector_memory": 0.25, "central_memory": 0.2,
                "effector": 0.15, "exhausted_progenitor": 0.15,
                "terminally_exhausted": 0.1, "stem_memory": 0.1,
                "memory_precursor": 0.05,
            },
            "month3": {
                "central_memory": 0.3, "effector_memory": 0.25,
                "stem_memory": 0.15, "exhausted_progenitor": 0.1,
                "terminally_exhausted": 0.1, "effector": 0.1,
            },
            "month6": {
                "central_memory": 0.35, "stem_memory": 0.25,
                "effector_memory": 0.2, "exhausted_progenitor": 0.1,
                "terminally_exhausted": 0.05, "effector": 0.05,
            },
        }

        results = {}
        for tp in timepoints:
            if tp in compositions_responder:
                result = self.simulate_population(
                    n_cells=n_cells_per_timepoint,
                    response_type="responder",
                    include_trajectory=True,
                )
                result["timepoint"] = tp
                results[tp] = result

        return results


def generate_validation_dataset(n_responders=5, n_non_responders=5, n_cells=3000):
    """Generate a complete validation dataset with known ground truth.

    Parameters
    ----------
    n_responders : int
        Number of responder patients.
    n_non_responders : int
        Number of non-responder patients.
    n_cells : int
        Cells per patient.

    Returns
    -------
    dict
        'patients': list of simulation results,
        'labels': list of 'responder'/'non_responder',
        'point_clouds': list of np.ndarray for TDA,
        'expected_topology': dict of expected topological differences.
    """
    sim = CARTSimulator()
    patients = []
    labels = []
    point_clouds = []

    for i in range(n_responders):
        result = sim.simulate_population(
            n_cells=n_cells,
            response_type="responder",
            noise_level=0.3 + sim.rng.uniform(-0.05, 0.05),
        )
        patients.append(result)
        labels.append("responder")
        point_clouds.append(result["latent"])

    for i in range(n_non_responders):
        result = sim.simulate_population(
            n_cells=n_cells,
            response_type="non_responder",
            noise_level=0.3 + sim.rng.uniform(-0.05, 0.05),
        )
        patients.append(result)
        labels.append("non_responder")
        point_clouds.append(result["latent"])

    return {
        "patients": patients,
        "labels": labels,
        "point_clouds": point_clouds,
        "expected_topology": {
            "responders_more_connected": True,
            "responders_have_loops": True,
            "non_responders_more_fragmented": True,
        },
    }
