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

        # State centers spread far enough apart to create distinct topology.
        # Key design: responder states form a connected loop
        # (naive -> activated -> effector -> effector_memory -> central_memory -> naive),
        # while non-responder populations collapse into a disconnected
        # exhaustion island separated from a small functional remnant.
        self.state_centers = {
            "naive":                  np.array([0,   0,   0,   0,   0  ]),
            "activated":              np.array([3,   2,   0,   0,   0  ]),
            "effector":               np.array([6,   4,  -2,   0,   0  ]),
            "memory_precursor":       np.array([4,   0,   2,   2,   0  ]),
            "central_memory":         np.array([2,  -2,   4,   4,   0  ]),
            "effector_memory":        np.array([5,   2,   2,   0,  -2  ]),
            "exhausted_progenitor":   np.array([5,   4,  -2,  -3,   4  ]),
            "terminally_exhausted":   np.array([7,   6,  -4,  -5,   6  ]),
            "stem_memory":            np.array([1,  -2,   4,   6,   0  ]),
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
        if response_type == "responder":
            composition = {
                "naive": 0.05,
                "activated": 0.15,
                "effector": 0.25,
                "memory_precursor": 0.15,
                "central_memory": 0.10,
                "effector_memory": 0.10,
                "exhausted_progenitor": 0.10,
                "terminally_exhausted": 0.03,
                "stem_memory": 0.07,
            }
            # Responder trajectories form a loop connecting effector and memory arms
            trajectories = [
                ("naive", "activated", "effector"),
                ("naive", "activated", "memory_precursor", "central_memory"),
                ("activated", "exhausted_progenitor"),
                ("memory_precursor", "effector_memory"),
                ("effector_memory", "effector"),            # closes the loop
                ("central_memory", "stem_memory"),
            ]
            expected_topology = {
                "H0_components": 1,
                "H1_loops": 1,
                "branching": True,
            }
        else:
            composition = {
                "naive": 0.02,
                "activated": 0.03,
                "effector": 0.05,
                "memory_precursor": 0.02,
                "central_memory": 0.02,
                "effector_memory": 0.03,
                "exhausted_progenitor": 0.33,
                "terminally_exhausted": 0.45,
                "stem_memory": 0.05,
            }
            # Non-responder: only trajectory toward exhaustion, no loop back
            trajectories = [
                ("activated", "exhausted_progenitor", "terminally_exhausted"),
            ]
            expected_topology = {
                "H0_components": 2,
                "H1_loops": 0,
                "branching": False,
            }

        latent_points = []
        state_labels = []
        pseudotimes = []
        exhaustion_scores = []

        for state, fraction in composition.items():
            n_state = int(n_cells * fraction)
            if n_state == 0:
                continue
            center = self.state_centers[state]

            points = self.rng.multivariate_normal(
                center,
                np.eye(5) * noise_level,
                size=n_state,
            )

            latent_points.append(points)
            state_labels.extend([state] * n_state)

            pt = np.linalg.norm(points - self.state_centers["naive"], axis=1)
            pseudotimes.append(pt)

            exh = np.linalg.norm(
                points - self.state_centers["terminally_exhausted"], axis=1
            )
            exhaustion_scores.append(1.0 / (1.0 + exh))

        if include_trajectory:
            traj_points, traj_states, traj_pt = self._generate_trajectories(
                n_cells // 4, noise_level, trajectories
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

        expression = self._latent_to_expression(latent, noise_level)

        return {
            "expression": expression,
            "latent": latent,
            "states": state_labels,
            "pseudotime": pseudotime,
            "exhaustion": exhaustion,
            "topology_ground_truth": expected_topology,
        }

    def _generate_trajectories(self, n_points, noise_level, trajectories):
        """Generate cells along differentiation trajectories."""
        points = []
        states = []
        pseudotimes = []

        per_traj = max(1, n_points // len(trajectories))

        for traj in trajectories:
            for i in range(len(traj) - 1):
                start = self.state_centers[traj[i]]
                end = self.state_centers[traj[i + 1]]

                n_seg = max(1, per_traj // (len(traj) - 1))
                t_vals = self.rng.uniform(0, 1, n_seg)
                for t in t_vals:
                    point = start * (1 - t) + end * t
                    point += self.rng.normal(0, noise_level * 0.3, 5)
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
        """Project 5D latent space to high-dimensional gene expression."""
        projection = self.rng.randn(5, self.n_genes) * 0.5

        projection[0, :200] *= 3
        projection[1, :200] *= 2
        projection[2, 200:400] *= 3
        projection[3, 200:400] *= 2
        projection[4, 400:600] *= 3
        projection[0, 600:800] *= 2
        projection[2, 800:1000] *= 2

        expression = latent @ projection
        expression += self.rng.randn(*expression.shape) * noise_level
        expression = np.maximum(expression, 0)

        return expression

    def simulate_treatment_timecourse(
        self, n_cells_per_timepoint=2000, timepoints=None
    ):
        """Simulate CAR-T population dynamics over a treatment timecourse.

        Each timepoint uses a different composition reflecting the biological
        evolution of the CAR-T product in vivo.
        """
        if timepoints is None:
            timepoints = [
                "pre_infusion", "day7", "day14", "day28", "month3", "month6"
            ]

        compositions = {
            "pre_infusion": {
                "naive": 0.50, "stem_memory": 0.20, "central_memory": 0.20,
                "activated": 0.10,
            },
            "day7": {
                "activated": 0.35, "effector": 0.30, "naive": 0.05,
                "memory_precursor": 0.10, "exhausted_progenitor": 0.10,
                "effector_memory": 0.10,
            },
            "day14": {
                "effector": 0.30, "effector_memory": 0.20,
                "memory_precursor": 0.10, "exhausted_progenitor": 0.15,
                "activated": 0.05, "terminally_exhausted": 0.10,
                "central_memory": 0.10,
            },
            "day28": {
                "effector_memory": 0.20, "central_memory": 0.20,
                "effector": 0.10, "exhausted_progenitor": 0.15,
                "terminally_exhausted": 0.15, "stem_memory": 0.10,
                "memory_precursor": 0.10,
            },
            "month3": {
                "central_memory": 0.30, "effector_memory": 0.20,
                "stem_memory": 0.15, "exhausted_progenitor": 0.10,
                "terminally_exhausted": 0.10, "effector": 0.05,
                "naive": 0.10,
            },
            "month6": {
                "central_memory": 0.35, "stem_memory": 0.25,
                "effector_memory": 0.15, "exhausted_progenitor": 0.05,
                "terminally_exhausted": 0.05, "naive": 0.10,
                "effector": 0.05,
            },
        }

        # Trajectories evolve: early = activation-heavy, late = memory-heavy
        trajectory_sets = {
            "pre_infusion": [
                ("naive", "activated"),
            ],
            "day7": [
                ("naive", "activated", "effector"),
                ("activated", "memory_precursor"),
                ("activated", "exhausted_progenitor"),
            ],
            "day14": [
                ("activated", "effector"),
                ("memory_precursor", "effector_memory"),
                ("activated", "exhausted_progenitor", "terminally_exhausted"),
                ("effector_memory", "effector"),  # loop
            ],
            "day28": [
                ("memory_precursor", "central_memory"),
                ("effector_memory", "effector"),
                ("exhausted_progenitor", "terminally_exhausted"),
                ("central_memory", "stem_memory"),
            ],
            "month3": [
                ("central_memory", "stem_memory"),
                ("effector_memory", "central_memory"),
                ("central_memory", "stem_memory"),
            ],
            "month6": [
                ("central_memory", "stem_memory"),
                ("stem_memory", "central_memory"),  # homeostatic loop
            ],
        }

        results = {}
        for tp in timepoints:
            if tp not in compositions:
                continue

            comp = compositions[tp]
            trajs = trajectory_sets.get(tp, [])

            latent_points = []
            state_labels = []
            pseudotimes = []
            exhaustion_scores = []

            for state, fraction in comp.items():
                n_state = int(n_cells_per_timepoint * fraction)
                if n_state == 0:
                    continue
                center = self.state_centers[state]
                points = self.rng.multivariate_normal(
                    center, np.eye(5) * 0.3, size=n_state,
                )
                latent_points.append(points)
                state_labels.extend([state] * n_state)
                pseudotimes.append(
                    np.linalg.norm(points - self.state_centers["naive"], axis=1)
                )
                exh = np.linalg.norm(
                    points - self.state_centers["terminally_exhausted"], axis=1
                )
                exhaustion_scores.append(1.0 / (1.0 + exh))

            if trajs:
                traj_points, traj_states, traj_pt = self._generate_trajectories(
                    n_cells_per_timepoint // 4, 0.3, trajs
                )
                latent_points.append(traj_points)
                state_labels.extend(traj_states)
                pseudotimes.append(traj_pt)
                exh = np.linalg.norm(
                    traj_points - self.state_centers["terminally_exhausted"], axis=1
                )
                exhaustion_scores.append(1.0 / (1.0 + exh))

            latent = np.vstack(latent_points)
            expression = self._latent_to_expression(latent, 0.3)

            results[tp] = {
                "expression": expression,
                "latent": latent,
                "states": state_labels,
                "pseudotime": np.concatenate(pseudotimes),
                "exhaustion": np.concatenate(exhaustion_scores),
                "timepoint": tp,
            }

        return results


def generate_validation_dataset(n_responders=5, n_non_responders=5, n_cells=3000):
    """Generate a complete validation dataset with known ground truth.

    Each patient gets a slightly different seed to introduce inter-patient
    variability while preserving the core topological distinction.
    """
    patients = []
    labels = []
    point_clouds = []

    for i in range(n_responders):
        sim = CARTSimulator(seed=42 + i * 7)
        result = sim.simulate_population(
            n_cells=n_cells,
            response_type="responder",
            noise_level=0.25 + sim.rng.uniform(0, 0.10),
        )
        patients.append(result)
        labels.append("responder")
        point_clouds.append(result["latent"])

    for i in range(n_non_responders):
        sim = CARTSimulator(seed=142 + i * 7)
        result = sim.simulate_population(
            n_cells=n_cells,
            response_type="non_responder",
            noise_level=0.25 + sim.rng.uniform(0, 0.10),
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
