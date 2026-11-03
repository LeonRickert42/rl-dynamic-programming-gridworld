"""Tabellarische Policy π(a|s)."""

from __future__ import annotations

import numpy as np


class Policy:
    """Policy als Tabelle ``probs[s, a] = π(a|s)`` (Zeilen summieren sich zu 1).

    Deterministische Policies sind der Spezialfall mit genau einer 1 pro Zeile.
    """

    def __init__(self, probs: np.ndarray):
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2 or not np.allclose(probs.sum(axis=1), 1.0):
            raise ValueError("probs muss (S, A) sein und Zeilensummen 1 haben")
        self.probs = probs

    @classmethod
    def uniform(cls, n_states: int, n_actions: int) -> Policy:
        """Zufalls-Policy: jede Aktion mit Wahrscheinlichkeit 1/|A|."""
        return cls(np.full((n_states, n_actions), 1.0 / n_actions))

    @classmethod
    def deterministic(cls, actions: np.ndarray, n_actions: int) -> Policy:
        probs = np.zeros((len(actions), n_actions))
        probs[np.arange(len(actions)), actions] = 1.0
        return cls(probs)

    @classmethod
    def greedy(cls, q: np.ndarray, tol: float = 1e-9, ties: str = "first") -> Policy:
        """Greedy bzgl. q[s, a]. ``ties="uniform"`` verteilt auf alle gleich guten Aktionen."""
        if ties == "uniform":
            mask = q >= q.max(axis=1, keepdims=True) - tol
            return cls(mask / mask.sum(axis=1, keepdims=True))
        return cls.deterministic(q.argmax(axis=1), q.shape[1])

    @property
    def n_states(self) -> int:
        return self.probs.shape[0]

    def actions(self) -> np.ndarray:
        """Wahrscheinlichste Aktion je Zustand."""
        return self.probs.argmax(axis=1)

    def is_deterministic(self) -> np.ndarray:
        return np.isclose(self.probs.max(axis=1), 1.0)

    def sample(self, s: int, rng: np.random.Generator) -> int:
        return int(rng.choice(self.probs.shape[1], p=self.probs[s]))

    def transition_matrix(self, env) -> np.ndarray:
        """P^π[s, s'] = Σ_a π(a|s) P[s, a, s'] (MDP + feste Policy = Markov Reward Process)."""
        return np.einsum("sa,sat->st", self.probs, env.P)

    def reward_vector(self, env) -> np.ndarray:
        """R^π[s] = Σ_a π(a|s) R[s, a]."""
        return (self.probs * env.R).sum(axis=1)

    def copy(self) -> Policy:
        return Policy(self.probs.copy())
