"""Die Policy π: die Strategie des Agenten.

Eine Policy beantwortet für jedes Feld die Frage: "Was mache ich hier?"
Wir speichern sie als Tabelle mit einer Zeile pro Feld und einer Spalte pro Aktion:

            ↑     →     ↓     ←
    s=0   0.25  0.25  0.25  0.25     <- zufällige Policy: jede Richtung gleich oft
    s=1   0.00  1.00  0.00  0.00     <- deterministische Policy: immer nach rechts

probs[s, a] = π(a|s) = Wahrscheinlichkeit, in Feld s die Aktion a zu wählen.
Jede Zeile ergibt zusammen 1 (= 100 %).
"""

from __future__ import annotations

import numpy as np


class Policy:
    """Policy als Tabelle ``probs[s, a] = π(a|s)``."""

    def __init__(self, probs: np.ndarray):
        probs = np.asarray(probs, dtype=float)
        # Sicherheitscheck: Form (Felder x Aktionen), und jede Zeile muss sich zu 1 aufsummieren.
        if probs.ndim != 2 or not np.allclose(probs.sum(axis=1), 1.0):
            raise ValueError("probs muss (S, A) sein und Zeilensummen 1 haben")
        self.probs = probs

    # --- Verschiedene Arten, eine Policy zu erzeugen ----------------------------------------
    @classmethod
    def uniform(cls, n_states: int, n_actions: int) -> Policy:
        """Zufalls-Policy: Jede Aktion hat dieselbe Wahrscheinlichkeit (bei 4 Aktionen je 25 %)."""
        return cls(np.full((n_states, n_actions), 1.0 / n_actions))

    @classmethod
    def deterministic(cls, actions: np.ndarray, n_actions: int) -> Policy:
        """Feste Policy: In Feld s wird immer ``actions[s]`` gewählt (eine 1 pro Zeile)."""
        probs = np.zeros((len(actions), n_actions))
        probs[np.arange(len(actions)), actions] = 1.0
        return cls(probs)

    @classmethod
    def greedy(cls, q: np.ndarray, tol: float = 1e-9, ties: str = "first") -> Policy:
        """Gierige ("greedy") Policy: In jedem Feld die Aktion mit dem besten q-Wert nehmen.

        q[s, a] sagt, wie gut es ist, in Feld s die Aktion a zu machen.
        Bei ``ties="uniform"`` werden gleich gute Aktionen gleichmäßig verteilt.
        Das sieht man dann als mehrere Pfeile im Bild (wie auf den Folien).
        """
        if ties == "uniform":
            mask = q >= q.max(axis=1, keepdims=True) - tol  # True bei allen besten Aktionen
            return cls(mask / mask.sum(axis=1, keepdims=True))
        return cls.deterministic(q.argmax(axis=1), q.shape[1])  # argmax = Index des Maximums

    # --- Abfragen ---------------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        return self.probs.shape[0]

    def actions(self) -> np.ndarray:
        """Die wahrscheinlichste Aktion in jedem Feld."""
        return self.probs.argmax(axis=1)

    def is_deterministic(self) -> np.ndarray:
        """Pro Feld: Wird immer dieselbe Aktion gewählt (eine 1 in der Zeile)?"""
        return np.isclose(self.probs.max(axis=1), 1.0)

    def sample(self, s: int, rng: np.random.Generator) -> int:
        """Eine Aktion auswürfeln, so wie der Agent es tun würde."""
        return int(rng.choice(self.probs.shape[1], p=self.probs[s]))

    # --- MDP + Policy = Markov-Kette mit Belohnungen ----------------------------------------
    # Ist die Policy fest, gibt es keine Entscheidungen mehr. Dann reicht es zu wissen,
    # "von s nach s' mit welcher Wahrscheinlichkeit" und "welche Belohnung im Schnitt".
    def transition_matrix(self, env) -> np.ndarray:
        """P^π[s, s'] = Σ_a π(a|s) · P[s, a, s']  (Mittelwert über die Aktionen)."""
        return np.einsum("sa,sat->st", self.probs, env.P)

    def reward_vector(self, env) -> np.ndarray:
        """R^π[s] = Σ_a π(a|s) · R[s, a]  (erwartete Belohnung in Feld s)."""
        return (self.probs * env.R).sum(axis=1)

    def copy(self) -> Policy:
        return Policy(self.probs.copy())
