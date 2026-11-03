"""Policy Evaluation: "Wie gut ist meine Strategie π?"  (Prediction)

Für jedes Feld wollen wir V^π(s) wissen: Wie viel Belohnung sammle ich im Schnitt
noch ein, wenn ich ab hier immer nach π handle?

Bellman-Erwartungsgleichung, in Worten:
    Wert hier = Durchschnitt über meine Aktionen von
                (Belohnung für den Schritt + γ · Wert dort, wo ich lande)

    v_{k+1}(s) = Σ_a π(a|s) · ( R(s,a) + γ · Σ_s' P(s'|s,a) · v_k(s') )

Wendet man diese Formel immer wieder an, nähert sich v_k dem richtigen V^π.
"""

from __future__ import annotations

import time

import numpy as np

from .policy import Policy
from .sweeps import RunResult, run

METHODS = ("synchronous", "in_place", "prioritized")


def expectation_backups(env, policy: Policy):
    """Baut die Backup-Formel (Bellman-Erwartung) für eine feste Policy."""
    R_pi = policy.reward_vector(env)  # R^π[s]: erwartete Belohnung pro Feld
    P_pi = policy.transition_matrix(env)  # P^π[s, s']: wohin komme ich im Schnitt?
    gamma = env.gamma

    def backup_all(V):
        # Alle Felder auf einmal: v_neu = R^π + γ · P^π · v   (Matrix mal Vektor)
        return R_pi + gamma * P_pi @ V

    def backup_one(V, s):
        # Nur Feld s: R^π[s] + γ · (gewichtete Summe der Nachbarwerte)
        return R_pi[s] + gamma * P_pi[s] @ V

    return backup_all, backup_one


def exact_policy_evaluation(env, policy: Policy) -> np.ndarray:
    """Exakte Lösung ohne Iteration (als Referenz für die Fehlerkurven).

    Die Bellman-Gleichung v = R^π + γ P^π v ist ein lineares Gleichungssystem:
        (I - γ P^π) v = R^π
    Für kleine Welten kann man es einfach direkt lösen.
    """
    nt = env.nonterminal_states  # terminale Felder haben sowieso Wert 0
    P_pi = policy.transition_matrix(env)[np.ix_(nt, nt)]
    R_pi = policy.reward_vector(env)[nt]
    V = np.zeros(env.n_states)
    V[nt] = np.linalg.solve(np.eye(len(nt)) - env.gamma * P_pi, R_pi)
    return V


def policy_evaluation(
    env,
    policy: Policy,
    method: str = "synchronous",
    theta: float = 1e-6,
    max_sweeps: int = 100_000,
    V0: np.ndarray | None = None,
    v_ref: np.ndarray | None = None,
    keep_snapshots: bool = False,
) -> RunResult:
    """Iterative Policy Evaluation.

    method:  "synchronous", "in_place", "prioritized" oder "exact"
    theta:   Abbruch, sobald sich kein Wert mehr um mehr als theta ändert
    V0:      Startwerte (Standard: überall 0)
    v_ref:   wenn angegeben, wird der Abstand zu v_ref mitprotokolliert (für Grafiken)
    keep_snapshots: V nach jedem Sweep speichern (für Bilder/Animationen)
    """
    if method == "exact":
        start = time.perf_counter()
        V = exact_policy_evaluation(env, policy)
        return RunResult("exact", V, 0, 0, 0, True, time.perf_counter() - start)
    V0 = np.zeros(env.n_states) if V0 is None else np.asarray(V0, dtype=float)
    backup_all, backup_one = expectation_backups(env, policy)
    return run(method, env.nonterminal_states, backup_all, backup_one, env.predecessors, V0, theta, max_sweeps,
               v_ref, keep_snapshots)
