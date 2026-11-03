"""Policy Evaluation: Wie gut ist eine gegebene Policy π?  (Prediction)

v_{k+1}(s) = Σ_a π(a|s) ( R(s,a) + γ Σ_s' P(s'|s,a) v_k(s') )      Bellman-Erwartungsgleichung
"""

from __future__ import annotations

import time

import numpy as np

from .policy import Policy
from .sweeps import RunResult, run

METHODS = ("synchronous", "in_place", "prioritized")


def expectation_backups(env, policy: Policy):
    """Bellman-Erwartungs-Backup für alle Zustände (vektorisiert) bzw. einen Zustand."""
    R_pi = policy.reward_vector(env)
    P_pi = policy.transition_matrix(env)
    gamma = env.gamma

    def backup_all(V):
        return R_pi + gamma * P_pi @ V

    def backup_one(V, s):
        return R_pi[s] + gamma * P_pi[s] @ V

    return backup_all, backup_one


def exact_policy_evaluation(env, policy: Policy) -> np.ndarray:
    """Exakte Lösung des linearen Gleichungssystems v = R^π + γ P^π v (Referenz)."""
    nt = env.nonterminal_states
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

    method: ``"synchronous"``, ``"in_place"``, ``"prioritized"`` oder ``"exact"``.
    theta:  Abbruch, sobald die größte Änderung (bzw. der größte Bellman-Fehler) < theta.
    v_ref:  optional; dann wird ||V - v_ref||_∞ nach jedem Update protokolliert.
    """
    if method == "exact":
        start = time.perf_counter()
        V = exact_policy_evaluation(env, policy)
        return RunResult("exact", V, 0, 0, 0, True, time.perf_counter() - start)
    V0 = np.zeros(env.n_states) if V0 is None else np.asarray(V0, dtype=float)
    backup_all, backup_one = expectation_backups(env, policy)
    return run(method, env.nonterminal_states, backup_all, backup_one, env.predecessors, V0, theta, max_sweeps,
               v_ref, keep_snapshots)
