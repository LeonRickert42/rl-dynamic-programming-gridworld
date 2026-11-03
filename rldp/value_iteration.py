"""Value Iteration: direkt die optimale Wertfunktion v* berechnen (Control).

v_{k+1}(s) = max_a ( R(s,a) + γ Σ_s' P(s'|s,a) v_k(s') )      Bellman-Optimalitätsgleichung

Es gibt unterwegs keine explizite Policy; π* wird am Ende greedy aus v* abgelesen.
"""

from __future__ import annotations

import numpy as np

from .improvement import greedy_policy
from .sweeps import RunResult, run


def optimality_backups(env):
    R, P, gamma = env.R, env.P, env.gamma

    def backup_all(V):
        return (R + gamma * P @ V).max(axis=1)

    def backup_one(V, s):
        return float((R[s] + gamma * P[s] @ V).max())

    return backup_all, backup_one


def value_iteration(
    env,
    method: str = "synchronous",
    theta: float = 1e-6,
    max_sweeps: int = 100_000,
    V0: np.ndarray | None = None,
    v_ref: np.ndarray | None = None,
    keep_snapshots: bool = False,
) -> RunResult:
    """Value Iteration mit ``"synchronous"``, ``"in_place"`` oder ``"prioritized"`` Updates.

    Das Ergebnis enthält ``result.policy`` = greedy Policy bzgl. des finalen V.
    """
    V0 = np.zeros(env.n_states) if V0 is None else np.asarray(V0, dtype=float)
    backup_all, backup_one = optimality_backups(env)
    result = run(method, env.nonterminal_states, backup_all, backup_one, env.predecessors, V0, theta, max_sweeps,
                 v_ref, keep_snapshots)
    result.policy = greedy_policy(env, result.V)
    return result
