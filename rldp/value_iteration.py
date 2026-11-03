"""Value Iteration: direkt die bestmöglichen Werte V* berechnen  (Control).

Unterschied zur Policy Evaluation: Statt über die Aktionen der Policy zu MITTELN,
nehmen wir in jedem Feld einfach die BESTE Aktion (max).

Bellman-Optimalitätsgleichung, in Worten:
    Wert hier = beste Aktion von (Belohnung + γ · Wert dort, wo ich lande)

    v_{k+1}(s) = max_a ( R(s,a) + γ · Σ_s' P(s'|s,a) · v_k(s') )

Unterwegs gibt es keine explizite Policy. Erst am Ende liest man π* ab:
In jedem Feld die Aktion nehmen, die zum besten Nachbarn führt (greedy).
"""

from __future__ import annotations

import numpy as np

from .improvement import greedy_policy
from .sweeps import RunResult, run


def optimality_backups(env):
    """Baut die Backup-Formel (Bellman-Optimalität)."""
    R, P, gamma = env.R, env.P, env.gamma

    def backup_all(V):
        # Für alle Felder: q-Werte aller Aktionen berechnen, dann pro Feld das Maximum nehmen.
        return (R + gamma * P @ V).max(axis=1)

    def backup_one(V, s):
        # Nur Feld s: R[s, a] + γ · Σ P(s'|s,a) V(s') für alle a, davon das Maximum.
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
    """Value Iteration mit "synchronous", "in_place" oder "prioritized" Updates.

    ``result.V`` ist (ungefähr) V*, ``result.policy`` die daraus abgelesene optimale Policy π*.
    """
    V0 = np.zeros(env.n_states) if V0 is None else np.asarray(V0, dtype=float)
    backup_all, backup_one = optimality_backups(env)
    result = run(method, env.nonterminal_states, backup_all, backup_one, env.predecessors, V0, theta, max_sweeps,
                 v_ref, keep_snapshots)
    result.policy = greedy_policy(env, result.V)  # π* am Ende ablesen
    return result
