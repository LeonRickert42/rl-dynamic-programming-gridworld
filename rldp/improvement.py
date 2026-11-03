"""Policy Improvement und Policy Iteration (Control).

π'(s) = argmax_a q(s, a),   q(s, a) = R(s,a) + γ Σ_s' P(s'|s,a) v(s')
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .evaluation import policy_evaluation
from .policy import Policy


def q_values(env, V: np.ndarray) -> np.ndarray:
    """Einschritt-Vorausschau: q[s, a] für alle Zustände und Aktionen."""
    return env.R + env.gamma * env.P @ V


def greedy_policy(env, V: np.ndarray, tol: float = 1e-9, ties: str = "first") -> Policy:
    return Policy.greedy(q_values(env, V), tol=tol, ties=ties)


def policy_improvement(env, V: np.ndarray, policy: Policy, tol: float = 1e-9) -> tuple[Policy, int]:
    """Greedy-Verbesserung bzgl. V.

    Gibt die neue (deterministische) Policy und die Anzahl geänderter Zustände zurück.
    ``n_changed > 0`` heißt: Policy hat sich geändert (in den Folien ist die Rückgabe von
    ``policy_improvement`` hier widersprüchlich dokumentiert).
    Bei Gleichstand bleibt die alte Aktion erhalten – sonst kann Policy Iteration
    zwischen gleich guten Aktionen endlos hin und her springen.
    """
    q = q_values(env, V)
    is_best = q >= q.max(axis=1, keepdims=True) - tol
    old_actions = policy.actions()
    keep = policy.is_deterministic() & is_best[np.arange(env.n_states), old_actions]
    new_actions = np.where(keep, old_actions, q.argmax(axis=1))
    new_policy = Policy.deterministic(new_actions, env.n_actions)
    nt = env.nonterminal_states
    n_changed = int(np.any(new_policy.probs[nt] != policy.probs[nt], axis=1).sum())
    return new_policy, n_changed


@dataclass
class PolicyIterationResult:
    policy: Policy
    V: np.ndarray
    iterations: int
    total_backups: int  # Backups in allen Evaluationsphasen
    evaluations: int  # inkl. Improvement-Schritte (je Zustand eine volle Einschritt-Vorausschau)
    elapsed: float
    history: list[dict] = field(default_factory=list)
    backups: list[int] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)


def policy_iteration(
    env,
    policy: Policy | None = None,
    eval_method: str = "in_place",
    theta: float = 1e-8,
    eval_max_sweeps: int | None = None,
    max_iterations: int = 10_000,
    v_ref: np.ndarray | None = None,
) -> PolicyIterationResult:
    """(Generalised / Modified) Policy Iteration: evaluieren, greedy verbessern, wiederholen.

    ``eval_max_sweeps=None`` evaluiert bis zur Konvergenz (klassische Policy Iteration).
    ``eval_max_sweeps=k`` bricht die Evaluation nach k Sweeps ab (Modified Policy Iteration);
    k = 1 entspricht (fast) Value Iteration.
    """
    start = time.perf_counter()
    policy = policy or Policy.uniform(env.n_states, env.n_actions)
    V = np.zeros(env.n_states)
    max_sweeps = eval_max_sweeps or 100_000
    history, backups_curve, errors_curve = [], [], []
    total_backups = evaluations = 0
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        ev = policy_evaluation(env, policy, eval_method, theta, max_sweeps, V0=V, v_ref=v_ref)
        V = ev.V
        backups_curve += [total_backups + b for b in ev.backups]
        errors_curve += ev.errors
        total_backups += ev.total_backups
        evaluations += ev.evaluations + len(env.nonterminal_states)
        new_policy, n_changed = policy_improvement(env, V, policy)
        history.append(dict(iteration=iteration, eval_sweeps=ev.sweeps, eval_backups=ev.total_backups,
                            changed=n_changed, V=V.copy(), policy=new_policy))
        policy = new_policy
        if n_changed == 0 and ev.converged:
            break
    return PolicyIterationResult(policy, V, iteration, total_backups, evaluations, time.perf_counter() - start,
                                 history, backups_curve, errors_curve)
