"""Policy Improvement und Policy Iteration: "Wie mache ich meine Strategie besser?"  (Control)

Idee: Ich kenne V (wie gut jedes Feld ist). Dann schaue ich in jedem Feld einen Schritt
voraus und frage für jede Aktion:
    q(s, a) = Belohnung für a  +  γ · Wert des Feldes, auf dem ich lande
Dann nehme ich einfach die beste Aktion ("greedy" = gierig):
    π'(s) = argmax_a q(s, a)

Policy Iteration wiederholt zwei Schritte, bis sich nichts mehr ändert:
    1. Evaluation:   Wie gut ist π?            -> V^π
    2. Improvement:  Greedy bzgl. V^π handeln  -> besseres π'
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .evaluation import policy_evaluation
from .policy import Policy


def q_values(env, V: np.ndarray) -> np.ndarray:
    """Einen Schritt vorausschauen: q[s, a] für alle Felder s und Aktionen a.

    env.P @ V rechnet für jedes (s, a) den erwarteten Wert des Folgefeldes aus.
    """
    return env.R + env.gamma * env.P @ V


def greedy_policy(env, V: np.ndarray, tol: float = 1e-9, ties: str = "first") -> Policy:
    """Die Policy, die in jedem Feld die beste Aktion bzgl. V wählt."""
    return Policy.greedy(q_values(env, V), tol=tol, ties=ties)


def policy_improvement(env, V: np.ndarray, policy: Policy, tol: float = 1e-9) -> tuple[Policy, int]:
    """Ein Verbesserungsschritt. Gibt (neue Policy, Anzahl geänderter Felder) zurück.

    n_changed == 0 bedeutet: Die Policy ist stabil, also optimal.
    (Auf den Folien ist die Rückgabe True/False an dieser Stelle widersprüchlich beschrieben.)

    Wichtig: Sind mehrere Aktionen gleich gut, BEHALTEN wir die alte. Sonst könnte die Policy
    ewig zwischen zwei gleich guten Aktionen hin und her springen und nie "stabil" werden.
    """
    q = q_values(env, V)
    is_best = q >= q.max(axis=1, keepdims=True) - tol  # Welche Aktionen sind (fast) optimal?
    old_actions = policy.actions()
    # Alte Aktion behalten, wenn die alte Policy deterministisch war und die Aktion noch optimal ist.
    keep = policy.is_deterministic() & is_best[np.arange(env.n_states), old_actions]
    new_actions = np.where(keep, old_actions, q.argmax(axis=1))
    new_policy = Policy.deterministic(new_actions, env.n_actions)

    # Zählen, in wie vielen (nicht-terminalen) Feldern sich etwas geändert hat.
    nt = env.nonterminal_states
    n_changed = int(np.any(new_policy.probs[nt] != policy.probs[nt], axis=1).sum())
    return new_policy, n_changed


@dataclass
class PolicyIterationResult:
    policy: Policy  # die gefundene (optimale) Policy
    V: np.ndarray  # ihre Werte
    iterations: int  # wie oft Evaluation + Improvement
    total_backups: int  # alle Backups in allen Evaluationen
    evaluations: int  # dazu noch die Improvement-Schritte (je Feld eine Vorausschau)
    elapsed: float
    history: list[dict] = field(default_factory=list)  # Zwischenstände für die Bilder
    backups: list[int] = field(default_factory=list)  # Fehlerkurve (x)
    errors: list[float] = field(default_factory=list)  # Fehlerkurve (y)


def policy_iteration(
    env,
    policy: Policy | None = None,
    eval_method: str = "in_place",
    theta: float = 1e-8,
    eval_max_sweeps: int | None = None,
    max_iterations: int = 10_000,
    v_ref: np.ndarray | None = None,
) -> PolicyIterationResult:
    """Policy Iteration (bzw. Generalised/Modified Policy Iteration).

    eval_max_sweeps=None: jede Evaluation läuft bis zur Konvergenz (klassisch).
    eval_max_sweeps=k:    Evaluation nach k Sweeps abbrechen ("Modified Policy Iteration").
                          Mit k = 1 ist das praktisch Value Iteration.
    """
    start = time.perf_counter()
    policy = policy or Policy.uniform(env.n_states, env.n_actions)  # Start: zufällige Policy
    V = np.zeros(env.n_states)
    max_sweeps = eval_max_sweeps or 100_000
    history, backups_curve, errors_curve = [], [], []
    total_backups = evaluations = 0
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        # 1. EVALUATION: Wie gut ist die aktuelle Policy?
        #    V0=V: Wir starten mit dem alten V ("Warmstart"), das spart viele Sweeps.
        ev = policy_evaluation(env, policy, eval_method, theta, max_sweeps, V0=V, v_ref=v_ref)
        V = ev.V
        backups_curve += [total_backups + b for b in ev.backups]
        errors_curve += ev.errors
        total_backups += ev.total_backups
        evaluations += ev.evaluations + len(env.nonterminal_states)

        # 2. IMPROVEMENT: in jedem Feld die beste Aktion bzgl. V wählen.
        new_policy, n_changed = policy_improvement(env, V, policy)
        history.append(dict(iteration=iteration, eval_sweeps=ev.sweeps, eval_backups=ev.total_backups,
                            changed=n_changed, V=V.copy(), policy=new_policy))
        policy = new_policy

        # 3. FERTIG? Die Policy ändert sich nicht mehr, und V ist genau genug.
        if n_changed == 0 and ev.converged:
            break

    return PolicyIterationResult(policy, V, iteration, total_backups, evaluations, time.perf_counter() - start,
                                 history, backups_curve, errors_curve)
