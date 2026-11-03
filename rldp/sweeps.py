"""Die drei Update-Schemata aus den Folien, unabhängig vom konkreten Backup-Operator.

* synchron:        zwei Kopien von V, jeder Sweep liest nur alte Werte
* in-place:        eine Kopie, neue Werte werden im selben Sweep sofort weiterverwendet
* prioritised:     immer den Zustand mit dem größten Bellman-Fehler zuerst updaten

Derselbe Code dient für Policy Evaluation (Bellman-Erwartungs-Backup) und
Value Iteration (Bellman-Optimalitäts-Backup) – nur die Backup-Funktion ist anders.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

BackupAll = Callable[[np.ndarray], np.ndarray]
BackupOne = Callable[[np.ndarray, int], float]


@dataclass
class RunResult:
    """Ergebnis + Konvergenz-Messdaten eines DP-Laufs."""

    method: str
    V: np.ndarray
    sweeps: float  # bei prioritised sweeping: Backups / Anzahl Zustände
    total_backups: int  # Anzahl Zustands-Updates
    evaluations: int  # Backups + reine Bellman-Fehler-Berechnungen (Rechenaufwand)
    converged: bool
    elapsed: float
    backups: list[int] = field(default_factory=list)  # x-Achse der Fehlerkurve
    errors: list[float] = field(default_factory=list)  # ||V - v_ref||_∞
    deltas: list[float] = field(default_factory=list)  # max. Änderung je Sweep (bzw. je Backup)
    backup_counts: np.ndarray | None = None  # wie oft wurde jeder Zustand aktualisiert
    snapshots: list[np.ndarray] = field(default_factory=list)  # V nach jedem Sweep
    policy: object | None = None


class _Recorder:
    def __init__(self, v_ref: np.ndarray | None):
        self.v_ref = v_ref
        self.backups: list[int] = []
        self.errors: list[float] = []

    def __call__(self, V: np.ndarray, backups: int) -> None:
        if self.v_ref is not None:
            self.backups.append(backups)
            self.errors.append(float(np.max(np.abs(V - self.v_ref))))


def run_synchronous(states, backup_all: BackupAll, V0, theta, max_sweeps, v_ref=None, keep_snapshots=False):
    start = time.perf_counter()
    V = V0.copy()
    counts = np.zeros(len(V), dtype=int)
    rec = _Recorder(v_ref)
    rec(V, 0)
    snapshots = [V.copy()] if keep_snapshots else []
    deltas, backups, converged, sweep = [], 0, False, 0
    for sweep in range(1, max_sweeps + 1):
        V_new = V.copy()
        V_new[states] = backup_all(V)[states]  # liest ausschließlich alte Werte
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        backups += len(states)
        counts[states] += 1
        deltas.append(delta)
        rec(V, backups)
        if keep_snapshots:
            snapshots.append(V.copy())
        if delta < theta:
            converged = True
            break
    return RunResult("synchronous", V, sweep, backups, backups, converged, time.perf_counter() - start,
                     rec.backups, rec.errors, deltas, counts, snapshots)


def run_in_place(states, backup_one: BackupOne, V0, theta, max_sweeps, v_ref=None, keep_snapshots=False):
    start = time.perf_counter()
    V = V0.copy()
    counts = np.zeros(len(V), dtype=int)
    rec = _Recorder(v_ref)
    rec(V, 0)
    snapshots = [V.copy()] if keep_snapshots else []
    deltas, backups, converged, sweep = [], 0, False, 0
    for sweep in range(1, max_sweeps + 1):
        delta = 0.0
        for s in states:
            new = backup_one(V, s)  # nutzt bereits aktualisierte Nachbarwerte
            delta = max(delta, abs(new - V[s]))
            V[s] = new
            backups += 1
            counts[s] += 1
            rec(V, backups)
        deltas.append(delta)
        if keep_snapshots:
            snapshots.append(V.copy())
        if delta < theta:
            converged = True
            break
    return RunResult("in_place", V, sweep, backups, backups, converged, time.perf_counter() - start,
                     rec.backups, rec.errors, deltas, counts, snapshots)


def run_prioritized(states, backup_one: BackupOne, predecessors, V0, theta, max_backups, v_ref=None,
                    keep_snapshots=False):
    start = time.perf_counter()
    V = V0.copy()
    n = len(states)
    updatable = np.zeros(len(V), dtype=bool)
    updatable[states] = True
    counts = np.zeros(len(V), dtype=int)
    priority = np.zeros(len(V))
    rec = _Recorder(v_ref)
    rec(V, 0)
    snapshots = [V.copy()] if keep_snapshots else []
    deltas, backups, error_evals = [], 0, 0
    heap: list[tuple[float, int]] = []

    def bellman_error(s: int) -> float:
        nonlocal error_evals
        error_evals += 1
        return abs(backup_one(V, s) - V[s])

    def push(s: int, err: float) -> None:
        priority[s] = err
        heapq.heappush(heap, (-err, s))

    for s in states:
        err = bellman_error(s)
        if err > theta:
            push(s, err)

    while heap and backups < max_backups:
        neg_err, s = heapq.heappop(heap)
        if -neg_err != priority[s]:
            continue  # veralteter Eintrag (Priorität wurde inzwischen geändert)
        priority[s] = 0.0
        new = backup_one(V, s)
        deltas.append(abs(new - V[s]))
        V[s] = new
        backups += 1
        counts[s] += 1
        rec(V, backups)
        if keep_snapshots and backups % n == 0:
            snapshots.append(V.copy())
        # Nur Vorgänger von s können durch die Änderung einen neuen Bellman-Fehler haben
        for p in predecessors[s]:
            if not updatable[p]:
                continue
            err = bellman_error(p)
            if err > theta:
                push(p, err)
            else:
                priority[p] = 0.0

    converged = not np.any(priority[states] > 0)
    if keep_snapshots:
        snapshots.append(V.copy())
    return RunResult("prioritized", V, backups / n, backups, backups + error_evals, converged,
                     time.perf_counter() - start, rec.backups, rec.errors, deltas, counts, snapshots)


def run(method, states, backup_all, backup_one, predecessors, V0, theta, max_sweeps, v_ref, keep_snapshots):
    if method == "synchronous":
        return run_synchronous(states, backup_all, V0, theta, max_sweeps, v_ref, keep_snapshots)
    if method == "in_place":
        return run_in_place(states, backup_one, V0, theta, max_sweeps, v_ref, keep_snapshots)
    if method == "prioritized":
        return run_prioritized(states, backup_one, predecessors, V0, theta, max_sweeps * len(states), v_ref,
                               keep_snapshots)
    raise ValueError(f"Unbekannte Methode {method!r}")
