"""Die drei Arten, die Werte V(s) zu aktualisieren (Folien: "DP Calculation Methods").

Ein "Backup" heißt: Für EIN Feld s wird ein neuer Wert V(s) aus den Werten der Nachbarn berechnet.
Die Formel für dieses Backup (z. B. Bellman-Erwartung oder Bellman-Optimalität) wird von außen
als Funktion übergeben. Deshalb funktioniert dieser Code für Policy Evaluation UND Value Iteration.

  1) SYNCHRON      Zwei Tafeln: Man liest nur von der alten und schreibt auf die neue.
                   Erst wenn alle Felder fertig sind, wird die neue Tafel zur alten.
  2) IN-PLACE      Eine Tafel: Ein neuer Wert wird sofort überschrieben und von den
                   nächsten Feldern im selben Durchlauf schon benutzt. Meist schneller.
  3) PRIORITISED   Keine feste Reihenfolge: Man aktualisiert immer das Feld, dessen Wert
     SWEEPING      "am falschesten" ist (größter Bellman-Fehler). Eine Prioritätswarteschlange
                   (Heap) merkt sich, welches Feld gerade am dringendsten ist.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Typen der Backup-Funktionen:
#   backup_all(V)   -> neue Werte für ALLE Felder auf einmal (für synchron)
#   backup_one(V,s) -> neuer Wert für EIN Feld s (für in-place und prioritised)
BackupAll = Callable[[np.ndarray], np.ndarray]
BackupOne = Callable[[np.ndarray, int], float]


@dataclass
class RunResult:
    """Ergebnis eines Laufs und alle Messwerte, die wir für die Grafiken brauchen."""

    method: str  # "synchronous", "in_place", "prioritized" oder "exact"
    V: np.ndarray  # die berechneten Werte, einer pro Feld
    sweeps: float  # Anzahl Durchläufe (bei prioritised: Backups / Anzahl Felder)
    total_backups: int  # wie viele Einzel-Updates insgesamt
    evaluations: int  # Rechenaufwand: Backups + reine Fehlerberechnungen (bei prioritised)
    converged: bool  # wurde die Genauigkeit theta erreicht?
    elapsed: float  # Laufzeit in Sekunden
    backups: list[int] = field(default_factory=list)  # x-Werte der Fehlerkurve
    errors: list[float] = field(default_factory=list)  # y-Werte: größter Fehler zur exakten Lösung
    deltas: list[float] = field(default_factory=list)  # größte Änderung pro Sweep (bzw. pro Backup)
    backup_counts: np.ndarray | None = None  # wie oft wurde jedes Feld aktualisiert?
    snapshots: list[np.ndarray] = field(default_factory=list)  # V nach jedem Sweep (für Bilder)
    update_order: list[int] = field(default_factory=list)  # in welcher Reihenfolge wurde aktualisiert?
    policy: object | None = None  # bei Value Iteration: die daraus abgelesene Policy


class _Recorder:
    """Merkt sich nach jedem Update, wie weit wir noch von der richtigen Lösung ``v_ref`` weg sind."""

    def __init__(self, v_ref: np.ndarray | None):
        self.v_ref = v_ref
        self.backups: list[int] = []
        self.errors: list[float] = []

    def __call__(self, V: np.ndarray, backups: int) -> None:
        if self.v_ref is not None:
            self.backups.append(backups)
            # ∞-Norm: der größte Abstand über alle Felder
            self.errors.append(float(np.max(np.abs(V - self.v_ref))))


# =========================================================================================
# 1) SYNCHRON
# =========================================================================================
def run_synchronous(states, backup_all: BackupAll, V0, theta, max_sweeps, v_ref=None, keep_snapshots=False):
    start = time.perf_counter()
    V = V0.copy()  # die "alte Tafel"
    counts = np.zeros(len(V), dtype=int)
    rec = _Recorder(v_ref)
    rec(V, 0)
    snapshots = [V.copy()] if keep_snapshots else []
    deltas, backups, converged, sweep = [], 0, False, 0

    for sweep in range(1, max_sweeps + 1):
        V_new = V.copy()  # die "neue Tafel"
        V_new[states] = backup_all(V)[states]  # alle Felder rechnen nur mit den ALTEN Werten
        delta = float(np.max(np.abs(V_new - V)))  # größte Änderung in diesem Sweep
        V = V_new  # neue Tafel wird zur alten
        backups += len(states)
        counts[states] += 1
        deltas.append(delta)
        rec(V, backups)
        if keep_snapshots:
            snapshots.append(V.copy())
        if delta < theta:  # ändert sich (fast) nichts mehr -> fertig
            converged = True
            break

    return RunResult("synchronous", V, sweep, backups, backups, converged, time.perf_counter() - start,
                     rec.backups, rec.errors, deltas, counts, snapshots)


# =========================================================================================
# 2) IN-PLACE
# =========================================================================================
def run_in_place(states, backup_one: BackupOne, V0, theta, max_sweeps, v_ref=None, keep_snapshots=False):
    start = time.perf_counter()
    V = V0.copy()  # nur EINE Tafel
    counts = np.zeros(len(V), dtype=int)
    rec = _Recorder(v_ref)
    rec(V, 0)
    snapshots = [V.copy()] if keep_snapshots else []
    deltas, order, backups, converged, sweep = [], [], 0, False, 0

    for sweep in range(1, max_sweeps + 1):
        delta = 0.0
        for s in states:  # Felder in fester Reihenfolge durchgehen
            new = backup_one(V, s)  # benutzt schon die NEUEN Werte der vorher besuchten Felder
            delta = max(delta, abs(new - V[s]))
            V[s] = new  # sofort überschreiben
            backups += 1
            counts[s] += 1
            order.append(int(s))
            rec(V, backups)
        deltas.append(delta)
        if keep_snapshots:
            snapshots.append(V.copy())
        if delta < theta:
            converged = True
            break

    return RunResult("in_place", V, sweep, backups, backups, converged, time.perf_counter() - start,
                     rec.backups, rec.errors, deltas, counts, snapshots, order)


# =========================================================================================
# 3) PRIORITISED SWEEPING
# =========================================================================================
def run_prioritized(states, backup_one: BackupOne, predecessors, V0, theta, max_backups, v_ref=None,
                    keep_snapshots=False):
    start = time.perf_counter()
    V = V0.copy()
    n = len(states)
    updatable = np.zeros(len(V), dtype=bool)  # nur nicht-terminale Felder werden aktualisiert
    updatable[states] = True
    counts = np.zeros(len(V), dtype=int)
    priority = np.zeros(len(V))  # aktuelle Dringlichkeit (= Bellman-Fehler) jedes Feldes
    rec = _Recorder(v_ref)
    rec(V, 0)
    snapshots = [V.copy()] if keep_snapshots else []
    deltas, order, backups, error_evals = [], [], 0, 0
    heap: list[tuple[float, int]] = []  # Prioritätswarteschlange: (-Fehler, Feld)

    def bellman_error(s: int) -> float:
        """Wie weit liegt V(s) daneben? |neuer Wert laut Bellman - aktueller Wert|"""
        nonlocal error_evals
        error_evals += 1
        return abs(backup_one(V, s) - V[s])

    def push(s: int, err: float) -> None:
        """Feld s mit Dringlichkeit err in die Warteschlange legen."""
        priority[s] = err
        heapq.heappush(heap, (-err, s))  # heapq liefert das KLEINSTE zuerst, darum das Minus

    # Schritt 1: Fehler für alle Felder einmal berechnen und dringende Felder einreihen.
    for s in states:
        err = bellman_error(s)
        if err > theta:
            push(s, err)

    # Schritt 2: Immer das dringendste Feld aktualisieren.
    while heap and backups < max_backups:
        neg_err, s = heapq.heappop(heap)
        if -neg_err != priority[s]:
            continue  # veralteter Eintrag: Die Priorität wurde inzwischen geändert, also überspringen
        priority[s] = 0.0
        new = backup_one(V, s)
        deltas.append(abs(new - V[s]))
        V[s] = new
        backups += 1
        counts[s] += 1
        order.append(int(s))
        rec(V, backups)
        if keep_snapshots and backups % n == 0:
            snapshots.append(V.copy())

        # Schritt 3: V(s) hat sich geändert. Nur die VORGÄNGER von s benutzen V(s) in ihrer
        # Formel, also müssen nur sie neu bewertet werden. Das spart viel Arbeit.
        for p in predecessors[s]:
            if not updatable[p]:
                continue
            err = bellman_error(p)
            if err > theta:
                push(p, err)
            else:
                priority[p] = 0.0  # nicht mehr dringend: alte Einträge werden so ungültig

    converged = not np.any(priority[states] > 0)  # fertig, wenn kein Feld mehr dringend ist
    if keep_snapshots:
        snapshots.append(V.copy())
    return RunResult("prioritized", V, backups / n, backups, backups + error_evals, converged,
                     time.perf_counter() - start, rec.backups, rec.errors, deltas, counts, snapshots, order)


def run(method, states, backup_all, backup_one, predecessors, V0, theta, max_sweeps, v_ref, keep_snapshots):
    """Das passende Verfahren anhand seines Namens starten."""
    if method == "synchronous":
        return run_synchronous(states, backup_all, V0, theta, max_sweeps, v_ref, keep_snapshots)
    if method == "in_place":
        return run_in_place(states, backup_one, V0, theta, max_sweeps, v_ref, keep_snapshots)
    if method == "prioritized":
        return run_prioritized(states, backup_one, predecessors, V0, theta, max_sweeps * len(states), v_ref,
                               keep_snapshots)
    raise ValueError(f"Unbekannte Methode {method!r}")
