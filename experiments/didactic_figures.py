"""Lern-Grafiken: erklären die Ideen Schritt für Schritt (für docs/LERNPFAD.md).

Aufruf:  python -m experiments.didactic_figures   (wird auch von run_all aufgerufen)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Bilder nur in Dateien speichern, kein Fenster öffnen
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from rldp import ACTION_ARROWS, Policy, exact_policy_evaluation, greedy_policy, value_iteration  # noqa: E402
from rldp.maps import corridor, maze  # noqa: E402
from rldp.plotting import plot_values  # noqa: E402
from rldp.sweeps import run_in_place, run_prioritized, run_synchronous  # noqa: E402
from rldp.value_iteration import optimality_backups  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
NAMES_DE = ("oben", "rechts", "unten", "links")


def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  ->", FIG / name)


# -----------------------------------------------------------------------------------------
# 00  Überblick: Wie hängen die Verfahren zusammen?
# -----------------------------------------------------------------------------------------
def overview():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, text, color, w=2.0, h=0.9, fs=10):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="black", lw=1.2,
                               zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, weight="bold")

    def arrow(p, q, text="", rad=0.0, color="black", dy=0.0, fs=9):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=1.8, color=color,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))
        if text:
            ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center", va="center", fontsize=fs,
                    color=color)

    ax.text(6.5, 5.9, "Dynamic Programming: Das Modell der Welt (P, R, γ) ist bekannt", ha="center",
            fontsize=14, weight="bold")

    # --- links: Policy Iteration -------------------------------------------------------
    ax.text(3.0, 5.15, "Policy Iteration", ha="center", fontsize=13, weight="bold", color="#1f4e9c")
    box(1.2, 3.4, "Policy π\n(Strategie)", "#cfe0ff")
    box(4.8, 3.4, "Werte V^π\n(wie gut?)", "#ffe3b3")
    arrow((2.2, 3.85), (3.8, 3.85), "1. Policy Evaluation\nMITTELN über π(a|s)", rad=-0.45, dy=0.95,
          color="#1f4e9c")
    arrow((3.8, 2.95), (2.2, 2.95), "2. Policy Improvement\nGREEDY: beste Aktion", rad=-0.45, dy=-0.95,
          color="#b35900")
    ax.text(3.0, 1.15, "wiederholen, bis sich π nicht mehr ändert  →  π* und V*\n"
                       "wenige Runden, aber jede Evaluation ist teuer", ha="center", fontsize=9.5)

    # --- rechts: Value Iteration -------------------------------------------------------
    ax.text(10.0, 5.15, "Value Iteration", ha="center", fontsize=13, weight="bold", color="#2e7d32")
    xs = [7.4, 8.9, 10.4, 11.9]
    for x, t in zip(xs, ["V₀ = 0", "V₁", "V₂", "… V*"]):
        box(x, 3.9, t, "#d9f2d9", w=1.0, h=0.7)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + 0.5, 3.9), (b - 0.5, 3.9), "max", dy=0.3, color="#2e7d32")
    box(11.9, 2.4, "π*", "#cfe0ff", w=1.0, h=0.7)
    arrow((11.9, 3.55), (11.9, 2.75), "", color="#b35900")
    ax.text(11.25, 3.15, "einmal greedy\nablesen", ha="right", fontsize=8.5, color="#b35900")
    ax.text(9.0, 2.35, "jede Runde in jedem Feld\ndie BESTE Aktion nehmen\n(Bellman-Optimalität)\n"
                       "= Policy Iteration mit nur\n1 Evaluations-Sweep", ha="center", fontsize=9.5)

    # --- unten: Wie rechnet man einen Durchlauf? ---------------------------------------
    ax.text(6.5, 0.55, "Wie wird gerechnet?   ", ha="right", fontsize=10, weight="bold")
    for x, t in zip([7.6, 9.6, 11.9], ["Synchron\n(2 Tafeln)", "In-Place\n(1 Tafel)", "Prioritised\nSweeping"]):
        box(x, 0.55, t, "#eeeeee", w=1.8, h=0.75, fs=8.5)
    save(fig, "00_ueberblick.png")


# -----------------------------------------------------------------------------------------
# 10  Was ist eine Policy? Verschiedene Policies -> verschiedene Werte
# -----------------------------------------------------------------------------------------
def what_is_a_policy():
    env = maze(slip=0.0)
    n, A = env.n_states, env.n_actions
    policies = [
        ("Zufalls-Policy: jede Richtung 25 %", Policy.uniform(n, A)),
        ("Immer nach rechts →", Policy.deterministic(np.full(n, 1), A)),
        ("Optimale Policy π*", value_iteration(env, theta=1e-10).policy),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (title, pol) in zip(axes, policies):
        V = exact_policy_evaluation(env, pol)  # wie gut ist diese Policy in jedem Feld?
        start = env.index[(0, 0)]
        plot_values(ax, env, V, pol, f"{title}\nWert am Start S: {V[start]:.1f}", vmin=-20, vmax=10)
    fig.suptitle("Eine Policy sagt in jedem Feld, was zu tun ist (Pfeile). "
                 "Policy Evaluation berechnet daraus, wie gut jedes Feld ist (Farbe/Zahl).", fontsize=11)
    save(fig, "10_was_ist_eine_policy.png")


# -----------------------------------------------------------------------------------------
# 11  Ein einziges Backup, Schritt für Schritt vorgerechnet
# -----------------------------------------------------------------------------------------
def one_backup_explained():
    env = maze(slip=0.0)  # deterministisch, damit die Rechnung einfach bleibt
    backup_all, _ = optimality_backups(env)
    run = run_synchronous(env.nonterminal_states, backup_all, np.zeros(env.n_states), 1e-9, 200,
                          keep_snapshots=True)
    k = 4
    V_k, V_next = run.snapshots[k], run.snapshots[k + 1]
    # Feld wählen, das sich in diesem Sweep am stärksten ändert (dort, wo die "Welle" gerade ankommt)
    s = int(np.argmax(np.abs(V_next - V_k)))
    r, c = env.states[s]
    g = env.gamma

    fig = plt.figure(figsize=(16, 5.6))
    ax_map = fig.add_axes([0.0, 0.05, 0.42, 0.85])
    ax_txt = fig.add_axes([0.43, 0.0, 0.57, 1.0])
    ax_txt.axis("off")
    plot_values(ax_map, env, V_k, title=f"Werte V_k nach k = {k} Sweeps (Value Iteration)")
    ax_map.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="blue", lw=3.5))

    lines = [f"Wir aktualisieren das blaue Feld (Zeile {r}, Spalte {c}).", "",
             "Für jede Aktion: Belohnung + γ · Wert des Feldes, auf dem man landet", ""]
    q = []
    for a in range(env.n_actions):
        s_next = int(np.argmax(env.P[s, a]))  # deterministisch: genau ein Folgefeld
        q_a = env.R[s, a] + g * V_k[s_next]
        q.append(q_a)
        note = "  (Wand → stehen bleiben)" if s_next == s else ""
        lines.append(f"  {ACTION_ARROWS[a]} {NAMES_DE[a]:<7}: {env.R[s, a]:+.1f} + {g} · ({V_k[s_next]:+.2f})"
                     f" = {q_a:+.2f}{note}")
    q = np.array(q)
    best = int(np.argmax(q))
    lines += ["",
              "Policy Evaluation (Zufalls-Policy, je 25 %) MITTELT:",
              f"  V_k+1 = 0.25·({q[0]:+.2f}) + 0.25·({q[1]:+.2f}) + 0.25·({q[2]:+.2f}) + 0.25·({q[3]:+.2f})"
              f" = {q.mean():+.2f}",
              "",
              "Value Iteration nimmt das MAXIMUM:",
              f"  V_k+1 = max(...) = {q.max():+.2f}   (beste Aktion: {ACTION_ARROWS[best]} {NAMES_DE[best]})",
              "",
              f"Alter Wert: {V_k[s]:+.2f}   →   neuer Wert: {V_next[s]:+.2f}",
              "",
              "Das ist ein 'Backup'. Ein 'Sweep' macht das für jedes Feld einmal."]
    ax_txt.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10.5)
    save(fig, "11_ein_backup_erklaert.png")


# -----------------------------------------------------------------------------------------
# 12  Flur: synchron vs. in-place (und warum die Reihenfolge wichtig ist)
# -----------------------------------------------------------------------------------------
def corridor_sync_vs_inplace():
    env = corridor(length=8, gamma=0.9)
    states = env.nonterminal_states  # Felder 0..6 von links nach rechts
    backup_all, backup_one = optimality_backups(env)
    V0 = np.zeros(env.n_states)
    runs = [
        ("Synchron\n(liest nur alte Werte)", run_synchronous(states, backup_all, V0, 1e-9, 50, keep_snapshots=True)),
        ("In-Place, Reihenfolge links → rechts\n(weg vom Ziel: kein Vorteil)",
         run_in_place(states, backup_one, V0, 1e-9, 50, keep_snapshots=True)),
        ("In-Place, Reihenfolge rechts → links\n(vom Ziel aus: fertig in 1 Sweep!)",
         run_in_place(states[::-1], backup_one, V0, 1e-9, 50, keep_snapshots=True)),
    ]
    n_rows = 9  # Sweeps 0..8 zeigen
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (title, res) in zip(axes, runs):
        snaps = res.snapshots[:n_rows]
        snaps += [snaps[-1]] * (n_rows - len(snaps))  # nach Konvergenz bleibt alles gleich
        M = np.array(snaps)[:, :-1]  # letzte Spalte = Ziel G weglassen
        ax.imshow(M, cmap="RdYlGn", vmin=-5, vmax=9, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_yticks(range(n_rows), [f"k = {i}" for i in range(n_rows)])
        ax.set_xticks(range(M.shape[1]), [f"Feld {j}" for j in range(M.shape[1])], fontsize=7)
        ax.set_title(f"{title}\nkonvergiert nach {res.sweeps} Sweeps", fontsize=10)
        ax.text(M.shape[1] - 0.5, -0.9, "G →", fontsize=11, weight="bold", color="#1a9850", ha="left")
    fig.suptitle("Value Iteration im Flur (Ziel G rechts, Schritt −1, Ziel +10, γ = 0.9). "
                 "Jede Zeile = Werte nach einem Sweep", fontsize=12)
    fig.tight_layout()
    save(fig, "12_flur_synchron_vs_inplace.png")


# -----------------------------------------------------------------------------------------
# 13  Prioritised Sweeping: In welcher Reihenfolge wird aktualisiert?
# -----------------------------------------------------------------------------------------
def prioritized_order():
    env = maze(slip=0.0)
    states = env.nonterminal_states
    backup_all, backup_one = optimality_backups(env)
    V0 = np.zeros(env.n_states)
    N = 30  # nur die ersten N Backups anschauen

    # In-Place: einfach die Felder in Lese-Reihenfolge (Zeile für Zeile) N-mal aktualisieren
    v_inplace = V0.copy()
    order_inplace = list(states[:N])
    for s in order_inplace:
        v_inplace[s] = backup_one(v_inplace, s)

    # Prioritised Sweeping: N Backups, immer das Feld mit dem größten Bellman-Fehler zuerst
    ps = run_prioritized(states, backup_one, env.predecessors, V0, 1e-9, N)

    def order_grid(order):
        """Für jedes Feld: bei welchem Backup wurde es zum ersten Mal aktualisiert? (sonst NaN)"""
        first = np.full(env.n_states, np.nan)
        for step, s in enumerate(order, start=1):
            if np.isnan(first[s]):
                first[s] = step
        return first

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (name, order, V) in enumerate([("In-Place (Zeile für Zeile)", order_inplace, v_inplace),
                                             ("Prioritised Sweeping", ps.update_order, ps.V)]):
        first = order_grid(order)
        plot_values(axes[0, col], env, np.nan_to_num(first, nan=0), fmt="{:.0f}", cmap="Blues", vmin=0,
                    vmax=N * 1.3, title=f"{name}: Nummer des ersten Updates (0 = noch nie)")
        plot_values(axes[1, col], env, V, greedy_policy(env, V), vmin=-2, vmax=9,
                    title=f"{name}: Werte nach nur {N} Backups")
    fig.suptitle("Prioritised Sweeping beginnt dort, wo sich etwas ändert (am Ziel), "
                 "und arbeitet sich rückwärts durch das Labyrinth.\n"
                 "In-Place beginnt stur oben links, wo es noch nichts zu lernen gibt.", fontsize=11)
    save(fig, "13_prioritised_sweeping_reihenfolge.png")


# -----------------------------------------------------------------------------------------
# 14  Kontraktion: Egal wo man startet, man landet bei V*
# -----------------------------------------------------------------------------------------
def contraction():
    env = maze(slip=0.1)
    states = env.nonterminal_states
    backup_all, _ = optimality_backups(env)
    rng = np.random.default_rng(0)
    U0 = np.zeros(env.n_states)
    U0[states] = rng.uniform(-30, 30, len(states))  # wilder, zufälliger Startpunkt
    a = run_synchronous(states, backup_all, np.zeros(env.n_states), 0, 60, keep_snapshots=True)
    b = run_synchronous(states, backup_all, U0, 0, 60, keep_snapshots=True)

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.4])
    for row, (name, res) in enumerate([("Start A: alles 0", a), ("Start B: Zufallswerte", b)]):
        for col, k in enumerate([0, 5, 60]):
            ax = fig.add_subplot(gs[row, col])
            plot_values(ax, env, res.snapshots[k], title=f"{name}, k = {k}", vmin=-30, vmax=30,
                        annotate=False)
    ax = fig.add_subplot(gs[:, 3])
    k = np.arange(61)
    dist = [np.max(np.abs(x - y)) for x, y in zip(a.snapshots, b.snapshots)]
    ax.semilogy(k, dist, lw=2, label="Abstand ||V_k(A) − V_k(B)||∞")
    ax.semilogy(k, dist[0] * env.gamma ** k, "k:", label="Garantie: γ^k · Abstand am Anfang")
    ax.set(xlabel="Sweep k", ylabel="größter Unterschied zwischen A und B",
           title="Jeder Sweep zieht die Werte mindestens um den Faktor γ\nzusammen (γ-Kontraktion)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("Warum Value Iteration immer funktioniert: Zwei völlig verschiedene Starts landen "
                 "beim selben V* (eindeutiger Fixpunkt)", fontsize=12)
    fig.tight_layout()
    save(fig, "14_kontraktion.png")


def make_all():
    FIG.mkdir(exist_ok=True)
    print("Lern-Grafiken …")
    overview()
    what_is_a_policy()
    one_backup_explained()
    corridor_sync_vs_inplace()
    prioritized_order()
    contraction()


if __name__ == "__main__":
    make_all()
