"""Bilder: Heatmaps (Farbe = Wert eines Feldes), Policy-Pfeile und Konvergenzkurven."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Pfeilrichtung im Bild (dx, dy) für ↑ → ↓ ←. Im Bild zeigt y nach UNTEN, darum ist "oben" = -1.
_ARROW = ((0, -1), (1, 0), (0, 1), (-1, 0))


def plot_values(ax, env, V, policy=None, title=None, *, annotate=True, fmt="{:.1f}", cmap="RdYlGn",
                vmin=None, vmax=None, fontsize=None):
    """Zeichnet die Karte als Heatmap.

    * Farbe eines Feldes = sein Wert V(s) (grün = gut, rot = schlecht)
    * Wände: dunkelgrau, Ziel G: grün, Falle X: rot
    * optional: Pfeile für die Policy (dicker/dunkler = höhere Wahrscheinlichkeit)
    """
    grid = env.to_grid(np.asarray(V, dtype=float))
    for s in np.flatnonzero(env.terminal):  # terminale Felder bekommen eigene Farbe (unten)
        r, c = env.states[s]
        grid[r, c] = np.nan
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#3a3a3a")  # NaN (Wände) = dunkelgrau
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cm, vmin=vmin, vmax=vmax)
    fontsize = fontsize or max(5, 28 / max(env.shape))  # Schrift passend zur Kartengröße

    for s, (r, c) in enumerate(env.states):
        ch = env.grid[r][c]
        if env.terminal[s]:  # Ziel/Falle/Terminal als farbiges Feld mit Buchstabe
            color = {"G": "#1a9850", "X": "#d73027"}.get(ch, "#bbbbbb")
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=color))
            ax.text(c, r, ch, ha="center", va="center", fontsize=fontsize * 1.4, weight="bold", color="white")
            continue
        if ch not in ".":  # Sonderfelder (S, A, B, ...) klein oben links beschriften
            ax.text(c - 0.42, r - 0.42, ch, ha="left", va="top", fontsize=fontsize * 0.9, weight="bold")
        if annotate:  # Zahlenwert ins Feld schreiben
            y = r + 0.28 if policy is not None else r
            ax.text(c, y, fmt.format(V[s]), ha="center", va="center", fontsize=fontsize)
        if policy is not None:  # einen Pfeil pro Aktion mit Wahrscheinlichkeit > 0
            for a, p in enumerate(policy.probs[s]):
                if p > 1e-9:
                    dx, dy = _ARROW[a]
                    ax.arrow(c, r - 0.08, 0.3 * dx, 0.3 * dy, head_width=0.12, head_length=0.1,
                             length_includes_head=True, color="black", alpha=0.35 + 0.65 * p, lw=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


def plot_error_curves(ax, curves: dict, n_states: int, title: str, ylabel="max. Fehler ||V − V_ref||∞"):
    """Fehler über Rechenaufwand, logarithmisch.

    curves: Name -> (backups, errors). Die x-Achse ist "Backups / Anzahl Felder",
    damit 1 auf der x-Achse ungefähr einem kompletten Durchlauf (Sweep) entspricht.
    Eine GERADE Linie im log-Plot heißt: Der Fehler schrumpft jedes Mal um denselben Faktor.
    """
    for name, (backups, errors) in curves.items():
        errors = np.maximum(np.asarray(errors), 1e-12)  # log(0) vermeiden
        ax.semilogy(np.asarray(backups) / n_states, errors, label=name, lw=1.6)
    ax.set_xlabel("Backups / Anzahl Zustände  (≈ Sweeps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def save_value_animation(env, snapshots, path, policy_fn=None, title="Value Iteration", fps=3):
    """GIF-Animation: wie sich die Werte Sweep für Sweep über die Karte ausbreiten."""
    finite = np.concatenate(snapshots)
    vmin, vmax = finite.min(), finite.max()  # feste Farbskala über alle Bilder
    fig, ax = plt.subplots(figsize=(1 + 0.6 * env.shape[1], 1 + 0.6 * env.shape[0]))

    def draw(k):
        ax.clear()
        policy = policy_fn(snapshots[k]) if policy_fn else None
        plot_values(ax, env, snapshots[k], policy, f"{title} – Sweep k = {k}", vmin=vmin, vmax=vmax)

    anim = animation.FuncAnimation(fig, draw, frames=len(snapshots))
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
