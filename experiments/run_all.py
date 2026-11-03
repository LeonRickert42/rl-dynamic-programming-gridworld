"""Erzeugt alle Abbildungen (figures/) und Vergleichstabellen (results/summary.md).

Aufruf aus dem Repo-Wurzelverzeichnis:  python -m experiments.run_all
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rldp import (  # noqa: E402
    Policy,
    exact_policy_evaluation,
    greedy_policy,
    policy_evaluation,
    policy_iteration,
    value_iteration,
)
from rldp.maps import maze, small_gridworld, sutton_5x5  # noqa: E402
from rldp.plotting import plot_error_curves, plot_values, save_value_animation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
NAMES = {"synchronous": "Synchron", "in_place": "In-Place", "prioritized": "Prioritised Sweeping"}
THETA = 1e-6
lines: list[str] = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print("  ->", FIG / name)


def table(header, rows):
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    lines.append("")


def uniform(env):
    return Policy.uniform(env.n_states, env.n_actions)


def small_gridworld_like_slides():
    """Abb. 1: Random-Policy-Evaluation im 4x4-Gitter + greedy Policy (wie Silver/Sutton)."""
    env = small_gridworld()
    run = policy_evaluation(env, uniform(env), "synchronous", theta=1e-8, keep_snapshots=True)
    ks = [0, 1, 2, 3, 10]
    snaps = [(f"k = {k}", run.snapshots[k]) for k in ks] + [("k = ∞", run.V)]
    fig, axes = plt.subplots(2, len(snaps), figsize=(2.2 * len(snaps), 4.6))
    for col, (label, V) in enumerate(snaps):
        plot_values(axes[0, col], env, V, title=f"V_k Random Policy, {label}", vmin=-22, vmax=0, fontsize=8)
        greedy = greedy_policy(env, V, tol=1e-6, ties="uniform")
        plot_values(axes[1, col], env, V, greedy, title=f"greedy Policy bzgl. V, {label}", annotate=False,
                    vmin=-22, vmax=0)
    save(fig, "01_small_gridworld_policy_evaluation.png")


def sutton_5x5_like_pareigis():
    """Abb. 2: 5x5-Gitter mit A/B-Teleportern: V der Random Policy vs. V* und π*."""
    env = sutton_5x5()
    v_rand = exact_policy_evaluation(env, uniform(env))
    vi = value_iteration(env, theta=1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_values(axes[0], env, v_rand, title="V^π der Random Policy", fmt="{:.2f}", fontsize=8)
    plot_values(axes[1], env, vi.V, greedy_policy(env, vi.V, 1e-6, "uniform"), title="V* und π* (Value Iteration)",
                fontsize=8)
    save(fig, "02_sutton_5x5.png")


def compare_policy_evaluation(env):
    """Abb. 3+4: Synchron vs. In-Place vs. Prioritised Sweeping bei der Policy Evaluation."""
    policy = uniform(env)
    v_ref = exact_policy_evaluation(env, policy)
    runs = {m: policy_evaluation(env, policy, m, THETA, v_ref=v_ref) for m in NAMES}
    n = len(env.nonterminal_states)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_error_curves(axes[0], {NAMES[m]: (r.backups, r.errors) for m, r in runs.items()}, n,
                      f"Policy Evaluation (Random Policy) – {env.name}")
    for m in ("synchronous", "in_place"):
        axes[1].semilogy(range(1, len(runs[m].deltas) + 1), runs[m].deltas, label=NAMES[m])
    axes[1].axhline(THETA, color="gray", ls="--", label="θ (Abbruch)")
    axes[1].set(xlabel="Sweep k", ylabel="max |V_{k+1} − V_k|", title="Größte Änderung pro Sweep")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    save(fig, "03_policy_evaluation_convergence.png")

    lines.append(f"## Policy Evaluation der Random Policy ({env.name}, θ = {THETA})\n")
    table(["Verfahren", "Sweeps", "Backups", "Bellman-Auswertungen", "Zeit [ms]", "Fehler zu exakt"],
          [[NAMES[m], f"{r.sweeps:.1f}", r.total_backups, r.evaluations, f"{1000 * r.elapsed:.1f}",
            f"{np.max(np.abs(r.V - v_ref)):.1e}"] for m, r in runs.items()])
    return runs, v_ref


def compare_value_iteration(env, v_star):
    """Abb. 6+7: Value Iteration – Wellenausbreitung und Konvergenz der drei Verfahren."""
    runs = {m: value_iteration(env, m, THETA, v_ref=v_star, keep_snapshots=True) for m in NAMES}
    n = len(env.nonterminal_states)
    sync = runs["synchronous"]
    vmin, vmax = v_star[env.nonterminal_states].min(), v_star.max()

    ks = [0, 1, 2, 3, 5, 8, 12, len(sync.snapshots) - 1]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
    for ax, k in zip(axes.flat, ks):
        plot_values(ax, env, sync.snapshots[k], greedy_policy(env, sync.snapshots[k]),
                    title=f"Value Iteration (synchron), k = {k}", vmin=vmin, vmax=vmax)
    save(fig, "06_value_iteration_sweeps.png")

    save_value_animation(env, sync.snapshots[: min(len(sync.snapshots), 30)], FIG / "value_iteration.gif",
                         policy_fn=lambda V: greedy_policy(env, V))
    print("  ->", FIG / "value_iteration.gif")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    curves = {NAMES[m]: (r.backups, r.errors) for m, r in runs.items()}
    plot_error_curves(axes[0], curves, n, f"Value Iteration – {env.name}", "max. Fehler ||V − V*||∞")
    k = np.arange(0, int(sync.sweeps) + 1)
    axes[0].semilogy(k, np.max(np.abs(v_star)) * env.gamma ** k, "k:", label="Schranke γ^k·||V*||∞")
    axes[0].legend(fontsize=8)
    for m in ("synchronous", "in_place"):
        axes[1].semilogy(range(1, len(runs[m].deltas) + 1), runs[m].deltas, label=NAMES[m])
    axes[1].axhline(THETA, color="gray", ls="--", label="θ (Abbruch)")
    axes[1].set(xlabel="Sweep k", ylabel="max |V_{k+1} − V_k|", title="Größte Änderung pro Sweep")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    save(fig, "07_value_iteration_convergence.png")
    return runs


def backup_count_heatmaps(env, pe_runs, vi_runs):
    """Abb. 4: Wie oft wurde welcher Zustand aktualisiert?"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    for row, (label, runs) in enumerate([("Policy Evaluation", pe_runs), ("Value Iteration", vi_runs)]):
        vmax = max(r.backup_counts.max() for r in runs.values())
        for col, (m, r) in enumerate(runs.items()):
            plot_values(axes[row, col], env, r.backup_counts, fmt="{:.0f}", cmap="viridis", vmin=0, vmax=vmax,
                        title=f"{label}: {NAMES[m]} (Σ = {r.total_backups})")
    save(fig, "04_backups_per_state.png")


def show_policy_iteration(env, v_star):
    """Abb. 5: Policy Iteration Schritt für Schritt (Evaluation -> Improvement)."""
    pi = policy_iteration(env, eval_method="in_place", theta=THETA)
    vmin, vmax = v_star[env.nonterminal_states].min(), v_star.max()
    steps = [dict(iteration=0, V=np.zeros(env.n_states), policy=uniform(env), changed="–", eval_sweeps=0)]
    steps += pi.history
    cols = min(4, len(steps))
    rows = int(np.ceil(len(steps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.6 * rows), squeeze=False)
    for ax, h in zip(axes.flat, steps):
        title = ("Start: Random Policy" if h["iteration"] == 0 else
                 f"Iteration {h['iteration']}: {h['eval_sweeps']} Eval-Sweeps,\n{h['changed']} Zustände geändert")
        plot_values(ax, env, h["V"], h["policy"], title=title, vmin=vmin, vmax=vmax)
    for ax in axes.flat[len(steps):]:
        ax.axis("off")
    save(fig, "05_policy_iteration_steps.png")

    lines.append(f"## Policy Iteration Schritt für Schritt ({env.name})\n")
    table(["Iteration", "Eval-Sweeps", "Eval-Backups", "geänderte Zustände"],
          [[h["iteration"], h["eval_sweeps"], h["eval_backups"], h["changed"]] for h in pi.history])


def compare_pi_vs_vi(env, v_star):
    """Abb. 8: Policy Iteration (voll/modifiziert) vs. Value Iteration – Aufwand bis zum Optimum."""
    configs = {
        "Policy Iteration (volle Eval.)": dict(eval_max_sweeps=None),
        "Modified PI (k = 10)": dict(eval_max_sweeps=10),
        "Modified PI (k = 3)": dict(eval_max_sweeps=3),
        "Modified PI (k = 1)": dict(eval_max_sweeps=1),
    }
    curves, rows, bars = {}, [], {}
    for name, cfg in configs.items():
        r = policy_iteration(env, eval_method="synchronous", theta=THETA, v_ref=v_star, **cfg)
        optimal = np.allclose(exact_policy_evaluation(env, r.policy), v_star, atol=1e-4)
        curves[name] = (r.backups, r.errors)
        bars[name] = r.evaluations
        rows.append([name, r.iterations, r.total_backups, r.evaluations, f"{1000 * r.elapsed:.1f}",
                     f"{np.max(np.abs(r.V - v_star)):.1e}", "ja" if optimal else "nein"])
    for m in NAMES:
        r = value_iteration(env, m, THETA, v_ref=v_star)
        optimal = np.allclose(exact_policy_evaluation(env, r.policy), v_star, atol=1e-4)
        name = f"Value Iteration ({NAMES[m]})"
        curves[name] = (r.backups, r.errors)
        bars[name] = r.evaluations
        rows.append([name, f"{r.sweeps:.1f} Sweeps", r.total_backups, r.evaluations, f"{1000 * r.elapsed:.1f}",
                     f"{np.max(np.abs(r.V - v_star)):.1e}", "ja" if optimal else "nein"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    plot_error_curves(axes[0], curves, len(env.nonterminal_states),
                      "Abstand zu V* – Policy Iteration vs. Value Iteration", "max. Fehler ||V − V*||∞")
    names = list(bars)
    axes[1].barh(names, [bars[n] for n in names], color=["C0"] * 4 + ["C1"] * 3)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Bellman-Auswertungen bis Konvergenz (inkl. Improvement)")
    axes[1].set_title("Gesamtaufwand", fontsize=10)
    axes[1].tick_params(axis="y", labelsize=8)
    save(fig, "08_policy_iteration_vs_value_iteration.png")

    lines.append(f"## Policy Iteration vs. Value Iteration ({env.name}, θ = {THETA})\n")
    table(["Verfahren", "Iterationen", "Backups (Eval.)", "Bellman-Auswertungen gesamt", "Zeit [ms]",
           "Fehler zu V*", "π optimal?"], rows)


def parameter_study():
    """Abb. 9: Einfluss von γ und θ auf die Anzahl der Sweeps bis zur Konvergenz."""
    gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    thetas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rows = []
    for m in NAMES:
        vi_g = [value_iteration(maze(gamma=g), m, THETA).sweeps for g in gammas]
        pe_g = [policy_evaluation(maze(gamma=g), uniform(maze(gamma=g)), m, THETA).sweeps for g in gammas]
        vi_t = [value_iteration(maze(), m, t).sweeps for t in thetas]
        axes[0].plot(gammas, vi_g, "o-", label=f"VI {NAMES[m]}")
        axes[1].plot(gammas, pe_g, "o-", label=f"PE {NAMES[m]}")
        axes[2].semilogx(thetas, vi_t, "o-", label=f"VI {NAMES[m]}")
        rows.append([f"VI {NAMES[m]}"] + [f"{x:.1f}" for x in vi_g])
        rows.append([f"PE (Random) {NAMES[m]}"] + [f"{x:.1f}" for x in pe_g])
    axes[0].set(xlabel="γ", ylabel="Sweeps bis Konvergenz", title="Value Iteration: Einfluss von γ")
    axes[1].set(xlabel="γ", ylabel="Sweeps bis Konvergenz", title="Policy Evaluation (Random): Einfluss von γ")
    axes[2].set(xlabel="θ", ylabel="Sweeps bis Konvergenz", title="Value Iteration: Einfluss von θ (γ = 0.95)")
    axes[2].invert_xaxis()
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    save(fig, "09_parameters_gamma_theta.png")
    lines.append(f"## Sweeps bis Konvergenz in Abhängigkeit von γ (Labyrinth, θ = {THETA})\n")
    table(["Verfahren"] + [f"γ={g}" for g in gammas], rows)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    lines.extend(["# Ergebnisse", "", "Automatisch erzeugt von `python -m experiments.run_all`.", "",
                  "*Backup* = ein Zustand wird aktualisiert. *Bellman-Auswertung* = jede Berechnung der",
                  "rechten Seite der Bellman-Gleichung (bei Prioritised Sweeping auch das Berechnen der",
                  "Bellman-Fehler, bei Policy Iteration auch der Improvement-Schritt).", ""])
    print("Folien-Beispiele …")
    small_gridworld_like_slides()
    sutton_5x5_like_pareigis()

    env = maze()
    v_star = policy_iteration(env, eval_method="exact").V
    print(f"{env.name}: {len(env.nonterminal_states)} nicht-terminale Zustände")
    pe_runs, _ = compare_policy_evaluation(env)
    show_policy_iteration(env, v_star)
    vi_runs = compare_value_iteration(env, v_star)
    backup_count_heatmaps(env, pe_runs, vi_runs)
    compare_pi_vs_vi(env, v_star)
    parameter_study()

    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("  ->", RES / "summary.md")


if __name__ == "__main__":
    main()
