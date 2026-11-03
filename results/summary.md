# Ergebnisse

Automatisch erzeugt von `python -m experiments.run_all`.

*Backup* = ein Zustand wird aktualisiert. *Bellman-Auswertung* = jede Berechnung der
rechten Seite der Bellman-Gleichung (bei Prioritised Sweeping auch das Berechnen der
Bellman-Fehler, bei Policy Iteration auch der Improvement-Schritt).

## Policy Evaluation der Random Policy (Labyrinth (slip=0.1, γ=0.95), θ = 1e-06)

| Verfahren | Sweeps | Backups | Bellman-Auswertungen | Zeit [ms] | Fehler zu exakt |
|---|---|---|---|---|---|
| Synchron | 254.0 | 12700 | 12700 | 1.6 | 1.7e-05 |
| In-Place | 196.0 | 9800 | 9800 | 30.4 | 1.3e-05 |
| Prioritised Sweeping | 165.8 | 8290 | 33395 | 81.7 | 1.6e-05 |

## Policy Iteration Schritt für Schritt (Labyrinth (slip=0.1, γ=0.95))

| Iteration | Eval-Sweeps | Eval-Backups | geänderte Zustände |
|---|---|---|---|
| 1 | 196 | 9800 | 50 |
| 2 | 24 | 1200 | 3 |
| 3 | 19 | 950 | 2 |
| 4 | 7 | 350 | 1 |
| 5 | 17 | 850 | 2 |
| 6 | 21 | 1050 | 1 |
| 7 | 22 | 1100 | 1 |
| 8 | 21 | 1050 | 0 |

## Policy Iteration vs. Value Iteration (Labyrinth (slip=0.1, γ=0.95), θ = 1e-06)

| Verfahren | Iterationen | Backups (Eval.) | Bellman-Auswertungen gesamt | Zeit [ms] | Fehler zu V* | π optimal? |
|---|---|---|---|---|---|---|
| Policy Iteration (volle Eval.) | 8 | 20500 | 20900 | 3.1 | 4.4e-07 | ja |
| Modified PI (k = 10) | 9 | 4200 | 4650 | 1.0 | 1.3e-07 | ja |
| Modified PI (k = 3) | 17 | 2500 | 3350 | 1.2 | 2.5e-07 | ja |
| Modified PI (k = 1) | 45 | 2250 | 4500 | 2.6 | 3.1e-07 | ja |
| Value Iteration (Synchron) | 44.0 Sweeps | 2200 | 2200 | 0.5 | 3.4e-07 | ja |
| Value Iteration (In-Place) | 34.0 Sweeps | 1700 | 1700 | 6.6 | 3.0e-07 | ja |
| Value Iteration (Prioritised Sweeping) | 28.6 Sweeps | 1429 | 5777 | 18.1 | 3.7e-06 | ja |

## Sweeps bis Konvergenz in Abhängigkeit von γ (Labyrinth, θ = 1e-06)

| Verfahren | γ=0.5 | γ=0.7 | γ=0.8 | γ=0.9 | γ=0.95 | γ=0.98 | γ=0.99 |
|---|---|---|---|---|---|---|---|
| VI Synchron | 21.0 | 40.0 | 60.0 | 41.0 | 44.0 | 45.0 | 46.0 |
| PE (Random) Synchron | 21.0 | 40.0 | 63.0 | 130.0 | 254.0 | 574.0 | 977.0 |
| VI In-Place | 21.0 | 38.0 | 46.0 | 33.0 | 34.0 | 35.0 | 36.0 |
| PE (Random) In-Place | 19.0 | 33.0 | 51.0 | 102.0 | 196.0 | 440.0 | 747.0 |
| VI Prioritised Sweeping | 13.7 | 21.5 | 25.4 | 22.6 | 28.6 | 34.6 | 36.1 |
| PE (Random) Prioritised Sweeping | 15.9 | 28.7 | 44.0 | 88.1 | 165.8 | 350.7 | 567.8 |
