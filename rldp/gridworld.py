"""Gitterwelt als vollständig bekannter MDP (Modell P, R) für Dynamic Programming."""

from __future__ import annotations

import numpy as np

ACTIONS = ("up", "right", "down", "left")
ACTION_ARROWS = ("↑", "→", "↓", "←")
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

WALL = "#"
TERMINAL_CELLS = {"G", "X", "T"}


class GridWorld:
    """Tabellarischer MDP, gebaut aus einer ASCII-Karte.

    Zeichen der Karte:
      ``.`` freies Feld, ``#`` Wand, ``S`` Start (freies Feld),
      ``G`` Ziel (terminal, Bonus ``goal_reward``), ``X`` Falle (terminal, ``trap_reward``),
      ``T`` terminal ohne Bonus. Andere Buchstaben sind freie Felder (z. B. Teleporter).

    Das Modell liegt explizit vor (Voraussetzung für DP):
      ``P[s, a, s']`` Übergangswahrscheinlichkeiten, ``R[s, a]`` erwartete Belohnung.

    ``slip`` > 0 macht die Welt stochastisch: mit Wahrscheinlichkeit ``slip`` rutscht der
    Agent je zur Hälfte nach links/rechts (relativ zur gewählten Richtung).
    ``teleports`` bildet z. B. ``{"A": ("a", 10.0)}`` ab: von Feld A springt jede Aktion
    nach Feld a und bringt Belohnung 10 (Beispiel aus Sutton/Barto, Abb. 3.2).
    """

    def __init__(
        self,
        grid: list[str],
        *,
        gamma: float = 0.9,
        step_reward: float = -1.0,
        bump_reward: float | None = None,
        goal_reward: float = 0.0,
        trap_reward: float = -10.0,
        slip: float = 0.0,
        teleports: dict[str, tuple[str, float]] | None = None,
        name: str = "GridWorld",
    ):
        rows = [str(r) for r in grid]
        if not rows or len({len(r) for r in rows}) != 1:
            raise ValueError("Alle Zeilen der Karte müssen gleich lang sein")
        if not 0.0 <= slip < 1.0:
            raise ValueError("slip muss in [0, 1) liegen")
        self.grid = rows
        self.shape = (len(rows), len(rows[0]))
        self.name = name
        self.gamma = gamma
        self.step_reward = step_reward
        self.bump_reward = step_reward if bump_reward is None else bump_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.slip = slip

        self.states = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != WALL]
        self.index = {pos: i for i, pos in enumerate(self.states)}
        self.n_states = len(self.states)
        self.n_actions = len(ACTIONS)
        self.terminal = np.array([rows[r][c] in TERMINAL_CELLS for r, c in self.states])
        self.nonterminal_states = np.flatnonzero(~self.terminal)

        self._teleports = {
            self._find(src): (self._find(dst), float(rew)) for src, (dst, rew) in (teleports or {}).items()
        }
        self.P, self.R = self._build_model()
        # Vorgänger von s: alle Zustände, die mit irgendeiner Aktion nach s gelangen können
        self.predecessors = [np.flatnonzero(self.P[:, :, s].any(axis=1)) for s in range(self.n_states)]

    def cell(self, s: int) -> str:
        r, c = self.states[s]
        return self.grid[r][c]

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """Vektor über Zustände -> 2D-Array der Kartengröße (Wände = NaN)."""
        out = np.full(self.shape, np.nan)
        for s, (r, c) in enumerate(self.states):
            out[r, c] = values[s]
        return out

    def step(self, s: int, a: int, rng: np.random.Generator | None = None) -> tuple[int, float]:
        """Einen Schritt simulieren (wie ``env.step`` in den Folien)."""
        if self.terminal[s]:
            return s, 0.0
        rng = rng or np.random.default_rng()
        outcomes = self._action_outcomes(a)
        actual = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
        nxt, reward = self._outcome(self.states[s], actual)
        return self.index[nxt], reward

    def _find(self, ch: str) -> tuple[int, int]:
        hits = [(r, c) for r, row in enumerate(self.grid) for c, x in enumerate(row) if x == ch]
        if len(hits) != 1:
            raise ValueError(f"Zeichen {ch!r} muss genau einmal in der Karte vorkommen")
        return hits[0]

    def _move(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        dr, dc = _DELTAS[action]
        r, c = pos[0] + dr, pos[1] + dc
        if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]) or self.grid[r][c] == WALL:
            return pos, True
        return (r, c), False

    def _outcome(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], float]:
        if pos in self._teleports:
            return self._teleports[pos]
        nxt, bumped = self._move(pos, action)
        if bumped:
            return pos, self.bump_reward
        reward = self.step_reward
        ch = self.grid[nxt[0]][nxt[1]]
        if ch == "G":
            reward += self.goal_reward
        elif ch == "X":
            reward += self.trap_reward
        return nxt, reward

    def _action_outcomes(self, action: int) -> list[tuple[int, float]]:
        if self.slip == 0:
            return [(action, 1.0)]
        half = self.slip / 2
        return [(action, 1.0 - self.slip), ((action + 1) % 4, half), ((action - 1) % 4, half)]

    def _build_model(self) -> tuple[np.ndarray, np.ndarray]:
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for s, pos in enumerate(self.states):
            if self.terminal[s]:
                P[s, :, s] = 1.0  # absorbierend, Belohnung 0
                continue
            for a in range(self.n_actions):
                for actual, prob in self._action_outcomes(a):
                    nxt, reward = self._outcome(pos, actual)
                    P[s, a, self.index[nxt]] += prob
                    R[s, a] += prob * reward
        return P, R
