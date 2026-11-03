"""Die Gitterwelt: unser Spielfeld.

Stell dir ein Brettspiel vor: Ein Agent steht auf einem Feld und kann
nach oben, rechts, unten oder links gehen. Für jeden Schritt gibt es eine
Belohnung (meist eine kleine Strafe, z. B. -1). Erreicht er das Ziel G, gibt es einen Bonus,
tritt er in die Falle X, eine Strafe.

Dynamic Programming braucht ein VOLLSTÄNDIG BEKANNTES Modell der Welt:
  P[s, a, s']  = Wahrscheinlichkeit, von Feld s mit Aktion a auf Feld s' zu landen
  R[s, a]      = Belohnung, die man im Schnitt für Aktion a in Feld s bekommt
Genau diese zwei Tabellen baut diese Datei aus einer Text-Karte.
"""

from __future__ import annotations

import numpy as np

# Die vier möglichen Aktionen. Ihre Nummer (0..3) ist ihr Index in allen Tabellen.
ACTIONS = ("up", "right", "down", "left")
ACTION_ARROWS = ("↑", "→", "↓", "←")

# Wie sich (Zeile, Spalte) bei jeder Aktion verändert: oben = eine Zeile weniger, usw.
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

WALL = "#"  # Wände sind keine Zustände, man kann sie nicht betreten
TERMINAL_CELLS = {"G", "X", "T"}  # hier endet eine Episode (Ziel, Falle, Terminal)


class GridWorld:
    """Tabellarischer MDP (Markov Decision Process), gebaut aus einer ASCII-Karte.

    Zeichen der Karte:
      ``.`` freies Feld          ``#`` Wand            ``S`` Start (sonst ein freies Feld)
      ``G`` Ziel (Ende, Bonus)   ``X`` Falle (Ende, Strafe)   ``T`` Ende ohne Bonus
      andere Buchstaben = freie Felder (z. B. für Teleporter)

    Beispiel::

        env = GridWorld(["S..#",
                         "...G"], gamma=0.9, step_reward=-1, goal_reward=10)

    ``slip`` > 0 macht den Boden rutschig: Mit Wahrscheinlichkeit ``slip`` geht der Agent
    nicht geradeaus, sondern zur Hälfte nach links bzw. rechts (von ihm aus gesehen).
    ``teleports`` z. B. ``{"A": ("a", 10.0)}``: Jede Aktion auf Feld A beamt nach Feld a
    und bringt +10 (Beispiel aus Sutton/Barto Abb. 3.2 und den Pareigis-Folien).
    """

    def __init__(
        self,
        grid: list[str],
        *,
        gamma: float = 0.9,  # Discount: wie viel ist eine Belohnung "später" noch wert?
        step_reward: float = -1.0,  # Belohnung für jeden normalen Schritt
        bump_reward: float | None = None,  # Belohnung, wenn man gegen Wand/Rand läuft
        goal_reward: float = 0.0,  # Extra-Bonus beim Betreten von G
        trap_reward: float = -10.0,  # Extra-Strafe beim Betreten von X
        slip: float = 0.0,  # Rutsch-Wahrscheinlichkeit (0 = deterministisch)
        teleports: dict[str, tuple[str, float]] | None = None,
        name: str = "GridWorld",
    ):
        # --- Eingaben prüfen -------------------------------------------------------------
        rows = [str(r) for r in grid]
        if not rows or len({len(r) for r in rows}) != 1:
            raise ValueError("Alle Zeilen der Karte müssen gleich lang sein")
        if not 0.0 <= slip < 1.0:
            raise ValueError("slip muss in [0, 1) liegen")

        # --- Parameter merken ------------------------------------------------------------
        self.grid = rows
        self.shape = (len(rows), len(rows[0]))  # (Anzahl Zeilen, Anzahl Spalten)
        self.name = name
        self.gamma = gamma
        self.step_reward = step_reward
        self.bump_reward = step_reward if bump_reward is None else bump_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.slip = slip

        # --- Zustände durchnummerieren ---------------------------------------------------
        # Jedes Feld, das keine Wand ist, wird ein Zustand mit einer Nummer s = 0, 1, 2, ...
        # states[s]  -> (Zeile, Spalte)      index[(Zeile, Spalte)] -> s
        self.states = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != WALL]
        self.index = {pos: i for i, pos in enumerate(self.states)}
        self.n_states = len(self.states)
        self.n_actions = len(ACTIONS)

        # terminal[s] ist True, wenn das Spiel auf Feld s zu Ende ist (G, X, T).
        self.terminal = np.array([rows[r][c] in TERMINAL_CELLS for r, c in self.states])
        # Nur diese Zustände müssen später aktualisiert werden. Terminale Felder haben immer Wert 0.
        self.nonterminal_states = np.flatnonzero(~self.terminal)

        # Teleporter: Buchstaben in Positionen übersetzen.
        self._teleports = {
            self._find(src): (self._find(dst), float(rew)) for src, (dst, rew) in (teleports or {}).items()
        }

        # --- Das Modell bauen: P und R ---------------------------------------------------
        self.P, self.R = self._build_model()

        # Vorgänger von s: alle Felder, von denen aus man (mit irgendeiner Aktion) nach s kommt.
        # Wird für Prioritised Sweeping gebraucht: Ändert sich V(s), müssen nur die Vorgänger
        # neu geprüft werden, denn nur sie "schauen" auf s.
        self.predecessors = [np.flatnonzero(self.P[:, :, s].any(axis=1)) for s in range(self.n_states)]

    # ------------------------------------------------------------------------------------
    # Hilfsfunktionen
    # ------------------------------------------------------------------------------------
    def cell(self, s: int) -> str:
        """Welches Zeichen steht auf dem Feld von Zustand s?"""
        r, c = self.states[s]
        return self.grid[r][c]

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """Einen Vektor (ein Wert pro Zustand) zurück auf die 2D-Karte legen. Wände = NaN."""
        out = np.full(self.shape, np.nan)
        for s, (r, c) in enumerate(self.states):
            out[r, c] = values[s]
        return out

    def step(self, s: int, a: int, rng: np.random.Generator | None = None) -> tuple[int, float]:
        """Einen einzelnen Schritt ausprobieren, wie ``env.step`` in den Folien.

        DP braucht das nicht, weil es direkt mit P und R rechnet. Zum Simulieren ist es aber praktisch.
        """
        if self.terminal[s]:
            return s, 0.0
        rng = rng or np.random.default_rng()
        outcomes = self._action_outcomes(a)  # z. B. [(geradeaus, 0.9), (links, 0.05), (rechts, 0.05)]
        actual = outcomes[rng.choice(len(outcomes), p=[p for _, p in outcomes])][0]
        nxt, reward = self._outcome(self.states[s], actual)
        return self.index[nxt], reward

    def _find(self, ch: str) -> tuple[int, int]:
        """Position eines Buchstabens in der Karte (muss genau einmal vorkommen)."""
        hits = [(r, c) for r, row in enumerate(self.grid) for c, x in enumerate(row) if x == ch]
        if len(hits) != 1:
            raise ValueError(f"Zeichen {ch!r} muss genau einmal in der Karte vorkommen")
        return hits[0]

    def _move(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], bool]:
        """Ein Feld in Richtung ``action`` gehen. Gibt (neue Position, gegen Wand gelaufen?) zurück."""
        dr, dc = _DELTAS[action]
        r, c = pos[0] + dr, pos[1] + dc
        outside = not (0 <= r < self.shape[0] and 0 <= c < self.shape[1])
        if outside or self.grid[r][c] == WALL:
            return pos, True  # stehen bleiben
        return (r, c), False

    def _outcome(self, pos: tuple[int, int], action: int) -> tuple[tuple[int, int], float]:
        """Wohin führt ``action`` wirklich, und welche Belohnung gibt es dafür?"""
        if pos in self._teleports:  # Teleporter: egal welche Aktion
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
        """Welche Richtung passiert tatsächlich, mit welcher Wahrscheinlichkeit? (Rutschen)"""
        if self.slip == 0:
            return [(action, 1.0)]  # immer genau die gewünschte Richtung
        half = self.slip / 2
        # (action + 1) % 4 = 90° nach rechts gedreht, (action - 1) % 4 = 90° nach links gedreht
        return [(action, 1.0 - self.slip), ((action + 1) % 4, half), ((action - 1) % 4, half)]

    def _build_model(self) -> tuple[np.ndarray, np.ndarray]:
        """Die Tabellen P[s, a, s'] und R[s, a] für alle Felder und Aktionen ausfüllen."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for s, pos in enumerate(self.states):
            if self.terminal[s]:
                # Terminale Felder sind "absorbierend": Man bleibt für immer und bekommt 0.
                P[s, :, s] = 1.0
                continue
            for a in range(self.n_actions):
                for actual, prob in self._action_outcomes(a):
                    nxt, reward = self._outcome(pos, actual)
                    P[s, a, self.index[nxt]] += prob  # Wahrscheinlichkeiten aufsummieren
                    R[s, a] += prob * reward  # erwartete (= durchschnittliche) Belohnung
        return P, R
