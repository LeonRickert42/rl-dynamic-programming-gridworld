"""Fertige Spielfelder zum Ausprobieren."""

from .gridworld import GridWorld


def small_gridworld() -> GridWorld:
    """4x4-Gitter aus der Vorlesung (Silver Lecture 3, Sutton Abb. 4.1).

    Zwei Ausgänge T (oben links und unten rechts), jeder Schritt kostet -1, γ = 1.
    Der Wert eines Feldes ist dann: minus (erwartete Anzahl Schritte bis zum Ausgang).
    """
    return GridWorld(
        ["T...",
         "....",
         "....",
         "...T"],
        gamma=1.0,
        step_reward=-1.0,
        name="Small Gridworld 4x4",
    )


def sutton_5x5() -> GridWorld:
    """5x5-Gitter mit zwei Teleportern (Sutton Abb. 3.2, Pareigis-Folien).

    Jede Aktion auf A beamt nach a und bringt +10, jede Aktion auf B beamt nach b und bringt +5.
    Gegen den Rand laufen kostet -1, alles andere 0.
    """
    return GridWorld(
        [".A.B.",
         ".....",
         "...b.",
         ".....",
         ".a..."],
        gamma=0.9,
        step_reward=0.0,
        bump_reward=-1.0,
        teleports={"A": ("a", 10.0), "B": ("b", 5.0)},
        name="Sutton 5x5 (A/B)",
    )


def corridor(length: int = 8, gamma: float = 0.9) -> GridWorld:
    """Ein einfacher Flur: ganz rechts das Ziel. Ideal, um die Ausbreitung der Werte zu sehen."""
    return GridWorld(["." * (length - 1) + "G"], gamma=gamma, step_reward=-1.0, goal_reward=10.0,
                     name=f"Flur (Länge {length})")


# Ein Labyrinth: S = Start, # = Wand, X = Falle, G = Ziel.
MAZE = [
    "S...#.....",
    ".##.#.##.#",
    ".#..#..#..",
    ".#.###.#.#",
    "...#...#..",
    "##.#.#.##.",
    "...X.#....",
    ".#...#.#.G",
]


def maze(slip: float = 0.1, gamma: float = 0.95) -> GridWorld:
    """Labyrinth mit Ziel G (+10) und Falle X (-10). Mit ``slip`` > 0 ist der Boden rutschig."""
    return GridWorld(
        MAZE,
        gamma=gamma,
        step_reward=-1.0,
        goal_reward=10.0,
        trap_reward=-10.0,
        slip=slip,
        name=f"Labyrinth (slip={slip}, γ={gamma})",
    )
