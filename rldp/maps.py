"""Vordefinierte Gitterwelten."""

from .gridworld import GridWorld


def small_gridworld() -> GridWorld:
    """4x4 Small Gridworld aus Silver Lecture 3 / Sutton Abb. 4.1 (γ = 1, Belohnung -1 pro Schritt)."""
    return GridWorld(
        ["T...", "....", "....", "...T"],
        gamma=1.0,
        step_reward=-1.0,
        name="Small Gridworld 4x4",
    )


def sutton_5x5() -> GridWorld:
    """5x5 Gitter mit Teleportern A->A' (+10) und B->B' (+5), Sutton Abb. 3.2 / Pareigis-Folien."""
    return GridWorld(
        [".A.B.", ".....", "...b.", ".....", ".a..."],
        gamma=0.9,
        step_reward=0.0,
        bump_reward=-1.0,
        teleports={"A": ("a", 10.0), "B": ("b", 5.0)},
        name="Sutton 5x5 (A/B)",
    )


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
    """Labyrinth mit Wänden, Ziel G (+10) und Falle X (-10); optional rutschig."""
    return GridWorld(
        MAZE,
        gamma=gamma,
        step_reward=-1.0,
        goal_reward=10.0,
        trap_reward=-10.0,
        slip=slip,
        name=f"Labyrinth (slip={slip}, γ={gamma})",
    )
