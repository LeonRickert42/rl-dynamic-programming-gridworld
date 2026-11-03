import numpy as np
import pytest

from rldp import (
    METHODS,
    Policy,
    exact_policy_evaluation,
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    value_iteration,
)
from rldp.maps import maze, small_gridworld, sutton_5x5


@pytest.mark.parametrize("method", METHODS)
def test_random_policy_small_gridworld_matches_slides(method):
    env = small_gridworld()
    result = policy_evaluation(env, Policy.uniform(env.n_states, env.n_actions), method, theta=1e-8)
    grid = np.round(env.to_grid(result.V))
    expected = np.array([[0, -14, -20, -22], [-14, -18, -20, -20], [-20, -20, -18, -14], [-22, -20, -14, 0]])
    assert result.converged
    np.testing.assert_array_equal(grid, expected)


def test_small_gridworld_first_sweeps_match_slides():
    env = small_gridworld()
    result = policy_evaluation(env, Policy.uniform(env.n_states, env.n_actions), keep_snapshots=True)
    np.testing.assert_allclose(env.to_grid(result.snapshots[2])[0], [0, -1.75, -2, -2])
    np.testing.assert_allclose(env.to_grid(result.snapshots[3])[0], [0, -2.4375, -2.9375, -3], atol=1e-9)


def test_sutton_5x5_random_and_optimal_values():
    env = sutton_5x5()
    v_random = exact_policy_evaluation(env, Policy.uniform(env.n_states, env.n_actions))
    np.testing.assert_allclose(env.to_grid(v_random)[0], [3.3, 8.8, 4.4, 5.3, 1.5], atol=0.05)
    vi = value_iteration(env, theta=1e-10)
    np.testing.assert_allclose(env.to_grid(vi.V)[0], [22.0, 24.4, 22.0, 19.4, 17.5], atol=0.05)


@pytest.mark.parametrize("method", METHODS)
def test_value_iteration_and_policy_iteration_agree_on_stochastic_maze(method):
    env = maze(slip=0.2)
    pi = policy_iteration(env, eval_method="exact")
    vi = value_iteration(env, method, theta=1e-10)
    assert vi.converged
    np.testing.assert_allclose(vi.V, pi.V, atol=1e-6)
    np.testing.assert_allclose(exact_policy_evaluation(env, vi.policy), pi.V, atol=1e-6)


@pytest.mark.parametrize("k", [1, 3])
def test_modified_policy_iteration_reaches_optimum(k):
    env = maze()
    exact = policy_iteration(env, eval_method="exact")
    modified = policy_iteration(env, eval_method="synchronous", eval_max_sweeps=k, theta=1e-10)
    np.testing.assert_allclose(modified.V, exact.V, atol=1e-6)


def test_policy_improvement_reports_changes_and_stability():
    env = maze()
    random_policy = Policy.uniform(env.n_states, env.n_actions)
    improved, n_changed = policy_improvement(env, exact_policy_evaluation(env, random_policy), random_policy)
    assert n_changed == len(env.nonterminal_states)
    optimal = policy_iteration(env, eval_method="exact")
    _, n_changed = policy_improvement(env, optimal.V, optimal.policy)
    assert n_changed == 0


def test_prioritized_sweeping_needs_fewer_backups_for_value_iteration():
    env = maze(slip=0.0)
    sync = value_iteration(env, "synchronous", theta=1e-8)
    prio = value_iteration(env, "prioritized", theta=1e-8)
    np.testing.assert_allclose(prio.V, sync.V, atol=1e-6)
    assert prio.total_backups < sync.total_backups
