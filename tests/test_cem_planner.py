import numpy as np
from cem_planner import cem_plan


def additive_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return states + actions


def test_cem_plan_converges_to_correct_action_single_step_toy_dynamics():
    rng = np.random.default_rng(0)
    current_state = np.array([0.0, 0.0])
    goal_state = np.array([1.0, -0.5])

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=2, horizon=1, n_candidates=500, n_elite=50,
        n_iterations=8, action_std_init=1.0, max_action=2.0, rng=rng,
    )
    assert action.shape == (2,)
    assert np.allclose(action, [1.0, -0.5], atol=0.15)


def test_cem_plan_multi_step_horizon_moves_meaningfully_toward_goal():
    # horizon=3, additive dynamics -- any split of the goal-reaching delta
    # across 3 steps scores equally well under this toy dynamics, so this
    # checks the first returned action moves state meaningfully toward the
    # goal (not away from it, not near-zero) -- the property an MPC-style
    # caller actually depends on at every real control step.
    rng = np.random.default_rng(1)
    current_state = np.array([0.0])
    goal_state = np.array([0.9])

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=3, n_candidates=500, n_elite=50,
        n_iterations=8, action_std_init=0.5, max_action=0.5, rng=rng,
    )
    assert action.shape == (1,)
    assert action[0] > 0.1  # meaningfully toward the goal, not stalled/backward


def test_cem_plan_returns_near_zero_action_when_already_at_goal():
    rng = np.random.default_rng(2)
    current_state = np.array([0.5, 0.5])
    goal_state = np.array([0.5, 0.5])

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=2, horizon=1, n_candidates=300, n_elite=30,
        n_iterations=5, action_std_init=0.3, max_action=1.0, rng=rng,
    )
    assert np.allclose(action, [0.0, 0.0], atol=0.1)


def test_cem_plan_respects_max_action_clipping():
    rng = np.random.default_rng(3)
    current_state = np.array([0.0])
    goal_state = np.array([100.0])  # unreachable in one bounded step

    action = cem_plan(
        current_state, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=1, n_candidates=200, n_elite=20,
        n_iterations=5, action_std_init=1.0, max_action=0.3, rng=rng,
    )
    assert action[0] <= 0.3 + 1e-6
