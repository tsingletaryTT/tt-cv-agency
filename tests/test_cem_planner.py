import numpy as np
from cem_planner import cem_plan


def additive_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return states + actions


def test_cem_plan_converges_to_correct_action_single_step_toy_dynamics():
    rng = np.random.default_rng(0)
    current_state = np.array([0.0, 0.0])
    # current_cv chosen so the goal's exact delta (+1.0 / -0.5) is right at
    # the edge of real CV headroom (channel 0: 0.0 -> 1.0; channel 1:
    # 0.5 -> 0.0) rather than clipped away by the cumulative-CV-headroom
    # check cem_plan now applies -- this test is about the search finding
    # the correct action within its real bounds, not about the bounds
    # themselves.
    current_cv = np.array([0.0, 0.5])
    goal_state = np.array([1.0, -0.5])

    action = cem_plan(
        current_state, current_cv, goal_state, predict_fn=additive_predict_fn,
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
    # current_cv=0.0 gives a full [0,1] of real headroom -- well above
    # max_action=0.5, so the cumulative-CV-headroom clip stays out of the
    # way and this test still isolates the search dynamics themselves.
    current_cv = np.array([0.0])
    goal_state = np.array([0.9])

    action = cem_plan(
        current_state, current_cv, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=3, n_candidates=500, n_elite=50,
        n_iterations=8, action_std_init=0.5, max_action=0.5, rng=rng,
    )
    assert action.shape == (1,)
    assert action[0] > 0.1  # meaningfully toward the goal, not stalled/backward


def test_cem_plan_returns_near_zero_action_when_already_at_goal():
    rng = np.random.default_rng(2)
    current_state = np.array([0.5, 0.5])
    current_cv = np.array([0.5, 0.5])
    goal_state = np.array([0.5, 0.5])

    action = cem_plan(
        current_state, current_cv, goal_state, predict_fn=additive_predict_fn,
        action_dim=2, horizon=1, n_candidates=300, n_elite=30,
        n_iterations=5, action_std_init=0.3, max_action=1.0, rng=rng,
    )
    assert np.allclose(action, [0.0, 0.0], atol=0.1)


def test_cem_plan_respects_max_action_clipping():
    rng = np.random.default_rng(3)
    current_state = np.array([0.0])
    current_cv = np.array([0.5])  # plenty of headroom -- isolates the
    # max_action clip from the cumulative-CV-headroom clip tested below.
    goal_state = np.array([100.0])  # unreachable in one bounded step

    action = cem_plan(
        current_state, current_cv, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=1, n_candidates=200, n_elite=20,
        n_iterations=5, action_std_init=1.0, max_action=0.3, rng=rng,
    )
    assert action[0] <= 0.3 + 1e-6


def test_cem_plan_respects_cumulative_cv_headroom_across_horizon():
    # current_cv starts at 0.95 -- only 0.05 of real headroom before
    # hitting the physical CV ceiling of 1.0. A final-review pass found
    # cem_plan previously had no way to know this: at max_action=0.5 over
    # a 3-step horizon, an unconstrained candidate could ask for up to 1.5
    # of cumulative movement on this one channel. The returned (real,
    # post-clip) first action must never exceed the real headroom,
    # regardless of how hard the goal pulls toward the boundary.
    rng = np.random.default_rng(4)
    current_state = np.array([0.0])
    current_cv = np.array([0.95])
    goal_state = np.array([100.0])

    action = cem_plan(
        current_state, current_cv, goal_state, predict_fn=additive_predict_fn,
        action_dim=1, horizon=3, n_candidates=200, n_elite=20,
        n_iterations=5, action_std_init=1.0, max_action=0.5, rng=rng,
    )
    assert action[0] <= 0.05 + 1e-6


def test_cem_plan_clips_rolled_states_before_next_horizon_step():
    # exploding_predict_fn amplifies the action's effect 10x, so an
    # unclipped rollout would walk well outside [0, 1] within a couple of
    # horizon steps -- used only to prove cem_plan's internal state
    # clipping actually engages between horizon steps, not to model
    # anything real.
    recorded_input_states = []

    def exploding_predict_fn(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        recorded_input_states.append(states.copy())
        return states + actions * 10.0

    rng = np.random.default_rng(5)
    current_state = np.array([0.5])
    current_cv = np.array([0.5])
    goal_state = np.array([0.9])
    horizon = 3

    cem_plan(
        current_state, current_cv, goal_state, predict_fn=exploding_predict_fn,
        action_dim=1, horizon=horizon, n_candidates=50, n_elite=5,
        n_iterations=2, action_std_init=0.5, max_action=0.5, rng=rng,
    )

    # Every horizon step except each iteration's first (index 0, horizon,
    # 2*horizon, ...) receives whatever cem_plan fed forward from the
    # PRIOR step's real output -- which must have been clipped to [0, 1]
    # even though exploding_predict_fn's raw output would not be.
    for i, states in enumerate(recorded_input_states):
        if i % horizon != 0:
            assert np.all(states >= 0.0) and np.all(states <= 1.0)


def test_cem_plan_converges_at_production_scale_dimensionality():
    # A final-review pass found cem_plan's own tests previously validated
    # correctness only at 1-2 search dimensions with a far larger relative
    # sample budget than production ever uses. This exercises something
    # close to trajectory_control_loop.py's real horizon=3, action_dim=8
    # (24-dim) scale, confirming the search still finds a directionally
    # and magnitude-correct plan there, not just at toy scale.
    rng = np.random.default_rng(6)
    action_dim = 8
    horizon = 3
    current_state = np.zeros(action_dim)
    current_cv = np.full(action_dim, 0.5)
    goal_state = np.full(action_dim, 0.5)  # additive dynamics: need 0.5 of
    # total movement per dim, spread across 3 steps -- ~0.167/step average.

    action = cem_plan(
        current_state, current_cv, goal_state, predict_fn=additive_predict_fn,
        action_dim=action_dim, horizon=horizon, n_candidates=800, n_elite=80,
        n_iterations=6, action_std_init=0.3, max_action=0.3, rng=rng,
    )
    assert action.shape == (action_dim,)
    # Meaningfully positive on every dimension, not just some, and not
    # dominated by noise the way the pre-fix budget was found to be.
    assert np.all(action > 0.05)
