# instruction_to_goal.py
import argparse

import numpy as np

from backends.vcv_rack import VCVRackBackend
from instruction_parser.anthropic_parser import AnthropicInstructionParser
from instruction_parser.local_parser import LocalInstructionParser
from instruction_parser.schema import GOAL_DIMS
from trajectory_control_loop import run_trajectory_control_loop


def main():
    parser = argparse.ArgumentParser(
        description="Parse a natural-language instruction into a target sound "
        "(a goal in feature space), then steer the running VCV Rack patch "
        "toward it over time via the trajectory predictor + CEM planner."
    )
    parser.add_argument("instruction", help="e.g. 'make it bright and steady'")
    parser.add_argument("--config", default="configs/sequencer_test.yaml")
    parser.add_argument("--llm", choices=["anthropic", "local"], default="anthropic")
    parser.add_argument(
        "--model",
        default=None,
        help="model name/ID; defaults to claude-opus-5 for --llm anthropic, "
        "required (no default) for --llm local",
    )
    parser.add_argument("--base-url", default=None, help="required for --llm local")
    parser.add_argument(
        "--weights-path", default="data/sequencer_trajectory_model_weights.npz"
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()

    if args.llm == "anthropic":
        instruction_parser = AnthropicInstructionParser(model=args.model or "claude-opus-5")
    else:
        if not args.base_url or not args.model:
            parser.error("--base-url and --model are required when --llm local")
        instruction_parser = LocalInstructionParser(base_url=args.base_url, model=args.model)

    goal_dict = instruction_parser.parse_goal(args.instruction)
    goal_features = np.array([goal_dict[name] for name in GOAL_DIMS])

    backend = VCVRackBackend(args.config)
    try:
        # Import here, not at module top level, to avoid touching hardware
        # without a gozer lease when --help or other early-exit paths run.
        # This matches the pattern in control_loop.py and trajectory_control_loop.py.
        from tt_trajectory_inference import TrajectoryTTInferenceEngine

        engine = TrajectoryTTInferenceEngine(
            weights_path=args.weights_path, expected_channels=backend.channels(),
        )
        try:
            print(f"instruction: {args.instruction!r}")
            print("goal:")
            for name, value in zip(GOAL_DIMS, goal_features):
                print(f"  {name:<12} {value:.3f}")

            history = run_trajectory_control_loop(
                backend, predict_fn=engine.predict_next_state, goal_features=goal_features,
                max_iterations=args.max_iterations,
            )
            print(f"final measured features: {history[-1]}, goal: {goal_features}")
        finally:
            engine.close()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
