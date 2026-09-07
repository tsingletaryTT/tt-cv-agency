# instruction_to_preset.py
import argparse
import time

import numpy as np

from backends.vcv_rack import VCVRackBackend
from features import read_aggregated_window
from instruction_parser.anthropic_parser import AnthropicInstructionParser
from instruction_parser.local_parser import LocalInstructionParser


def main():
    parser = argparse.ArgumentParser(
        description="Parse a natural-language instruction into a CV recipe, "
        "apply it to the running VCV Rack patch, and report measured audio."
    )
    parser.add_argument("instruction", help="e.g. 'make a squelchy resonant acid bass'")
    parser.add_argument("--config", default="configs/sequencer_test.yaml")
    parser.add_argument("--llm", choices=["anthropic", "local"], default="anthropic")
    parser.add_argument("--model", default=None, help="model name/ID; provider-specific default if omitted")
    parser.add_argument("--base-url", default=None, help="required for --llm local")
    parser.add_argument("--settle-time-s", type=float, default=1.0)
    parser.add_argument("--aggregate-window-s", type=float, default=5.0)
    args = parser.parse_args()

    if args.llm == "anthropic":
        instruction_parser = AnthropicInstructionParser(model=args.model or "claude-opus-5")
    else:
        if not args.base_url:
            parser.error("--base-url is required when --llm local")
        instruction_parser = LocalInstructionParser(base_url=args.base_url, model=args.model)

    backend = VCVRackBackend(args.config)
    try:
        channels = backend.channel_descriptions()
        recipe = instruction_parser.parse_recipe(args.instruction, channels)

        for channel, value in recipe.items():
            backend.set_cv(channel, value)
        time.sleep(args.settle_time_s)

        measured = read_aggregated_window(backend, backend.sample_rate(), args.aggregate_window_s)

        print(f"instruction: {args.instruction!r}")
        print("recipe:")
        for channel, value in recipe.items():
            print(f"  {channel:<20} {value:.3f}")
        print(
            "measured [mean, std] x [loudness, brightness, pitch]:\n  "
            f"{np.array2string(measured, precision=4)}"
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
