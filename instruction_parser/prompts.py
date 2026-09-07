SYSTEM_PREAMBLE = (
    "You control a modular synthesizer via a set of continuous CV "
    "(control voltage) channels, each in the range [0, 1]. Given an "
    "instruction describing a desired sound, choose a value for every "
    "channel listed below to best match the instruction. Channels and "
    "what each one controls:\n"
)


def build_system_prompt(channels: dict[str, str]) -> str:
    """Build the shared system prompt describing the CV channels a model
    must fill in. Used by both AnthropicInstructionParser and
    LocalInstructionParser -- this function is plain string-building with
    no provider-specific code, so it lives in its own module rather than
    being imported from one provider's module into the other's."""
    lines = [SYSTEM_PREAMBLE]
    for name, description in channels.items():
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)
