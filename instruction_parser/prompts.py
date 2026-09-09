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


GOAL_SYSTEM_PREAMBLE = (
    "You listen to a modular synthesizer and translate a description of "
    "a desired sound into a target for 3 measured audio qualities, each "
    "reported as a [mean, std] pair over a several-second listening "
    "window, every value in [0, 1]:\n"
    "- loudness (loud_mean, loud_std): overall level. High mean = loud, "
    "low mean = quiet/near-silent. High std = the loudness noticeably "
    "swells or drops within the window (e.g. a pumping or gated sound); "
    "low std = a steady, unchanging level.\n"
    "- brightness (bright_mean, bright_std): tonal brightness/harshness "
    "(spectral centroid). High mean = bright/harsh/open-filter, low mean "
    "= dark/muffled/closed-filter. High std = the brightness is actively "
    "sweeping or moving (e.g. a filter sweep, a wah-like motion); low "
    "std = a fixed tone color.\n"
    "- pitch (pitch_mean, pitch_std): perceived fundamental pitch, on a "
    "log scale from low (0) to high (1). High std = the pitch is "
    "actively moving (a sequence, an arpeggio, vibrato, a pitch sweep); "
    "low std = a held, steady pitch.\n"
    "Respond with a target value in [0, 1] for each of the 6 fields. "
    "Example: 'a steady, dark, low drone' -> low pitch_mean, low "
    "bright_mean, low std on everything. 'a bright, sweeping, squelchy "
    "sequence' -> high bright_mean AND high bright_std (the sweep), high "
    "pitch_std (the moving sequence).\n"
    "Calibration note: in practice, std values for this kind of instrument "
    "rarely exceed about 0.3 -- a std around 0.15 already represents a "
    "strongly moving/sweeping quality, and 0.02 is essentially steady. "
    "Avoid requesting std values much above 0.3 unless you specifically "
    "intend an extreme, likely-unreachable target."
)


def build_goal_system_prompt() -> str:
    """Fixed system prompt for goal-parsing (unlike build_system_prompt,
    this never varies per patch -- the 6 feature dimensions it describes
    are the same regardless of which instrument is running, since they
    describe the SOUND, not any particular patch's CV channels)."""
    return GOAL_SYSTEM_PREAMBLE
