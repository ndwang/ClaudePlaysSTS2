"""All LLM prompts for the STS2 agent client."""

from i18n import t


def system_prompt() -> str:
    return t("prompt.system")


def reflection_prompt() -> str:
    return t("prompt.reflection")


def summarization_prompt() -> str:
    return t("prompt.summarization")
