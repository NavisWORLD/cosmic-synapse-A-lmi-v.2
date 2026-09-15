"""A-LMI core public API.

LightToken utilities are dependency-light. The Kafka-backed ``ALMIAgent`` is
loaded only when requested so importing the core package does not force the
infrastructure extra.
"""

from .light_token import LightToken, resonance_match, spectral_similarity

__all__ = ["LightToken", "spectral_similarity", "resonance_match", "ALMIAgent"]


def __getattr__(name):
    if name == "ALMIAgent":
        from .agent import ALMIAgent

        return ALMIAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
