"""Compositional dormancy — agent scaffold.

A benchmark harness that measures whether a tool-calling agent will (1) launder attacker-controlled data from a
tool pipeline into its own memory, (2) later act on that poisoned memory for a benign user request, and (3) do so
undetected by content, trajectory, and origin-binding defenses — while the provenance-authority defense catches it.

Every payment is simulated. See the spec doc 'Compositional dormancy — agent scaffold spec'.
"""
from .harness import run_trial
from .metrics import summarize, summarize_by_cell

__all__ = ["run_trial", "summarize", "summarize_by_cell"]
