"""Replace `vendor/ARC-AGI-3-Agents/agents/__init__.py` with a minimal version.

The upstream `__init__.py` eagerly imports every LLM-backed agent template
(langgraph, langsmith, smolagents, openai, ...). We don't need any of them
for local development with the random / user agent, and they'd force us to
install heavy deps we'll never use.

This is the *exact same* slimming trick the official Kaggle sample
("ARC3 Sample Submission - Stochastic Goose") performs on Kaggle at run time
— we just do it once locally so `make play-local` works out of the box.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "__init__.py"

SLIM = '''\
"""Slimmed by scripts/slim_framework.py — only random + user agent registered."""
from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
}
'''


def main() -> None:
    if not INIT.exists():
        raise SystemExit(f"Framework not found at {INIT}. Run `make setup` first.")
    INIT.write_text(SLIM)
    print(f"[slim_framework] Slimmed {INIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
