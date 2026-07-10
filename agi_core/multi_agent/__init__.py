"""Parallel multi-agent system with distributed cognition.

By default, the ``AgentSupervisor`` replaces ``ObjectiveReduction`` as the
top-level decision-maker, managing 4 parallel strategy agents: Movement,
Exploration, Interaction, and Puzzle. Set ``AGI_MULTI_AGENT_ENABLE=0`` to
disable it.

Each agent has its own private cognition bundle (WorkingMemory, GoalStack,
AttentionFocus) while sharing ConsolidatedMemory and ChunkLibrary.
A consensus module aggregates proposals, resolves conflicts, and selects
the final action each frame.
"""

from .supervisor import AgentSupervisor
from .strategy_agent import StrategyAgent
from .agents import MovementAgent, ExplorationAgent, InteractionAgent, PuzzleAgent
from .consensus import ConsensusModule

__all__ = [
    "AgentSupervisor",
    "ConsensusModule",
    "ExplorationAgent",
    "InteractionAgent",
    "MovementAgent",
    "PuzzleAgent",
    "StrategyAgent",
]
