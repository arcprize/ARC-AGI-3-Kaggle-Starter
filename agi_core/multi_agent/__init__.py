"""Parallel multi-agent system with distributed cognition.

ORCH-OR remains the default top-level decision-maker. Enable the
``AgentSupervisor`` research path with ``AGI_MULTI_AGENT_ENABLE=1`` to run
the parallel Movement, Exploration, Interaction, and Puzzle agents.

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
